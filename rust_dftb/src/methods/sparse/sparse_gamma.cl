// Tiled n-body γ/γ′ kernels — the SCC electrostatics of the sparse path.
//
// V_i   = Σ_j γ(r_ij, u_i, u_j)·dq_j            (gamma_matvec — replaces the
//                                                dense f64 gmat + CPU matvec)
// F_i   = -dq_i·Σ_j γ′(r_ij, u_i, u_j)·dq_j·r̂_ij  (gamma_force — replaces the
//                                                scc_double_counting_force loop)
//
// Gravity-kernel pattern: each work-item owns ONE target atom i (gathers, never
// scatters — no atomics), the work-group stages source j-atoms into __local
// memory in tiles of GWG, each staged atom costs one exp + a few FMAs.
// Coordinates in Å → Bohr inside; u in Hartree; V out in Hartree; F out in
// Hartree/Å (matches scc_double_counting_force's ANG2BOHR² factor).
//
// Exact f32 ports of methods/dftb/gamma.rs::gamma_full and
// methods/dftb/forces.rs::gamma_prime_full (Elstner exponential screening).

#ifndef GWG
#define GWG 128
#endif

#define A2B       1.889726133f   // Å → Bohr
#define G_TAU     3.2f
#define G_C0      0.6875f
#define G_C1      0.1875f
#define G_C2      0.020833333333333333f
#define G_HUBDIFF 1.0e-4f
#define G_RMIN    1.0e-2f        // MIN_NEIGH_DIST (Å) — also excludes j == i
#define G_RTOL    1.0e-10f       // TOL_SAME_DIST (Bohr)

// Short-range screening S(r) for |u1−u2| ≥ MIN_HUB_DIFF: the sum
// sub(tau1,tau2) + sub(tau2,tau1) of gamma.rs::gamma_sub_exprn.
inline float gamma_sub(float r, float t1, float t2) {
    const float dt2 = t1 * t1 - t2 * t2;
    const float dt2_sq = dt2 * dt2;
    const float dt2_cu = dt2_sq * dt2;
    const float t2_2 = t2 * t2, t2_4 = t2_2 * t2_2, t2_6 = t2_4 * t2_2;
    const float ta = 0.5f * t2_4 * t1 / dt2_sq;
    const float tb = (t2_6 - 3.0f * t2_4 * t1 * t1) / (r * dt2_cu);
    return exp(-t1 * r) * (ta - tb);
}

// γ(r, u1, u2) in Hartree — r in Bohr. On-site (r→0) returns the mean U,
// matching gamma_full's r < TOL_SAME_DIST branch (covers j == i).
inline float gamma_eval(float r, float u1, float u2) {
    if (r < G_RTOL) return 0.5f * (u1 + u2);
    float s;
    if (fabs(u1 - u2) < G_HUBDIFF) {
        const float tau = G_TAU * 0.5f * (u1 + u2);
        const float e = exp(-tau * r);
        s = e * (1.0f / r + G_C0 * tau + G_C1 * r * tau * tau + G_C2 * r * r * tau * tau * tau);
    } else {
        s = gamma_sub(r, G_TAU * u1, G_TAU * u2) + gamma_sub(r, G_TAU * u2, G_TAU * u1);
    }
    return 1.0f / r - s;
}

// d/dr of gamma_sub.
inline float gamma_sub_prime(float r, float t1, float t2) {
    const float dt2 = t1 * t1 - t2 * t2;
    const float dt2_sq = dt2 * dt2;
    const float dt2_cu = dt2_sq * dt2;
    const float t2_2 = t2 * t2, t2_4 = t2_2 * t2_2, t2_6 = t2_4 * t2_2;
    const float ta = 0.5f * t2_4 * t1 / dt2_sq;
    const float tb = (t2_6 - 3.0f * t2_4 * t1 * t1) / (r * dt2_cu);
    const float e = exp(-t1 * r);
    // d/dr [e·(ta − tb)] = −t1·e·(ta − tb) + e·(tb/r)   (tb = C/r)
    return -t1 * e * (ta - tb) + e * (tb / r);
}

// γ′(r, u1, u2) = −1/r² − S′(r), per Bohr — gamma_prime_full port.
// Caller guarantees r ≥ G_RMIN·A2B (no on-site call).
inline float gamma_prime_eval(float r, float u1, float u2) {
    float sp;
    if (fabs(u1 - u2) < G_HUBDIFF) {
        const float tau = G_TAU * 0.5f * (u1 + u2);
        const float e = exp(-tau * r);
        const float poly = 1.0f / r + G_C0 * tau + G_C1 * r * tau * tau + G_C2 * r * r * tau * tau * tau;
        const float poly_p = -1.0f / (r * r) + G_C1 * tau * tau + 2.0f * G_C2 * r * tau * tau * tau;
        sp = -tau * e * poly + e * poly_p;
    } else {
        sp = gamma_sub_prime(r, G_TAU * u1, G_TAU * u2) + gamma_sub_prime(r, G_TAU * u2, G_TAU * u1);
    }
    return -1.0f / (r * r) - sp;
}

// V_i = Σ_j γ_ij·dq_j — includes the diagonal (γ(0,u,u) = u_i).
__attribute__((reqd_work_group_size(GWG, 1, 1)))
__kernel void gamma_matvec(
    const uint n,
    __global const float4* xyzu,   // x,y,z [Å], w = Hubbard u [Ha]
    __global const float*  dq,     // Δq_j
    __global float*        v       // V_i [Ha]
){
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);
    // F1 replica axis: dim1 = eval slot — xyzu/v are per-replica strided
    // (JobId = launch index), dq is shared central state (read-only).
    const uint b = get_global_id(1);
    xyzu += (size_t)b * n;
    v    += (size_t)b * n;
    __local float4 tj[GWG];
    __local float  tdq[GWG];
    const float4 mi = (i < n) ? xyzu[i] : (float4)(0.0f, 0.0f, 0.0f, 0.5f);
    float acc = 0.0f;
    for (uint t = 0; t < n; t += GWG) {
        const uint j = t + lid;
        tj[lid]  = (j < n) ? xyzu[j] : (float4)(0.0f, 0.0f, 0.0f, 0.5f);
        tdq[lid] = (j < n) ? dq[j] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < n) {
            #pragma unroll 8
            for (uint k = 0; k < GWG; k++) {
                const float4 mj = tj[k];
                const float3 d = mi.xyz - mj.xyz;
                const float r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z) * A2B;
                acc += gamma_eval(r, mi.w, mj.w) * tdq[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n) v[i] = acc;
}

// F_i = −dq_i·Σ_{j≠i} γ′_ij·dq_j·(r_i−r_j)/|r_i−r_j|  [Ha/Å]
// Pair coefficient matches scc_double_counting_force:
//   coeff = −dq_i·dq_j·γ′(r_bohr)/r_bohr·A2B²,  F_i += coeff·(r_i−r_j)_Å.
__attribute__((reqd_work_group_size(GWG, 1, 1)))
__kernel void gamma_force(
    const uint n,
    __global const float4* xyzu,
    __global const float*  dq,
    __global float4*       f       // out xyz = force [Ha/Å], w unused
){
    const uint i = get_global_id(0);
    const uint lid = get_local_id(0);
    // F1 replica axis: dim1 = eval slot — xyzu/f are per-replica strided,
    // dq is shared central state (read-only).
    const uint b = get_global_id(1);
    xyzu += (size_t)b * n;
    f    += (size_t)b * n;
    __local float4 tj[GWG];
    __local float  tdq[GWG];
    const float4 mi = (i < n) ? xyzu[i] : (float4)(0.0f, 0.0f, 0.0f, 0.5f);
    const float dqi = (i < n) ? dq[i] : 0.0f;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (uint t = 0; t < n; t += GWG) {
        const uint j = t + lid;
        tj[lid]  = (j < n) ? xyzu[j] : (float4)(0.0f, 0.0f, 0.0f, 0.5f);
        tdq[lid] = (j < n) ? dq[j] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < n) {
            #pragma unroll 8
            for (uint k = 0; k < GWG; k++) {
                const float4 mj = tj[k];
                const float dx = mi.x - mj.x;
                const float dy = mi.y - mj.y;
                const float dz = mi.z - mj.z;
                const float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < G_RMIN * G_RMIN) continue;   // j == i and fused pairs
                const float r_b = sqrt(r2) * A2B;
                const float c = -dqi * tdq[k] * gamma_prime_eval(r_b, mi.w, mj.w) / r_b * A2B * A2B;
                fx += c * dx;
                fy += c * dy;
                fz += c * dz;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n) f[i] = (float4)(fx, fy, fz, 0.0f);
}
