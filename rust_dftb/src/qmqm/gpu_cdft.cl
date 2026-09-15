// ------------------------------------------------------------------
// gpu_cdft.cl — constrained-DFT augmentation kernels (Dense_Multi_CDFT)
//
// Separate program from gpu_matrix_ops.cl so the CDFT layer stays a
// pure add-on: the only solver-side hook is one enqueue after
// fused_dq_v_hscc_batched inside GpuSccPlan::enq_dq_v_hscc.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// cdft_hscc_shift_batched
//
// Constrained-DFT on-site shift, applied AFTER the fused
// dq→V→H_scc build and BEFORE the eigensolve. For a fragment charge
// constraint Q_F = Σ_{A∈F} Δq_A with multiplier λ_F, the augmented
// Hamiltonian term is
//
//     H[μ,ν] += ½·λ_F·S[μ,ν]·(w_μ + w_ν),   w_μ = 1 iff A(μ) ∈ F
//
// — i.e. V_A → V_A + λ_F·w_A in the usual ½S(V_A+V_B) formula.
// Generalized to nfrag fragments: λw[A] = λ[b][frag[A]] (frag[A]<0 → 0).
//
// The shift enters h_scc itself, so the existing force path (which
// derives W from h_scc eigenpairs) automatically includes the
// constraint force −λ·dQ_F/dR via dS/dR. The energy dot ½Δq·V uses the
// UNSHIFTED V buffer, but e_band picks up the FULL shift contribution
// λ_F·Q_gross(F) (gross Mulliken pop, incl. q0) — the host subtracts
// Σλ_F·(Q_F+Q0_F) to recover E_DFTB of the constrained state.
//
// One workgroup per system; λw cached in __local; strided N² update.
// Gather-in / own-write — no atomics, no cross-WG communication.
// ------------------------------------------------------------------
__kernel void cdft_hscc_shift_batched(
    const int n,
    const int n_atoms,
    const int batch,
    const int nfrag,
    __global const float* S,        // [batch*n*n]
    __global const int* orb_atom,   // [batch*n]
    __global const int* frag,       // [n_atoms] fragment id, -1 = unconstrained
    __global const float* lam,      // [batch*nfrag]
    __global float* H,              // [batch*n*n] h_scc, updated in place
    __local float* lw,              // [n_atoms] λw per atom
    __global const int* active      // [batch] 0 → replica frozen, early-out
) {
    const int sid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    if (sid >= batch || active[sid] == 0) return;
    __global const float* Sb = S + (size_t)sid * n * n;
    __global const int* oa   = orb_atom + (size_t)sid * n;
    __global const float* lb = lam + (size_t)sid * nfrag;
    __global float* Hb       = H + (size_t)sid * n * n;

    // λw[A] = λ_F(A), 0 for unconstrained atoms.
    for (int a = lid; a < n_atoms; a += lsz) {
        const int f = frag[a];
        lw[a] = (f >= 0) ? lb[f] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int nn = n * n;
    for (int idx = lid; idx < nn; idx += lsz) {
        const int i = idx / n;
        const int j = idx - i * n;
        Hb[idx] += 0.5f * Sb[idx] * (lw[oa[i]] + lw[oa[j]]);
    }
}
