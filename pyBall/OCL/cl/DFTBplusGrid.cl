// OpenCL kernel for DFTB+ waveplot - Slater-type orbital projection
// Based on DFTB+ waveplot.F90 implementation

#ifndef DEBUG_EARLY_EXIT
#define DEBUG_EARLY_EXIT 0
#endif

#ifndef DEBUG_CLEAR_ONLY
#define DEBUG_CLEAR_ONLY 0
#endif

#ifndef DEBUG_RETURN0
#define DEBUG_RETURN0 0
#endif

#ifndef DEBUG_READ_TASK
#define DEBUG_READ_TASK 0
#endif

#ifndef DEBUG_READ_GRID
#define DEBUG_READ_GRID 0
#endif

typedef struct {
    float4 origin;
    float4 dA;
    float4 dB;
    float4 dC;
    int4 ngrid;
} GridSpec;

typedef struct {
    float4 pos_rcut; // x, y, z, Rcut
    int type;        // index into basis data
    int i0orb;       // start index in global orbital list
    int norb;        // number of orbitals
    int pad;
} AtomData;

typedef struct {
    int x, y, z, w;
    int na;
    int nj;
    int pad1;
    int pad2;
} TaskData;

// Real spherical harmonic normalization
#define PREF_S 0.28209479f   // 1/sqrt(4*pi)
#define PREF_P 0.48860251f   // sqrt(3/(4*pi))

// Cubic B-spline interpolation for radial part
float evaluate_radial_sto(
    float r,
    int ityp,
    int ish,
    __global const float* basis_data,
    int n_nodes,
    float dr,
    int max_shells
) {
    if (ityp < 0) return 0.0f;
    if (ish < 0) return 0.0f;
    if (ish >= max_shells) return 0.0f;
    if (r >= (n_nodes - 1) * dr) return 0.0f;
    
    const __global float2* basis2 = (const __global float2*)basis_data;
    
    const float x = r / dr;
    int i = (int)floor(x);
    if (i < 0) i = 0;
    if (i > (n_nodes - 2)) i = (n_nodes - 2);
    const float t = x - (float)i;
    
    const int base = (ityp * max_shells + ish) * n_nodes;
    const float2 lo = basis2[base + i];
    const float2 hi = basis2[base + i + 1];
    
    const float a = 1.0f - t;
    const float b = t;
    const float h2_6 = (dr * dr) * (1.0f/6.0f);
    const float corr = ((a*a*a - a) * lo.y + (b*b*b - b) * hi.y) * h2_6;
    return a * lo.x + b * hi.x + corr;
}

// Real spherical harmonics for l=0,1
float real_spherical_harmonic(int l, int m, float3 rvec, float r) {
    if (r < 1e-10f) {
        // At origin, only l=0,m=0 is non-zero
        if (l == 0 && m == 0) return PREF_S;
        return 0.0f;
    }
    
    float x = rvec.x / r;
    float y = rvec.y / r;
    float z = rvec.z / r;
    
    if (l == 0) {
        // s orbital
        return PREF_S;
    } else if (l == 1) {
        // p orbitals
        if (m == -1) return PREF_P * y;  // py
        if (m == 0)  return PREF_P * z;  // pz
        if (m == 1)  return PREF_P * x;  // px
    }
    
    return 0.0f;
}

__kernel void project_orbital_dftb(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,
    __global const float* coeffs,
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    const int nMaxAtom,
    __global float* out_grid
) {
    const int gid = get_global_id(0);
    const int threads_per_task = get_local_size(0);
    const int i_task = get_group_id(0);
    const int t_idx = get_local_id(0);
    
    if (i_task >= n_tasks) return;
    
    const TaskData task = tasks[i_task];
    const int na = task.na;
    
    // Process 512 voxels per task (8x8x8 block)
    for (int v = t_idx; v < 512; v += threads_per_task) {
        float3 r_vox;
        int g_idx;
        const int lx = v & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        
        {
            const int gx = task.x * 8 + lx;
            const int gy = task.y * 8 + ly;
            const int gz = task.z * 8 + lz;
            const int3 ngrid_dim = grid->ngrid.xyz;
            if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
            g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
            r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;
        }
        
        float psi = 0.0f;
        
        // Loop over atoms in this task
        for (int i = 0; i < na; i++) {
            const int i_atom = task_atoms[i_task * nMaxAtom + i];
            AtomData ad_i = atoms[i_atom];
            float rcut_i2 = ad_i.pos_rcut.w;
            rcut_i2 *= rcut_i2;
            
            float3 dri;
            dri = r_vox - ad_i.pos_rcut.xyz;
            float ri2 = dri.x*dri.x + dri.y*dri.y + dri.z*dri.z;
            
            if (ri2 >= rcut_i2) continue;
            
            float ri = sqrt(ri2);
            int ityp = ad_i.type;
            int i0orb = ad_i.i0orb;
            int norb = ad_i.norb;
            
            // Loop over orbitals of this atom
            // DFTB+ ordering: for each shell (l), orbitals are m = -l, ..., l
            int orb_idx = i0orb;
            int ish = 0;
            
            // s orbital (l=0, m=0)
            if (norb > 0) {
                float R_s = evaluate_radial_sto(ri, ityp, ish, basis_data, n_nodes, dr_basis, max_shells);
                float Y_s = real_spherical_harmonic(0, 0, dri, ri);
                psi += coeffs[orb_idx] * R_s * Y_s;
                orb_idx++;
                ish++;
            }
            
            // p orbitals (l=1, m=-1,0,1)
            if (norb > 1) {
                float R_p = evaluate_radial_sto(ri, ityp, ish, basis_data, n_nodes, dr_basis, max_shells);
                
                // py (m=-1)
                float Y_py = real_spherical_harmonic(1, -1, dri, ri);
                if (orb_idx < i0orb + norb) psi += coeffs[orb_idx] * R_p * Y_py;
                orb_idx++;
                
                // pz (m=0)
                float Y_pz = real_spherical_harmonic(1, 0, dri, ri);
                if (orb_idx < i0orb + norb) psi += coeffs[orb_idx] * R_p * Y_pz;
                orb_idx++;
                
                // px (m=1)
                float Y_px = real_spherical_harmonic(1, 1, dri, ri);
                if (orb_idx < i0orb + norb) psi += coeffs[orb_idx] * R_p * Y_px;
                orb_idx++;
                
                ish++;
            }
            
            // d orbitals (l=2) - could add later if needed
        }
        
        out_grid[g_idx] = psi;
    }
}

// ============================================================================
// Bloch orbital / Tersoff-Hamann slice on an explicit list of points.
// One work-item = one point.
//
// psi_s(r) = sum_{mu, cell n} C_s,mu * phi_mu(r - tau - R_n) * exp(-i * 2*pi * k_s · n)
// rho(r)   = sum_s  weight_s * |psi_s(r)|^2
//
// C is the home-cell MO matrix from DFTBcore, packed [state, orb] as float2 (re, im).
// A state is one (molecular orbital, k-point) pair. Orbital order per atom is the
// DFTB+ order: s, then py, pz, px. Cell shifts n are fractional (integer) lattice indices;
// cell_cart is the same shift in Angstrom. Phase sign matches waveplot (dot_product
// conjugates exp(+ikr) down to exp(-ikr)).
// ============================================================================
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void project_bloch_points(
    const int n_points,
    __global const float4* points,
    const int natoms,
    __global const AtomData* atoms,
    const int ncells,
    __global const float4* cell_cart,
    __global const float4* cell_n,
    const int nstate,
    const int norb,
    __global const float2* coeffs,
    __global const float4* k_weight,
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    const int write_psi,
    __global float* out_rho,
    __global float2* out_psi
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    const float TWOPI = 6.28318530718f;
    float rho = 0.0f;

    for (int ist = 0; ist < nstate; ++ist) {
        const float4 kw = k_weight[ist];
        const __global float2* cstate = coeffs + ist * norb;
        float2 psi = (float2)(0.0f, 0.0f);

        for (int ic = 0; ic < ncells; ++ic) {
            const float th = TWOPI * dot(kw.xyz, cell_n[ic].xyz);
            const float2 phase = (float2)(cos(th), -sin(th));
            const float3 R = cell_cart[ic].xyz;

            for (int ia = 0; ia < natoms; ++ia) {
                const AtomData ad = atoms[ia];
                const float3 dri = p - (ad.pos_rcut.xyz + R);
                const float ri2 = dot(dri, dri);
                const float rcut = ad.pos_rcut.w;
                if (ri2 >= rcut * rcut) continue;
                const float ri = sqrt(ri2);

                const int i0 = ad.i0orb;
                const int naorb = ad.norb;
                int orb = i0;
                int ish = 0;

                if (naorb > 0) {
                    const float Rs = evaluate_radial_sto(ri, ad.type, ish, basis_data, n_nodes, dr_basis, max_shells);
                    const float Ys = real_spherical_harmonic(0, 0, dri, ri);
                    const float2 cp = cmul(cstate[orb], phase);
                    psi.x += cp.x * Rs * Ys;
                    psi.y += cp.y * Rs * Ys;
                    orb++;
                    ish++;
                }
                if (naorb > 1) {
                    const float Rp = evaluate_radial_sto(ri, ad.type, ish, basis_data, n_nodes, dr_basis, max_shells);
                    const float Ypy = real_spherical_harmonic(1, -1, dri, ri);
                    const float Ypz = real_spherical_harmonic(1,  0, dri, ri);
                    const float Ypx = real_spherical_harmonic(1,  1, dri, ri);
                    if (orb < i0 + naorb) {
                        const float2 cp = cmul(cstate[orb], phase);
                        psi.x += cp.x * Rp * Ypy;
                        psi.y += cp.y * Rp * Ypy;
                    }
                    orb++;
                    if (orb < i0 + naorb) {
                        const float2 cp = cmul(cstate[orb], phase);
                        psi.x += cp.x * Rp * Ypz;
                        psi.y += cp.y * Rp * Ypz;
                    }
                    orb++;
                    if (orb < i0 + naorb) {
                        const float2 cp = cmul(cstate[orb], phase);
                        psi.x += cp.x * Rp * Ypx;
                        psi.y += cp.y * Rp * Ypx;
                    }
                }
            }
        }
        rho += kw.w * (psi.x * psi.x + psi.y * psi.y);
        if (write_psi) out_psi[ist * n_points + ip] = psi;
    }
    out_rho[ip] = rho;
}
