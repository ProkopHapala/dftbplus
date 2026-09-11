// ============================================================================
// sparse_bsr4_core.cl
//
// Block-CSR sparse matrices for DFTB with 4 orbitals/atom.
//
// CSR structure:
//
//   row_ptr[i] ... row_ptr[i+1]-1    blocks belonging to atom i
//   col_idx[b]                       neighbor atom j
//   val[16*b + 4*r + c]             A_{i,r ; j,c}
//
// Store BOTH (i,j) and (j,i) blocks for symmetric matrices.
// This costs some memory, but gives simple gather-only GPU kernels.
//
// All calculations are float.
//
// NON-ORTHOGONAL DENSITY KERNEL:
//
//   K S K = K                                             (1)
//
//   Nocc = Tr(K S)                                        (2)
//
// closed shell:
//   Ne = 2 Tr(KS)
//
// Generalized McWeeny:
//
//   K' = 3 K S K - 2 K S K S K                          (3)
//
// Metric TC2:
//
//   Q = K S K
//
//   if Tr(KS) > Nocc:
//       K' = Q
//   else:
//       K' = 2K - Q                                       (4)
//
// Sparse multiplication is always MASKED:
//
//   C = P_M(A B)                                          (5)
//
// C_row/C_col explicitly define which blocks are allowed to exist.
// Blocks outside this mask are NEVER calculated and NEVER allocated.
//
// ============================================================================

#ifndef WG
#define WG 128
#endif

#define BS      4
#define BS2     16

// Natural unit for one 4x4 block:
// 16 threads = one thread/output matrix element.
#define TEAM    16
#define NTEAM   (WG/TEAM)

#ifndef MAX_LEFT_BLOCKS
#define MAX_LEFT_BLOCKS 256
#endif

#ifndef REDUCE_WG
#define REDUCE_WG 256
#endif

#define INVALID_BLOCK 0xffffffffu


// ============================================================================
// Find block (row,col) inside a sorted CSR row.
// Used only by generic sparse multiplication.
// ============================================================================

inline uint bsr4_find(
    __global const uint* row_ptr,
    __global const uint* col_idx,
    uint row,
    uint col
){
    uint lo = row_ptr[row];
    uint hi = row_ptr[row+1];

    while(lo < hi){

        uint mid = (lo + hi) >> 1;
        uint c   = col_idx[mid];

        if(c < col) lo = mid + 1;
        else        hi = mid;
    }

    if(
        lo < row_ptr[row+1]
        && col_idx[lo] == col
    ){
        return lo;
    }

    return INVALID_BLOCK;
}


// ============================================================================
// GENERIC MASKED BLOCK-SPARSE MATRIX PRODUCT
//
//             C_ij = sum_k A_ik B_kj
//
// C_row/C_col define the OUTPUT mask.
//
// GPU decomposition:
//
//     one WG             = one atom row i
//     one 16-thread TEAM = one output 4x4 block C_ij
//     one lane           = one scalar C_ij[r,c]
//
// Entire sparse row A_i* is loaded into local memory ONCE and reused for
// all output neighbors j.
//
// B_kj is found by binary search in row k.
//
// Use this when B is not symmetric.
//
// ============================================================================

__attribute__((reqd_work_group_size(WG,1,1)))
__kernel void bsr4_spgemm_masked(
    const uint nrow,

    __global const uint*  A_row,
    __global const uint*  A_col,
    __global const float* A,

    __global const uint*  B_row,
    __global const uint*  B_col,
    __global const float* B,

    __global const uint*  C_row,
    __global const uint*  C_col,
    __global float*       C
){
    const uint i   = get_group_id(0);
    const uint lid = get_local_id(0);

    if(i >= nrow) return;

    const uint a0 = A_row[i];
    const uint a1 = A_row[i+1];
    const uint na = a1-a0;

    // Uniform branch for whole WG.
    // Better solution: bucket rows and compile 64/128/256/512 variants.
    if(na > MAX_LEFT_BLOCKS) return;


    // ------------------------------------------------------------------------
    // Cache complete left sparse row.
    //
    // MAX_LEFT_BLOCKS=256:
    //
    //   values: 256 * 16 * 4 = 16 KiB
    //   columns: 256 * 4      =  1 KiB
    //
    // ------------------------------------------------------------------------

    __local uint  lcol[MAX_LEFT_BLOCKS];
    __local float lA[MAX_LEFT_BLOCKS*BS2];


    for(uint a=lid; a<na; a+=WG){
        lcol[a] = A_col[a0+a];
    }


    for(uint t=lid; t<na*BS2; t+=WG){
        lA[t] = A[a0*BS2+t];
    }


    barrier(CLK_LOCAL_MEM_FENCE);


    // 16 lanes = 16 scalar entries of one 4x4 block.
    const uint team = lid >> 4;
    const uint lane = lid & 15;

    const uint r = lane >> 2;
    const uint c = lane & 3;


    const uint c0 = C_row[i];
    const uint c1 = C_row[i+1];


    // Each TEAM processes several desired output neighbors.
    for(uint cb=c0+team; cb<c1; cb+=NTEAM){

        const uint j = C_col[cb];

        float sum = 0.0f;


        // --------------------------------------------------------------------
        // C_ij = sum_k A_ik B_kj
        // --------------------------------------------------------------------

        for(uint al=0; al<na; ++al){

            const uint k = lcol[al];

            const uint bb =
                bsr4_find(
                    B_row,
                    B_col,
                    k,
                    j
                );

            if(bb == INVALID_BLOCK) continue;


            __local const float* Ab =
                lA + al*BS2;

            __global const float* Bb =
                B + bb*BS2;


            // 4x4 * 4x4.
            //
            // Each lane computes one C[r,c].
            //
            sum = fma(Ab[4*r+0], Bb[4*0+c], sum);
            sum = fma(Ab[4*r+1], Bb[4*1+c], sum);
            sum = fma(Ab[4*r+2], Bb[4*2+c], sum);
            sum = fma(Ab[4*r+3], Bb[4*3+c], sum);
        }


        C[cb*BS2 + lane] = sum;
    }
}


// ============================================================================
// OPTIMIZED MASKED PRODUCT WHEN RIGHT MATRIX B IS SYMMETRIC
//
//             C_ij = sum_k A_ik B_kj
//
// Since
//
//             B_kj = transpose(B_jk)
//
// we can instead intersect the sorted neighbor lists:
//
//             k in N_A(i) INTERSECTION N_B(j)
//
// and use
//
//             C_ij += A_ik * transpose(B_jk)
//
// This avoids one binary search for every A_ik.
//
// This is especially useful for purification:
//
//      KS      = K*S          S symmetric
//      KSK     = (KS)*K       K symmetric
//      KSKS    = KSK*S        S symmetric
//      KSKSK   = KSKS*K       K symmetric
//
// Notice that A itself DOES NOT need to be symmetric.
//
// ============================================================================

__attribute__((reqd_work_group_size(WG,1,1)))
__kernel void bsr4_spgemm_masked_Bsym(
    const uint nrow,

    __global const uint*  A_row,
    __global const uint*  A_col,
    __global const float* A,

    __global const uint*  B_row,
    __global const uint*  B_col,
    __global const float* B,

    __global const uint*  C_row,
    __global const uint*  C_col,
    __global float*       C
){
    const uint i   = get_group_id(0);
    const uint lid = get_local_id(0);

    if(i >= nrow) return;


    const uint a0 = A_row[i];
    const uint a1 = A_row[i+1];
    const uint na = a1-a0;

    if(na > MAX_LEFT_BLOCKS) return;


    __local uint  lcol[MAX_LEFT_BLOCKS];
    __local float lA[MAX_LEFT_BLOCKS*BS2];


    for(uint a=lid; a<na; a+=WG){
        lcol[a] = A_col[a0+a];
    }


    for(uint t=lid; t<na*BS2; t+=WG){
        lA[t] = A[a0*BS2+t];
    }


    barrier(CLK_LOCAL_MEM_FENCE);


    const uint team = lid >> 4;
    const uint lane = lid & 15;

    const uint r = lane >> 2;
    const uint c = lane & 3;


    const uint c0 = C_row[i];
    const uint c1 = C_row[i+1];


    for(uint cb=c0+team; cb<c1; cb+=NTEAM){

        const uint j = C_col[cb];


        // A row is cached:
        //
        //     lcol[0 ... na)
        //
        // Right matrix row j is global:
        //
        //     B_col[B_row[j] ... B_row[j+1])
        //
        // Both lists are sorted, therefore classic two-pointer intersection.

        uint ia = 0;

        uint ib  = B_row[j];
        uint ib1 = B_row[j+1];


        float sum = 0.0f;


        while(
            ia < na
            && ib < ib1
        ){

            const uint ka = lcol[ia];
            const uint kb = B_col[ib];


            if(ka < kb){

                ++ia;

            }else if(kb < ka){

                ++ib;

            }else{

                // Same atom k appears in both rows.
                //
                // We have A_ik directly.
                //
                // B row j gives B_jk.
                //
                // But product requires B_kj:
                //
                //      B_kj = transpose(B_jk)
                //
                // Hence:
                //
                // C_ij[r,c]
                //    += sum_m A_ik[r,m] * B_jk[c,m]

                __local const float* Ab =
                    lA + ia*BS2;

                __global const float* Bjk =
                    B + ib*BS2;


                sum = fma(Ab[4*r+0], Bjk[4*c+0], sum);
                sum = fma(Ab[4*r+1], Bjk[4*c+1], sum);
                sum = fma(Ab[4*r+2], Bjk[4*c+2], sum);
                sum = fma(Ab[4*r+3], Bjk[4*c+3], sum);


                ++ia;
                ++ib;
            }
        }


        C[cb*BS2 + lane] = sum;
    }
}


// ============================================================================
// P4: SYMBOLIC SpGEMM PLAN — Bsym variant
//
// Precomputed intersection plan for C = P_M(A · B) where B is symmetric.
//
// For each output block C_ij (global block index cb), the plan stores the
// exact list of contributing (A_ik, B_jk) block pairs:
//
//   plan_ptr[cb] .. plan_ptr[cb+1]  ->  range of terms for output block cb
//   plan_a_idx[t]                    ->  local index of A_ik in row i (0-based)
//   plan_b_idx[t]                    ->  global block index of B_jk in B
//
// The kernel loads A's row into local memory (as before), then for each
// output block iterates the plan terms — no intersection, no binary search.
// Only loads + 4x4 FMAs.
//
// Fail-loud: if plan_a_idx[t] >= na (left degree), the kernel writes NaN.
// ============================================================================

__attribute__((reqd_work_group_size(WG,1,1)))
__kernel void bsr4_spgemm_plan_Bsym(
    const uint nrow,

    __global const uint*  A_row,
    __global const float* A,

    __global const float* B,
    __global const uint*  plan_ptr,
    __global const uint*  plan_a_idx,
    __global const uint*  plan_b_idx,

    __global const uint*  C_row,
    __global float*       C
){
    const uint i   = get_group_id(0);
    const uint lid = get_local_id(0);

    if(i >= nrow) return;

    const uint a0 = A_row[i];
    const uint a1 = A_row[i+1];
    const uint na = a1-a0;

    if(na > MAX_LEFT_BLOCKS) return;

    // Cache complete left sparse row values into local memory.
    // GPT-5.6 #5: removed lcol[] — never used (plan_a_idx indexes lA directly).
    __local float lA[MAX_LEFT_BLOCKS*BS2];

    for(uint t=lid; t<na*BS2; t+=WG){
        lA[t] = A[a0*BS2+t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint team = lid >> 4;
    const uint lane = lid & 15;
    const uint r = lane >> 2;
    const uint c = lane & 3;

    const uint c0 = C_row[i];
    const uint c1 = C_row[i+1];

    for(uint cb=c0+team; cb<c1; cb+=NTEAM){

        const uint t0 = plan_ptr[cb];
        const uint t1 = plan_ptr[cb+1];

        // Four independent FMA accumulators, one per column index m —
        // breaks the serial dependency chain across plan terms (F7).
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        bool bad = false;

        // Iterate precomputed plan terms — no intersection, no search.
        for(uint t=t0; t<t1; ++t){

            const uint ia = plan_a_idx[t];
            const uint bb = plan_b_idx[t];

            // Fail-loud: ia out of range means the plan is corrupt.
            if(ia >= na){
                bad = true;
                break;
            }

            __local const float* Ab = lA + ia*BS2;
            __global const float* Bjk = B + bb*BS2;

            // C_ij[r,c] += sum_m A_ik[r,m] * B_jk[c,m]  (B_kj = B_jk^T)
            s0 = fma(Ab[4*r+0], Bjk[4*c+0], s0);
            s1 = fma(Ab[4*r+1], Bjk[4*c+1], s1);
            s2 = fma(Ab[4*r+2], Bjk[4*c+2], s2);
            s3 = fma(Ab[4*r+3], Bjk[4*c+3], s3);
        }

        C[cb*BS2 + lane] = bad ? NAN : (s0 + s1) + (s2 + s3);
    }
}


// ============================================================================
// SIMPLE ELEMENTWISE OPERATIONS
// ============================================================================

__kernel void bsr4_zero(
    const uint nblock,
    __global float* A
){
    const uint i = get_global_id(0);

    if(i < nblock*BS2){
        A[i] = 0.0f;
    }
}


// C = alpha*A + beta*B
//
// A,B,C must have identical CSR block ordering.
//
__kernel void bsr4_axpby(
    const uint nblock,

    const float alpha,
    __global const float* A,

    const float beta,
    __global const float* B,

    __global float* C
){
    const uint i = get_global_id(0);

    if(i < nblock*BS2){

        C[i] =
            fma(
                alpha,
                A[i],
                beta*B[i]
            );
    }
}


// ============================================================================
// GENERALIZED McWEENY COMBINATION
//
// First compute by masked sparse products:
//
//      T1 = P_MT(K*S)
//
//      Q  = P_MK(T1*K)
//         = P_MK(K*S*K)
//
//      T2 = P_MT(Q*S)
//
//      V  = P_MK(T2*K)
//         = P_MK(K*S*K*S*K)
//
// then
//
//      Knew = 3Q - 2V
//
// No polynomial product is ever represented densely.
//
// ============================================================================

__kernel void bsr4_mcweeny(
    const uint nblock,

    __global const float* Q_KSK,
    __global const float* V_KSKSK,

    __global float* Knew
){
    const uint i = get_global_id(0);

    if(i < nblock*BS2){

        Knew[i] =
            fma(
                3.0f,
                Q_KSK[i],
                -2.0f*V_KSKSK[i]
            );
    }
}


// ============================================================================
// METRIC TC2
//
//      Q = K S K
//
//      n = Tr(KS)
//
//      n > Nocc : Knew = Q
//
//      n < Nocc : Knew = 2K-Q
//
// trace_KS points to ONE FLOAT IN GPU MEMORY.
// CPU does not need to read it.
//
// ============================================================================

__kernel void bsr4_tc2(
    const uint nblock,

    __global const float* K,
    __global const float* Q_KSK,

    __global const float* trace_KS,

    const float Nocc,

    __global float* Knew
){
    const uint i = get_global_id(0);

    if(i >= nblock*BS2) return;


    const float k = K[i];
    const float q = Q_KSK[i];


    if(trace_KS[0] > Nocc){

        Knew[i] = q;

    }else{

        Knew[i] = 2.0f*k-q;
    }
}


// ============================================================================
// SYMMETRIZE BLOCK-CSR MATRIX
//
// Because sparse truncation + finite precision can introduce tiny asymmetry.
//
// transpose_block[b] gives the block index corresponding to:
//
//      b  = (i,j)
//      bt = (j,i)
//
// This mapping should be precomputed once in Rust.
//
// ============================================================================

__kernel void bsr4_symmetrize(
    const uint nblock,

    __global const uint* transpose_block,

    __global float* A
){
    const uint b = get_global_id(0);

    if(b >= nblock) return;


    const uint bt = transpose_block[b];


    // One side owns pair, avoiding races.
    if(b > bt) return;


    __global float* X =
        A + b*BS2;

    __global float* Y =
        A + bt*BS2;


    if(b == bt){

        // Diagonal atom block.
        //
        // X <- (X + X^T)/2

        for(uint r=0; r<4; ++r){

            for(uint c=r+1; c<4; ++c){

                const float v =
                    0.5f*(
                        X[4*r+c]
                        +
                        X[4*c+r]
                    );

                X[4*r+c] = v;
                X[4*c+r] = v;
            }
        }

    }else{

        // Offdiagonal pair:
        //
        // X_ij <- 1/2 (X_ij + X_ji^T)
        // X_ji <- transpose(X_ij)

        for(uint r=0; r<4; ++r){

            for(uint c=0; c<4; ++c){

                const float v =
                    0.5f*(
                        X[4*r+c]
                        +
                        Y[4*c+r]
                    );

                X[4*r+c] = v;
                Y[4*c+r] = v;
            }
        }
    }
}


// ============================================================================
// SCC HAMILTONIAN UPDATE
//
// DFTB SCC:
//
// H_mu,nu = H0_mu,nu
//         + 1/2 S_mu,nu ( V_A(mu) + V_A(nu) )             (6)
//
// H0,S,H use the short physical Slater-Koster block CSR mask.
//
// one WG = one atom row
// one 16-thread TEAM = one 4x4 block
//
// V of all neighbors is cached in local memory.
//
// ============================================================================

__attribute__((reqd_work_group_size(WG,1,1)))
__kernel void bsr4_build_Hscc(
    const uint nrow,

    __global const uint* row_ptr,
    __global const uint* col_idx,

    __global const float* H0,
    __global const float* S,

    __global const float* V_atom,

    // Physical orbitals per atom (1 or 4). The SCC shift applies ONLY to
    // physical (i_r < n_orb[i], j_c < n_orb[j]) lanes — dummy lanes keep H0
    // (E_DUMMY on dummy diagonals, zero elsewhere). Applying ½S·(V_i+V_j) to
    // dummy lanes would pollute the padded space (second review R13).
    __global const uint* n_orb,

    __global float* H
){
    const uint i   = get_group_id(0);
    const uint lid = get_local_id(0);

    if(i >= nrow) return;


    const uint b0 = row_ptr[i];
    const uint b1 = row_ptr[i+1];

    const uint nb = b1-b0;


    if(nb > MAX_LEFT_BLOCKS) return;


    __local float lVj[MAX_LEFT_BLOCKS];
    __local uint  lNj[MAX_LEFT_BLOCKS];
    __local float lVi;


    if(lid == 0){
        lVi = V_atom[i];
    }


    for(uint k=lid; k<nb; k+=WG){

        const uint j =
            col_idx[b0+k];

        lVj[k] =
            V_atom[j];
        lNj[k] =
            n_orb[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint team = lid >> 4;
    const uint lane = lid & 15;
    const uint lr   = lane >> 2;   // row within 4x4 block
    const uint lc   = lane & 3;    // col within 4x4 block
    const uint ni   = n_orb[i];
    for(uint bl=team; bl<nb; bl+=NTEAM){

        const uint b = b0+bl;


        const float dV =
            0.5f*(
                lVi
                +
                lVj[bl]
            );


        const uint p =
            b*BS2
            +
            lane;

        const bool active = (lr < ni) && (lc < lNj[bl]);

        H[p] = active
            ? fma(
                S[p],
                dV,
                H0[p]
            )
            : H0[p];
    }
}


// ============================================================================
// MULLIKEN CHARGES DIRECTLY FROM KS
//
// closed shell:
//
//      q_A = 2 sum_{mu in A} (KS)_mu,mu
//
// With four orbitals/atom:
//
//      q_A = 2 Tr[(KS)_AA]                               (7)
//
// diag_block[i] gives the block offset of (i,i).
//
// ============================================================================

// Physical-orbital masked Mulliken (second review R14): only lanes
// d < n_orb[i] are physical; the rest are padded dummies that must not
// contribute to the charge. `q_dum[i]` reports the dummy-lane diagonal
// content of KS — a leak/occupation diagnostic (should be ~0).
// Packed output (F7): q_pack[i] = q_phys, q_pack[nrow+i] = q_dum — one
// device buffer, one host read for both.
__kernel void bsr4_mulliken_KS(
    const uint nrow,

    __global const uint* diag_block,

    __global const float* KS,

    __global const uint* n_orb,

    __global float* q_pack
){
    const uint i =
        get_global_id(0);

    if(i >= nrow) return;


    const uint b =
        diag_block[i];


    __global const float* X =
        KS + b*BS2;

    const uint ni = n_orb[i];
    float qp = 0.0f;
    float qd = 0.0f;
    for(uint d = 0; d < BS; d++){
        const float v = X[d * BS + d];
        if(d < ni){ qp += v; } else { qd += v; }
    }
    q_pack[i]        = 2.0f * qp;
    q_pack[nrow + i] = 2.0f * qd;
}


// ============================================================================
// TRACE Tr(KS)
//
// Since KS already exists:
//
//      Tr(KS) = sum_A Tr[(KS)_AA]
//
// Only diagonal 4x4 blocks are touched.
//
// Use this first-stage reduction and recursively call reduce_sum_f32 until
// one float remains.
//
// ============================================================================

// Physical-orbital masked trace (second review R14): sums only diagonal
// lanes d < n_orb[i] — padded dummy lanes do not count toward Tr(KS)=Nocc.
__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void bsr4_trace_KS_partial(
    const uint nrow,

    __global const uint* diag_block,

    __global const float* KS,

    __global const uint* n_orb,

    __global float* partial
){
    const uint lid =
        get_local_id(0);

    const uint gid =
        get_global_id(0);

    const uint gsize =
        get_global_size(0);


    __local float buf[REDUCE_WG];


    float sum = 0.0f;


    for(
        uint i=gid;
        i<nrow;
        i+=gsize
    ){

        const uint b =
            diag_block[i];


        __global const float* X =
            KS + b*BS2;

        const uint ni = n_orb[i];
        for(uint d = 0; d < ni; d++){
            sum += X[d * BS + d];
        }
    }


    buf[lid] = sum;


    barrier(CLK_LOCAL_MEM_FENCE);


    for(
        uint step=REDUCE_WG>>1;
        step>0;
        step>>=1
    ){

        if(lid < step){

            buf[lid] +=
                buf[lid+step];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if(lid == 0){

        partial[get_group_id(0)] =
            buf[0];
    }
}


// ============================================================================
// TRACE Tr(K·H0) OVER THE H/S SUPPORT  (sparse masked band energy, review R7)
//
//   Tr(K·H0) = Σ_{(i,j)∈M_HS} <K_ji, H_ij>_F
//
// `hs_to_kT[b]` maps an M_HS block (i,j) to the M_K block index of the
// *transposed* block (j,i), or −1 if (j,i) is outside M_K (contributes 0 —
// consistent with K truncation on M_K). One FMA per stored element.
// ============================================================================

__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void bsr4_trace_hk_partial(
    const uint nblock_hs,

    __global const float* H,        // values on M_HS
    __global const float* K,        // values on M_K
    __global const int* hs_to_kT,   // hs block -> k block of (j,i), or -1

    __global float* partial
){
    const uint lid =
        get_local_id(0);
    const uint gid =
        get_global_id(0);
    const uint gsize =
        get_global_size(0);

    __local float buf[REDUCE_WG];
    float sum = 0.0f;

    for(uint idx = gid; idx < nblock_hs * BS2; idx += gsize){
        const uint b = idx / BS2;
        const int kb = hs_to_kT[b];
        if(kb < 0) continue;
        const uint lane = idx % BS2;
        const uint r = lane / BS;
        const uint c = lane % BS;
        // K_ji[c,r] pairs with H_ij[r,c]
        sum += H[idx] * K[(uint)kb * BS2 + c * BS + r];
    }

    buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step = REDUCE_WG >> 1; step > 0; step >>= 1){
        if(lid < step){ buf[lid] += buf[lid + step]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){ partial[get_group_id(0)] = buf[0]; }
}


// ============================================================================
// FROBENIUS NORM²  (for normalized residuals, review R12)
//
//   partial reduction of ||X||_F² = Σ_l x_l² over the whole values array.
// ============================================================================

__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void bsr4_frobenius_sq_partial(
    const uint n,

    __global const float* x,

    __global float* partial
){
    const uint lid =
        get_local_id(0);
    const uint gid =
        get_global_id(0);
    const uint gsize =
        get_global_size(0);

    __local float buf[REDUCE_WG];
    float sum = 0.0f;
    for(uint idx = gid; idx < n; idx += gsize){
        sum += x[idx] * x[idx];
    }

    buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step = REDUCE_WG >> 1; step > 0; step >>= 1){
        if(lid < step){ buf[lid] += buf[lid + step]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){ partial[get_group_id(0)] = buf[0]; }
}


// ============================================================================
// RESTRICT / PROJECT VALUES BETWEEN MASKS  (Z on M_Z → Z|M_K for K0, R8b)
//
//   out[b] = in[map[b]]   if map[b] >= 0   (map: out-block -> in-block)
//   out[b] = 0            otherwise
//
// Gather-only copy; used to project Z (on M_Z) onto M_K for the K0 axpby.
// ============================================================================

__kernel void bsr4_restrict(
    const uint nblock_out,

    __global const int* map,
    __global const float* in,

    __global float* out
){
    const uint idx = get_global_id(0);
    if(idx >= nblock_out * BS2) return;
    const uint b = idx / BS2;
    const int src = map[b];
    out[idx] = (src < 0) ? 0.0f : in[(uint)src * BS2 + (idx % BS2)];
}


// ============================================================================
// GENERIC GPU REDUCTION
// ============================================================================

__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void reduce_sum_f32(
    const uint n,

    __global const float* x,

    __global float* out
){
    const uint lid =
        get_local_id(0);

    const uint gid =
        get_global_id(0);

    const uint gsize =
        get_global_size(0);


    __local float buf[REDUCE_WG];


    float sum = 0.0f;


    for(
        uint i=gid;
        i<n;
        i+=gsize
    ){
        sum += x[i];
    }


    buf[lid] = sum;


    barrier(CLK_LOCAL_MEM_FENCE);


    for(
        uint step=REDUCE_WG>>1;
        step>0;
        step>>=1
    ){

        if(lid < step){

            buf[lid] +=
                buf[lid+step];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if(lid == 0){

        out[get_group_id(0)] =
            buf[0];
    }
}


// ============================================================================
// DIRECT IDENTITY RESIDUAL
//
//      R² = ||A - I||²_F
//
// This avoids the cancellation in ||A||² - 2 Tr(A) + N when A is close to I.
// diag_flag[b] is one only for diagonal atom blocks.
// ============================================================================

__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void bsr4_identity_residual_partial(
    const uint nblock,

    __global const uint* diag_flag,

    __global const float* A,

    __global float* partial
){
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = nblock*BS2;

    __local float buf[REDUCE_WG];
    float sum = 0.0f;

    for(uint i=gid; i<n; i+=gsize){
        const uint b = i/BS2;
        const uint lane = i & 15;
        float d = A[i];
        if(diag_flag[b] != 0u && (lane == 0u || lane == 5u || lane == 10u || lane == 15u)){
            d -= 1.0f;
        }
        sum = fma(d, d, sum);
    }

    buf[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint step=REDUCE_WG>>1; step>0; step>>=1){
        if(lid < step) buf[lid] += buf[lid+step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) partial[get_group_id(0)] = buf[0];
}


// ============================================================================
// GENERALIZED IDEMPOTENCY ERROR
//
//      R² = || K S K - K ||²_F                            (8)
//
// Q and K must use identical K masks.
//
// Together with
//
//      |Tr(KS)-Nocc|
//
// this is a useful purification convergence diagnostic.
//
// ============================================================================

__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void bsr4_idempotency_partial(
    const uint nblock,

    __global const float* Q_KSK,

    __global const float* K,

    __global float* partial
){
    const uint lid =
        get_local_id(0);

    const uint gid =
        get_global_id(0);

    const uint gsize =
        get_global_size(0);


    const uint n =
        nblock*BS2;


    __local float buf[REDUCE_WG];


    float sum =
        0.0f;


    for(
        uint i=gid;
        i<n;
        i+=gsize
    ){

        const float d =
            Q_KSK[i]
            -
            K[i];


        sum =
            fma(
                d,
                d,
                sum
            );
    }


    buf[lid] =
        sum;


    barrier(CLK_LOCAL_MEM_FENCE);


    for(
        uint step=REDUCE_WG>>1;
        step>0;
        step>>=1
    ){

        if(lid < step){

            buf[lid] +=
                buf[lid+step];
        }


        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if(lid == 0){

        partial[get_group_id(0)] =
            buf[0];
    }
}


// ============================================================================
// BLOCK THRESHOLDING TEST
//
// This DOES NOT compact CSR and therefore DOES NOT save memory.
//
// It is only useful for testing:
//
//      "what happens if blocks smaller than epsilon are considered zero?"
//
// before implementing dynamic CSR rebuilding.
//
// One work-item/block, because thresholding is infrequent and this is simple.
//
// block_row[b] = atom row owning block b.
//
// ============================================================================

__kernel void bsr4_drop_small(
    const uint nblock,

    __global const uint* block_row,
    __global const uint* col_idx,

    const float epsilon2,

    __global float* A
){
    const uint b =
        get_global_id(0);


    if(b >= nblock) return;


    const uint i =
        block_row[b];

    const uint j =
        col_idx[b];


    // Never drop diagonal.
    if(i == j) return;


    __global float* X =
        A + b*BS2;


    float norm2 =
        0.0f;


    for(uint k=0; k<16; ++k){

        norm2 =
            fma(
                X[k],
                X[k],
                norm2
            );
    }


    if(norm2 < epsilon2){

        for(uint k=0; k<16; ++k){

            X[k] =
                0.0f;
        }
    }
}


// ============================================================================
// OPTIONAL LNV GRADIENT COMBINATION
//
// No inverse overlap is required.
//
// Let:
//
//      F = H - mu S
//
//      K(L) = 3 L S L - 2 L S L S L
//
// Grand potential:
//
//      Omega[L] = 2 Tr( K(L) F )
//
// For symmetric L,S,F:
//
//  1/2 dOmega/dL =
//
//        3 ( S L F + F L S )
//
//      - 2 ( S L S L F
//           +S L F L S
//           +F L S L S )                                (9)
//
// Each matrix term is generated through the SAME masked sparse multiplication
// primitive above.
//
// This kernel only combines the already-computed terms.
//
// ============================================================================

__kernel void bsr4_lnv_gradient(
    const uint nblock,

    __global const float* SLF,
    __global const float* FLS,

    __global const float* SLSLF,
    __global const float* SLFLS,
    __global const float* FLSLS,

    __global float* G
){
    const uint i =
        get_global_id(0);


    if(i >= nblock*BS2) return;


    G[i] =

        3.0f*(
            SLF[i]
            +
            FLS[i]
        )

        -

        2.0f*(
            SLSLF[i]
            +
            SLFLS[i]
            +
            FLSLS[i]
        );
}


// Lnew = L - alpha*G
//
// Factor 2 omitted from G above can simply be absorbed into alpha.
//
__kernel void bsr4_gradient_step(
    const uint nblock,

    const float alpha,

    __global const float* L,
    __global const float* G,

    __global float* Lnew
){
    const uint i =
        get_global_id(0);


    if(i < nblock*BS2){

        Lnew[i] =
            fma(
                -alpha,
                G[i],
                L[i]
            );
    }
}

// ============================================================================
// GPT-5.6 #9: Device-side inf_norm and identity construction.
//
// These avoid downloading S to compute ||S||_inf and avoid downloading
// row_ptr/col_idx to build the identity. Used by Newton-Schulz Z0 init.
// ============================================================================

// Compute per-orbital absolute row sums into a buffer of size n_atom*BS.
// Each work-item handles one orbital (i_atom, r) and sums |A[i,j][r,c]|
// over all blocks j in row i and all columns c.
//
// row_ptr, col_idx: CSR structure of the matrix
// values: BSR4 values
// row_sums: output, size n_atom*BS
__kernel void bsr4_row_abs_sum(
    const uint n_atom,
    __global const uint* row_ptr,
    __global const uint* col_idx,
    __global const float* values,
    __global float* row_sums
){
    const uint idx = get_global_id(0);
    if (idx >= n_atom * BS) return;
    const uint i = idx / BS;
    const uint r = idx % BS;
    float sum = 0.0f;
    const uint row_start = row_ptr[i];
    const uint row_end   = row_ptr[i+1];
    for (uint blk = row_start; blk < row_end; blk++) {
        const uint base = blk * BS2 + r * BS;
        for (uint c = 0; c < BS; c++) {
            sum += fabs(values[base + c]);
        }
    }
    row_sums[idx] = sum;
}

// Max reduction: finds the maximum value in input[0..n) into output[0].
// Same structure as reduce_sum_f32 but with max instead of sum.
__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void reduce_max_f32(
    const uint n,
    __global const float* x,
    __global float* out
){
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    __local float buf[REDUCE_WG];
    float mx = -INFINITY;
    for (uint i = gid; i < n; i += gsize) {
        mx = fmax(mx, fabs(x[i]));
    }
    buf[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = REDUCE_WG >> 1; step > 0; step >>= 1) {
        if (lid < step) buf[lid] = fmax(buf[lid], buf[lid + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[get_group_id(0)] = buf[0];
}

// Build identity matrix on the device: I = 1 on diagonal-block diagonal
// lanes, 0 **everywhere else**. Writes EVERY structural entry — a cold
// restart must not leave stale off-diagonal values in a persistent buffer
// (second review R6 / old N4 stale-Z bug).
// Each work-item handles one element of the values array.
__kernel void bsr4_build_identity_dev(
    const uint nblock,
    __global const uint* diag_flag,
    __global float* values
){
    const uint idx = get_global_id(0);
    if (idx >= nblock * BS2) return;
    const uint b = idx / BS2;
    const uint lane = idx % BS2;
    const bool is_diag_lane = (lane == 0u || lane == 5u || lane == 10u || lane == 15u);
    values[idx] = (diag_flag[b] && is_diag_lane) ? 1.0f : 0.0f;
}

// Scale all elements of a BSR4 values buffer by a scalar alpha.
// Used to scale identity by alpha = 1/||S||_inf for Z0 initialization.
__kernel void bsr4_scale_dev(
    const uint nblock,
    const float alpha,
    __global float* values
){
    const uint idx = get_global_id(0);
    if (idx >= nblock * BS2) return;
    values[idx] *= alpha;
}

// ============================================================================
// GPT-5.6 #19: Device-side Gershgorin spectral bounds.
//
// Computes per-orbital emin_i and emax_i contributions, then host reduces
// to global min/max. Used by spectral_bounds_dev for K0 construction.
// ============================================================================

// Compute per-orbital Gershgorin bounds:
//   emin_i = B_{i,i} - sum_{j!=i} |B_{i,j}|
//   emax_i = B_{i,i} + sum_{j!=i} |B_{i,j}|
// where i,j run over individual orbitals (not atom blocks).
//
// row_ptr, col_idx: CSR structure
// values: BSR4 values
// diag_flag: 1 for diagonal blocks, 0 otherwise
// emin_partial, emax_partial: output, size n_atom*BS
__kernel void bsr4_gershgorin_partial(
    const uint n_atom,
    __global const uint* row_ptr,
    __global const uint* col_idx,
    __global const float* values,
    __global const uint* diag_flag,
    __global float* emin_partial,
    __global float* emax_partial
){
    const uint idx = get_global_id(0);
    if (idx >= n_atom * BS) return;
    const uint i_atom = idx / BS;
    const uint r = idx % BS;
    float diag = 0.0f;
    float offdiag = 0.0f;
    const uint row_start = row_ptr[i_atom];
    const uint row_end   = row_ptr[i_atom + 1];
    for (uint blk = row_start; blk < row_end; blk++) {
        const uint j_atom = col_idx[blk];
        const uint base = blk * BS2 + r * BS;
        for (uint c = 0; c < BS; c++) {
            float v = values[base + c];
            if (j_atom == i_atom && r == c) {
                diag = v;
            } else {
                offdiag += fabs(v);
            }
        }
    }
    emin_partial[idx] = diag - offdiag;
    emax_partial[idx] = diag + offdiag;
}

// Min reduction: finds the minimum value in input[0..n) into output[0].
__attribute__((reqd_work_group_size(REDUCE_WG,1,1)))
__kernel void reduce_min_f32(
    const uint n,
    __global const float* x,
    __global float* out
){
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    __local float buf[REDUCE_WG];
    float mn = INFINITY;
    for (uint i = gid; i < n; i += gsize) {
        mn = fmin(mn, x[i]);
    }
    buf[lid] = mn;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = REDUCE_WG >> 1; step > 0; step >>= 1) {
        if (lid < step) buf[lid] = fmin(buf[lid], buf[lid + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[get_group_id(0)] = buf[0];
}
