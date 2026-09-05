#!/usr/bin/env python3
"""Scaling test: sparse Chebyshev+Ritz solver on H-passivated carbon ribbons.

Runs the optimized sparse solver on all ribbon TSV files and produces:
  - Timing table (CSV)
  - Scaling plots: time vs N, nnz vs N, parity vs N

Usage:
  python3 scripts/ribbon_scaling_test.py
"""
import sys, time, json, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("/home/prokophapala/git/NumericalMathPlayground/topics/LinearAlgebra/SpectralFiltering")))
from sparse_homo_lumo import (
    parse_hs_matrix_tsv, sparse_cholesky, CholeskyTransformedOperator,
    run_chebyshev_ritz_band, estimate_spectral_range
)

RIBBON_DIR = Path(__file__).parent.parent / "debug" / "graphene_sparse" / "ribbons"
OUT_DIR = RIBBON_DIR
RIBBONS = [
    ("zigzag_w4_L4", 4, 28),
    ("zigzag_w4_L8", 8, 52),
    ("zigzag_w4_L16", 16, 100),
    ("zigzag_w4_L32", 32, 196),
    ("zigzag_w4_L64", 64, 388),
]

def auto_params(norb):
    nvec = 20 if norb < 300 else (30 if norb < 800 else 40)
    cheb_deg = 40 if norb < 300 else (60 if norb < 800 else 80)
    iters = 20 if norb < 300 else (30 if norb < 800 else 40)
    return nvec, cheb_deg, iters

def run_one(name, L, natoms):
    tsv = RIBBON_DIR / f"{name}_hs_matrix.tsv"
    if not tsv.exists():
        print(f"  SKIP: {tsv} not found")
        return None

    print(f"  [{name}] L={L}, natoms={natoms}", flush=True)
    species, coords, H_dense, S_dense, eigs_dense, n_occ = parse_hs_matrix_tsv(tsv)
    norb = H_dense.shape[0]
    homo_dense = eigs_dense[n_occ - 1]
    lumo_dense = eigs_dense[n_occ]
    gap_dense = lumo_dense - homo_dense

    S_csr = sp.csr_matrix(S_dense)
    H_csr = sp.csr_matrix(H_dense)

    # Cholesky
    t0 = time.perf_counter()
    L_csr = sparse_cholesky(S_csr)
    t_chol = time.perf_counter() - t0
    nnz_S = S_csr.nnz
    nnz_H = H_csr.nnz
    nnz_L = L_csr.nnz

    op = CholeskyTransformedOperator(L_csr, H_csr)
    nvec, cheb_deg, iters = auto_params(norb)
    half_w = max(0.03, 2.0 * gap_dense)

    # HOMO band
    t0 = time.perf_counter()
    np.random.seed(42)
    V0 = np.random.randn(norb, nvec)
    w_h, U_h, r_h, spmv_h = run_chebyshev_ritz_band(
        op, V0, homo_dense - half_w, homo_dense + half_w, cheb_deg, iters, True)
    t_homo = time.perf_counter() - t0

    # LUMO band
    t0 = time.perf_counter()
    np.random.seed(123)
    V0 = np.random.randn(norb, nvec)
    w_l, U_l, r_l, spmv_l = run_chebyshev_ritz_band(
        op, V0, lumo_dense - half_w, lumo_dense + half_w, cheb_deg, iters, True)
    t_lumo = time.perf_counter() - t0

    # Pick HOMO/LUMO
    homo_idx = np.argmin(np.abs(w_h - homo_dense))
    lumo_idx = np.argmin(np.abs(w_l - lumo_dense))
    homo_sp = w_h[homo_idx]
    lumo_sp = w_l[lumo_idx]
    homo_err = abs(homo_sp - homo_dense)
    lumo_err = abs(lumo_sp - lumo_dense)
    max_res = max(r_h.max(), r_l.max())
    total_time = t_chol + t_homo + t_lumo
    total_spmv = spmv_h + spmv_l

    result = {
        'name': name, 'L': L, 'natoms': natoms, 'norb': norb,
        'nvec': nvec, 'cheb_deg': cheb_deg, 'iters': iters,
        'nnz_S': nnz_S, 'nnz_H': nnz_H, 'nnz_L': nnz_L,
        'fill_ratio': nnz_L / (norb * norb),
        't_chol_ms': t_chol * 1000,
        't_homo_ms': t_homo * 1000,
        't_lumo_ms': t_lumo * 1000,
        't_total_ms': total_time * 1000,
        'spmv_count': total_spmv,
        'homo_dense': homo_dense, 'lumo_dense': lumo_dense, 'gap_dense': gap_dense,
        'homo_sp': homo_sp, 'lumo_sp': lumo_sp, 'gap_sp': lumo_sp - homo_sp,
        'homo_err': homo_err, 'lumo_err': lumo_err, 'max_residual': max_res,
    }
    print(f"    N={norb}  t={total_time*1000:.0f}ms  HOMO Δ={homo_err:.2e}  LUMO Δ={lumo_err:.2e}  r_max={max_res:.2e}", flush=True)
    return result

def main():
    print("=== Ribbon scaling test ===", flush=True)
    results = []
    for name, L, natoms in RIBBONS:
        r = run_one(name, L, natoms)
        if r: results.append(r)

    # Save CSV
    csv_path = OUT_DIR / "scaling_results.csv"
    with open(csv_path, 'w') as f:
        f.write("name,L,natoms,norb,nvec,cheb_deg,iters,nnz_S,nnz_H,nnz_L,fill_ratio,t_chol_ms,t_homo_ms,t_lumo_ms,t_total_ms,spmv_count,homo_dense,lumo_dense,gap_dense,homo_sp,lumo_sp,gap_sp,homo_err,lumo_err,max_residual\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in [
                'name','L','natoms','norb','nvec','cheb_deg','iters',
                'nnz_S','nnz_H','nnz_L','fill_ratio',
                't_chol_ms','t_homo_ms','t_lumo_ms','t_total_ms','spmv_count',
                'homo_dense','lumo_dense','gap_dense','homo_sp','lumo_sp','gap_sp',
                'homo_err','lumo_err','max_residual'
            ]) + "\n")
    print(f"\nCSV saved: {csv_path}", flush=True)

    # Also save JSON
    json_path = OUT_DIR / "scaling_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {json_path}", flush=True)

    # Print summary table
    print("\n=== Summary ===", flush=True)
    print(f"{'name':20s} {'N':>5s} {'nnz(L)':>8s} {'fill%':>6s} {'t_chol':>8s} {'t_total':>8s} {'HOMO Δ':>10s} {'LUMO Δ':>10s} {'r_max':>10s}")
    for r in results:
        print(f"{r['name']:20s} {r['norb']:5d} {r['nnz_L']:8d} {r['fill_ratio']*100:5.1f}% {r['t_chol_ms']:7.1f}ms {r['t_total_ms']:7.0f}ms {r['homo_err']:10.2e} {r['lumo_err']:10.2e} {r['max_residual']:10.2e}")

    # ─── Plots ───
    Ns = [r['norb'] for r in results]
    natoms_arr = [r['natoms'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time vs N
    ax = axes[0, 0]
    t_chol = [r['t_chol_ms'] for r in results]
    t_solve = [r['t_homo_ms'] + r['t_lumo_ms'] for r in results]
    t_total = [r['t_total_ms'] for r in results]
    ax.plot(Ns, t_total, 'o-', label='Total', linewidth=2, markersize=8)
    ax.plot(Ns, t_chol, 's--', label='Cholesky', linewidth=1.5)
    ax.plot(Ns, t_solve, '^--', label='Chebyshev+Ritz', linewidth=1.5)
    # Reference: O(N^3) dense
    if len(Ns) >= 2:
        # Fit dense reference times (known: 0.10, 0.66, 4.63, 33.5, 285s)
        dense_times = [100, 660, 4630, 33500, 285000]  # ms
        ax.plot(Ns, dense_times, 'k:', label='Dense O(N³)', linewidth=1)
    ax.set_xlabel('Number of orbitals (N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Runtime vs System Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. nnz vs N
    ax = axes[0, 1]
    nnz_S = [r['nnz_S'] for r in results]
    nnz_H = [r['nnz_H'] for r in results]
    nnz_L = [r['nnz_L'] for r in results]
    N2 = [n * n for n in Ns]
    ax.plot(Ns, nnz_S, 'o-', label='nnz(S)', linewidth=2, markersize=8)
    ax.plot(Ns, nnz_H, 's--', label='nnz(H)', linewidth=1.5)
    ax.plot(Ns, nnz_L, '^-', label='nnz(L) Cholesky', linewidth=2, markersize=8)
    ax.plot(Ns, N2, 'k:', label='N² (dense)', linewidth=1)
    ax.set_xlabel('Number of orbitals (N)')
    ax.set_ylabel('Nonzeros')
    ax.set_title('Sparsity vs System Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Parity vs N
    ax = axes[1, 0]
    homo_err = [r['homo_err'] for r in results]
    lumo_err = [r['lumo_err'] for r in results]
    max_res = [r['max_residual'] for r in results]
    ax.semilogy(Ns, homo_err, 'o-', label='|HOMO Δ|', linewidth=2, markersize=8)
    ax.semilogy(Ns, lumo_err, 's-', label='|LUMO Δ|', linewidth=2, markersize=8)
    ax.semilogy(Ns, max_res, '^--', label='max residual', linewidth=1.5)
    ax.axhline(1e-8, color='k', linestyle=':', alpha=0.5, label='1e-8 tolerance')
    ax.set_xlabel('Number of orbitals (N)')
    ax.set_ylabel('Error')
    ax.set_title('Parity vs System Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Fill ratio vs N
    ax = axes[1, 1]
    fill = [r['fill_ratio'] * 100 for r in results]
    ax.plot(Ns, fill, 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Number of orbitals (N)')
    ax.set_ylabel('Cholesky fill ratio (%)')
    ax.set_title('Cholesky Fill-in vs System Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT_DIR / "ribbon_scaling.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}", flush=True)
    plt.close()

    print("\nDone.", flush=True)

if __name__ == '__main__':
    main()
