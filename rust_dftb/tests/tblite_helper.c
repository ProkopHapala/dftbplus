/* tblite C API helper – outputs JSON with H, S, charges, eigenvalues, energy.
 *
 * Compile:
 *   gcc -O2 tblite_helper.c -o tblite_helper \
 *       -I/home/prokophapala/git/dftbplus/external/tblite/origin/include \
 *       -L/home/prokophapala/git/dftbplus/_build/external/tblite/origin \
 *       -ltblite -lgfortran -lblas -llapack -lm
 *
 * Usage:
 *   ./tblite_helper <nat> <charge> <uhf> <method>
 *   followed by nat lines of:  <Z> <x> <y> <z>
 *
 * method: 1 = GFN1, 2 = GFN2, 3 = GFN2 without dispersion
 * Coordinates are expected in Angstrom.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tblite.h"

/* Local declarations for APIs not yet in system headers */
extern void tblite_get_result_number_of_shells(tblite_error error, tblite_result res, int* nsh);
extern void tblite_get_result_shell_charges(tblite_error error, tblite_result res, double* shell_charges);
extern void tblite_get_result_effective_hamiltonian_matrix(tblite_error error, tblite_result res, double* hmat);
extern void tblite_get_result_dipole_integrals(tblite_error error, tblite_result res, double* dint);
extern void tblite_get_result_quadrupole_integrals(tblite_error error, tblite_result res, double* qint);
extern void tblite_get_result_charge_potential(tblite_error error, tblite_result res, double* vat);
extern void tblite_get_result_dipole_potential(tblite_error error, tblite_result res, double* vdp);
extern void tblite_get_result_quadrupole_potential(tblite_error error, tblite_result res, double* vqp);
extern tblite_calculator tblite_new_gfn2_calculator_nodisp(tblite_context ctx, tblite_structure mol);

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <nat> <charge> <uhf> <method>\n", argv[0]);
        fprintf(stderr, "  method: 1 = GFN1, 2 = GFN2, 3 = GFN2 (no dispersion)\n");
        fprintf(stderr, "Then <nat> lines of: <Z> <x> <y> <z> (Angstrom)\n");
        return 1;
    }

    int nat = atoi(argv[1]);
    int charge = atoi(argv[2]);
    int uhf = atoi(argv[3]);
    int method = atoi(argv[4]);
    if (nat <= 0) die("nat must be > 0");
    if (method < 1 || method > 3) die("method must be 1 (GFN1), 2 (GFN2), or 3 (GFN2 no-disp)");

    double *xyz = malloc(3 * nat * sizeof(double));
    int    *num = malloc(nat * sizeof(int));
    if (!xyz || !num) die("malloc failed");

    for (int i = 0; i < nat; ++i) {
        if (scanf("%d %lf %lf %lf", &num[i], &xyz[3*i], &xyz[3*i+1], &xyz[3*i+2]) != 4)
            die("failed to read atom line");
    }

    /* Angstrom → Bohr (tblite uses Bohr internally) */
    for (int i = 0; i < 3*nat; ++i) xyz[i] *= 1.889726133;

    tblite_error err = tblite_new_error();
    tblite_context ctx = tblite_new_context();
    double charge_d = charge;
    int uhf_i = uhf;
    tblite_structure mol = tblite_new_structure(err, nat, num, xyz, &charge_d, &uhf_i, NULL, NULL);
    if (!mol) die("tblite_new_structure failed");

    /* Select calculator based on method */
    tblite_calculator calc;
    if (method == 1) {
        calc = tblite_new_gfn1_calculator(ctx, mol);
        if (!calc) die("tblite_new_gfn1_calculator failed");
    } else if (method == 2) {
        calc = tblite_new_gfn2_calculator(ctx, mol);
        if (!calc) die("tblite_new_gfn2_calculator failed");
    } else {
        calc = tblite_new_gfn2_calculator_nodisp(ctx, mol);
        if (!calc) die("tblite_new_gfn2_calculator_nodisp failed");
    }

    /* Enable saving H and S */
    tblite_set_calculator_save_integrals(ctx, calc, 1);

    tblite_result res = tblite_new_result();
    if (!res) die("tblite_new_result failed");

    tblite_get_singlepoint(ctx, mol, calc, res);

    /* Extract dimensions */
    int nao = 0;
    tblite_get_result_number_of_orbitals(err, res, &nao);

    /* Energy */
    double energy = 0.0;
    tblite_get_result_energy(err, res, &energy);

    /* Charges */
    double *qat = calloc(nat, sizeof(double));
    tblite_get_result_charges(err, res, qat);

    /* Shell charges */
    int nsh = 0;
    tblite_get_result_number_of_shells(err, res, &nsh);
    double *qsh = calloc(nsh, sizeof(double));
    tblite_get_result_shell_charges(err, res, qsh);

    /* Eigenvalues */
    double *emo = calloc(nao, sizeof(double));
    tblite_get_result_orbital_energies(err, res, emo);

    /* Hamiltonian (dense nao×nao) - non-SCC H0 */
    double *hmat = calloc(nao * nao, sizeof(double));
    tblite_get_result_hamiltonian_matrix(err, res, hmat);

    /* Effective Hamiltonian (dense nao×nao) - SCC H_scc */
    double *hmat_scc = calloc(nao * nao, sizeof(double));
    tblite_get_result_effective_hamiltonian_matrix(err, res, hmat_scc);

    /* Dipole integrals (3 × nao × nao) */
    double *dint = calloc(3 * nao * nao, sizeof(double));
    tblite_get_result_dipole_integrals(err, res, dint);

    /* Quadrupole integrals (6 × nao × nao) */
    double *qint = calloc(6 * nao * nao, sizeof(double));
    tblite_get_result_quadrupole_integrals(err, res, qint);

    /* Charge potential (nat) */
    double *vat = calloc(nat, sizeof(double));
    tblite_get_result_charge_potential(err, res, vat);

    /* Dipole potential (3 × nat) */
    double *vdp = calloc(3 * nat, sizeof(double));
    tblite_get_result_dipole_potential(err, res, vdp);

    /* Quadrupole potential (6 × nat) */
    double *vqp = calloc(6 * nat, sizeof(double));
    tblite_get_result_quadrupole_potential(err, res, vqp);

    /* Overlap (dense nao×nao) */
    double *smat = calloc(nao * nao, sizeof(double));
    tblite_get_result_overlap_matrix(err, res, smat);

    /* Density matrix (dense nao×nao) */
    double *pmat = calloc(nao * nao, sizeof(double));
    tblite_get_result_density_matrix(err, res, pmat);

    /* Orbital coefficients (dense nao×nao) */
    double *cmat = calloc(nao * nao, sizeof(double));
    tblite_get_result_orbital_coefficients(err, res, cmat);

    /* Dipole moment */
    double dipole[3] = {0.0, 0.0, 0.0};
    tblite_get_result_dipole(err, res, dipole);

    /* Output JSON */
    printf("{\n");
    printf("  \"nat\": %d,\n", nat);
    printf("  \"nao\": %d,\n", nao);
    printf("  \"nsh\": %d,\n", nsh);
    printf("  \"energy\": %.16e,\n", energy);
    printf("  \"charges\": [");
    for (int i = 0; i < nat; ++i) {
        printf("%.16e%s", qat[i], (i + 1 < nat) ? ", " : "");
    }
    printf("],\n");
    printf("  \"shell_charges\": [");
    for (int i = 0; i < nsh; ++i) {
        printf("%.16e%s", qsh[i], (i + 1 < nsh) ? ", " : "");
    }
    printf("],\n");
    printf("  \"dipole\": [%.16e, %.16e, %.16e],\n", dipole[0], dipole[1], dipole[2]);
    printf("  \"eigenvalues\": [");
    for (int i = 0; i < nao; ++i) {
        printf("%.16e%s", emo[i], (i + 1 < nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"hamiltonian\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", hmat[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"effective_hamiltonian\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", hmat_scc[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"dipole_integrals\": [");
    for (int i = 0; i < 3 * nao * nao; ++i) {
        printf("%.16e%s", dint[i], (i + 1 < 3 * nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"quadrupole_integrals\": [");
    for (int i = 0; i < 6 * nao * nao; ++i) {
        printf("%.16e%s", qint[i], (i + 1 < 6 * nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"dipole_potential\": [");
    for (int i = 0; i < 3 * nat; ++i) {
        printf("%.16e%s", vdp[i], (i + 1 < 3 * nat) ? ", " : "");
    }
    printf("],\n");
    printf("  \"quadrupole_potential\": [");
    for (int i = 0; i < 6 * nat; ++i) {
        printf("%.16e%s", vqp[i], (i + 1 < 6 * nat) ? ", " : "");
    }
    printf("],\n");
    printf("  \"charge_potential\": [");
    for (int i = 0; i < nat; ++i) {
        printf("%.16e%s", vat[i], (i + 1 < nat) ? ", " : "");
    }
    printf("],\n");
    printf("  \"overlap\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", smat[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"density\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", pmat[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("],\n");
    printf("  \"coefficients\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", cmat[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("]\n");
    printf("}\n");

    free(xyz); free(num); free(qat); free(qsh); free(emo); free(hmat); free(hmat_scc); free(dint); free(qint); free(vat); free(vdp); free(vqp); free(smat); free(pmat); free(cmat);
    tblite_delete_result(&res);
    tblite_delete_calculator(&calc);
    tblite_delete_structure(&mol);
    tblite_delete_context(&ctx);
    tblite_delete_error(&err);
    return 0;
}
