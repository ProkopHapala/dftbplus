/* tblite C API helper – outputs JSON with H, S, charges, eigenvalues, energy.
 *
 * Compile:
 *   gcc -O2 tblite_helper.c -o tblite_helper \
 *       -I/home/prokophapala/git/dftbplus/external/tblite/origin/include \
 *       -L/home/prokophapala/git/dftbplus/_build/external/tblite/origin \
 *       -ltblite -lgfortran -lblas -llapack -lm
 *
 * Usage:
 *   ./tblite_helper <nat> <charge> <uhf>
 *   followed by nat lines of:  <Z> <x> <y> <z>
 *
 * Coordinates are expected in Angstrom.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tblite.h"

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <nat> <charge> <uhf>\n", argv[0]);
        fprintf(stderr, "Then <nat> lines of: <Z> <x> <y> <z> (Angstrom)\n");
        return 1;
    }

    int nat = atoi(argv[1]);
    int charge = atoi(argv[2]);
    int uhf = atoi(argv[3]);
    if (nat <= 0) die("nat must be > 0");

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

    /* GFN1-xTB calculator */
    tblite_calculator calc = tblite_new_gfn1_calculator(ctx, mol);
    if (!calc) die("tblite_new_gfn1_calculator failed");

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

    /* Eigenvalues */
    double *emo = calloc(nao, sizeof(double));
    tblite_get_result_orbital_energies(err, res, emo);

    /* Hamiltonian (dense nao×nao) */
    double *hmat = calloc(nao * nao, sizeof(double));
    tblite_get_result_hamiltonian_matrix(err, res, hmat);

    /* Overlap (dense nao×nao) */
    double *smat = calloc(nao * nao, sizeof(double));
    tblite_get_result_overlap_matrix(err, res, smat);

    /* Output JSON */
    printf("{\n");
    printf("  \"nat\": %d,\n", nat);
    printf("  \"nao\": %d,\n", nao);
    printf("  \"energy\": %.16e,\n", energy);
    printf("  \"charges\": [");
    for (int i = 0; i < nat; ++i) {
        printf("%.16e%s", qat[i], (i + 1 < nat) ? ", " : "");
    }
    printf("],\n");
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
    printf("  \"overlap\": [");
    for (int i = 0; i < nao * nao; ++i) {
        printf("%.16e%s", smat[i], (i + 1 < nao * nao) ? ", " : "");
    }
    printf("]\n");
    printf("}\n");

    free(xyz); free(num); free(qat); free(emo); free(hmat); free(smat);
    tblite_delete_result(&res);
    tblite_delete_calculator(&calc);
    tblite_delete_structure(&mol);
    tblite_delete_context(&ctx);
    tblite_delete_error(&err);
    return 0;
}
