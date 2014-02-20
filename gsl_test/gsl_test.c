#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>

#define N 5

int main (int argc, char *argv[]) {
	unsigned int i, j;
	int s;
	double ent;
	gsl_matrix *const H = gsl_matrix_alloc(N, N);
	gsl_matrix *const inv_H = gsl_matrix_alloc(N, N);
	gsl_permutation *const perm = gsl_permutation_alloc(N);
	/* NOTE: the condition number of the n-by-n Hilbert matrix grows as O((1+\sqrt{2})^{4n}/\sqrt{n}). */
	for (i = 1; i <= N; ++i) {
		for (j = 1; j <= N; ++j) {
			ent = 1.0 / (i + j - 1);
			gsl_matrix_set(H, i - 1, j - 1, ent); 
			printf("%f", ent);
			printf(j == N ? ";\n" : ",");
		}
	}
	printf("\n");
	gsl_linalg_LU_decomp(H, perm, &s);
	gsl_linalg_LU_invert(H, perm, inv_H);
	/* alternatively, refer to invhilb(5) in MATLAB */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			printf("%f", gsl_matrix_get(inv_H, i, j));
			printf(j + 1 == N ? ";\n" : ",");
		}
	}
	gsl_matrix_free(H);
	gsl_matrix_free(inv_H);
	gsl_permutation_free(perm);
	return 0;
}
