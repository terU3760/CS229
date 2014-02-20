#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#define DIM 10
#define NUM_SAMPLE 4096
#define RADIUS 10.0

int comp(const void *const a, const void *const b) {   /* sort principal component scores in descending order */
	return *((const double *const)a) < *((const double *const)b) ? 1 : -1;
}

void pca(double sample[][NUM_SAMPLE]) {
	unsigned int n, d, i, j;
	double s, avg[DIM], stdev[DIM], pcs[DIM];
	gsl_vector *const eigenvals = gsl_vector_alloc(DIM);
	gsl_matrix *const C = gsl_matrix_alloc(DIM, DIM), *const eigenvecs = gsl_matrix_alloc(DIM, DIM);
	gsl_eigen_symmv_workspace *const w = gsl_eigen_symmv_alloc(DIM);
	for (d = 0; d < DIM; ++d) avg[d] = 0, stdev[d] = 0; 
	for (d = 0; d < DIM; ++d) for (n = 0; n < NUM_SAMPLE; ++n) avg[d] += sample[d][n] / NUM_SAMPLE;
	for (d = 0; d < DIM; ++d) for (n = 0; n < NUM_SAMPLE; ++n) { sample[d][n] -= avg[d]; stdev[d] += sample[d][n] * sample[d][n] / NUM_SAMPLE; } 
	for (d = 0; d < DIM; ++d) stdev[d] = sqrt(stdev[d]);
	for (d = 0; d < DIM; ++d) printf("stdev[%u] == %f\n", d, stdev[d]);
	for (d = 0; d < DIM; ++d) for (n = 0; n < NUM_SAMPLE; ++n) sample[d][n] /= stdev[d]; 
	for (i = 0; i < DIM; ++i) for (j = 0; j < DIM; ++j) {
		for (s = 0, n = 0; n < NUM_SAMPLE; ++n) s += sample[i][n] * sample[j][n] / NUM_SAMPLE; 
		gsl_matrix_set(C, i, j, s);
	}
	gsl_eigen_symmv(C, eigenvals, eigenvecs, w);
	for (i = 0; i < DIM; ++i) pcs[i] = gsl_vector_get(eigenvals, i);
	qsort(pcs, DIM, sizeof(double), comp);
	for (i = 0; i < DIM; ++i) printf("%f ", pcs[i]);
	printf("\n");
	gsl_vector_free(eigenvals);
	gsl_matrix_free(C);
	gsl_matrix_free(eigenvecs);
	gsl_eigen_symmv_free(w);
}

int main(int argc, char *argv[]) {
	unsigned int n, d;
	double /* r, */phi/*, theta, sp, cp, st, ct*/, sample[DIM][NUM_SAMPLE];
	srand(time(0));
	/* generate sample points (with some pattern hidden among random data) */ 
	for (n = 0; n < NUM_SAMPLE; ++n) {
		/* spherical correlation */	
		/*
		phi = 2 * M_PI * rand() / RAND_MAX;
		theta = M_PI * rand() / RAND_MAX;
		sp = sin(phi);
		cp = cos(phi);
		st = sin(theta);
		ct = cos(theta);
		sample[0][n] = RADIUS * st * cp;
		sample[1][n] = RADIUS * st * sp;
		sample[2][n] = RADIUS * ct;
		*/
		
		/* conic, truncated by 1 half-plane */
		phi = M_PI * rand() / RAND_MAX;
		/* theta = M_PI * rand() / RAND_MAX; */
		sample[2][n] = RADIUS * rand() / RAND_MAX;
		sample[3][n] = 1000 * RADIUS * rand() / RAND_MAX;
		sample[0][n] = sample[2][n] * cos(phi);
		sample[1][n] = sample[2][n] * sin(phi);
		/*
		sample[4][n] = sample[3][n] * cos(theta);
		sample[5][n] = sample[3][n] * sin(theta);
		*/
		sample[4][n] = sample[3][n] * cos(phi);
		sample[5][n] = sample[3][n] * sin(phi);

		/* compare with all-random data points: */
		/* for (d = 0; d < DIM; ++d) sample[n][d] = RADIUS * rand() / RAND_MAX; */
	}
	for (d = 6; d < DIM; ++d) for (n = 0; n < NUM_SAMPLE; ++n) sample[d][n] = d * (2 * RADIUS * rand() / RAND_MAX - RADIUS);  /* the other coordinates are all non-negative */
	pca(sample);
	return 0;
}
