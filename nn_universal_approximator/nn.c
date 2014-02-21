#include <stdio.h>
#include <math.h>

#define NUM_INPUT 2
#define NUM_HIDDEN 4
#define NUM_OUTPUT 1
#define NUM_PT 100
#define ETA 1e-3
#define GAMMA 0.999999
#define TOL 0.0001

double sigmoid(const double x) { return 1.0 / (1.0 + exp(-x)); }

int main(int argc, char *argv[]) {
	unsigned int i, j, n, flag;
	double eta, net_sq_err, err, s, d, z, p, pred, y[NUM_PT], h[NUM_HIDDEN], x[NUM_PT][NUM_INPUT], w1[NUM_HIDDEN][NUM_INPUT], w2[NUM_HIDDEN];
	for (n = 0; n < NUM_PT; ++n) x[n][0] = -1.0 + 2.0 * n / NUM_PT, x[n][1] = 1.0;
	for (n = 0; n < NUM_PT; ++n) y[n] = sin(x[n][0]);
	for (j = 0; j < NUM_HIDDEN; ++j) {
		w2[j] = 1.0; 
		for (i = 0; i < NUM_INPUT; ++i) w1[j][i] = 1.0;
	}
	p = 0.0;
	flag = 0;
	eta = ETA;
	do {
		net_sq_err = 0.0;
		for (n = 0; n < NUM_PT; ++n) {
			/* fwd prop */
			for (pred = 0.0, j = 0; j < NUM_HIDDEN; ++j) {
				for (z = 0.0, i = 0; i < NUM_INPUT; ++i) z += w1[j][i] * x[n][i];
				pred += w2[j] * (h[j] = tanh(z));
			}
			err = fabs(pred - y[n]);
			if (flag) printf("%f: train_err == %f\n", x[n][0], err);
			net_sq_err += err * err; 
			/* bwd prop */
			d = eta * (pred - y[n]);
			p -= d;
			for (j = 0; j < NUM_HIDDEN; ++j) {
				s = 1 - h[j] * h[j];
				for (i = 0; i < NUM_INPUT; ++i) w1[j][i] -= d * w2[j] * s * x[n][i]; 
				w2[j] -= d * h[j]; 
			}
		}
		if (net_sq_err <= TOL) ++flag;
		printf("eta == %f, net_sq_err == %f\n", eta, net_sq_err);
		eta *= GAMMA; 
	}while (net_sq_err >= TOL || flag < 2);
	return 0;
}
