#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <jpeglib.h>

#define FAIL -1
#define SUCCESS 1
#define WIDTH 1600
#define HEIGHT 1200
#define PT_SIZE 10
#define NC 3  /* assuming color_space is JCS_RGB */
#define X_DAT "x.dat"
#define Y_DAT "y.dat"
#define EPS 0.000001
#define MIN_R 0.001
#define LAMBDA 0.0001
#define ETA 0.5
#define M 120
#define M_CLS 80
#define MAX_NUM_ITER 1000
#define DISP_DATA_OUTPUT "data.jpg"
#define LWLR_OUTPUT "lwlr_result.jpg"
#define LWLR_PRED_OUTPUT "lwlr_pred.jpg"

int write_jpeg_file(const char *const fn, const unsigned char *const img) {
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	JSAMPROW row_pointer[1];
	FILE *const fp = fopen(fn, "wb");
	if (!fp) {
		fprintf(stderr, "Error opening output jpeg file %s\n!", fn);
		return FAIL;
	}
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, fp);
	cinfo.image_width = WIDTH;
	cinfo.image_height = HEIGHT;
	cinfo.input_components = NC;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_start_compress(&cinfo, TRUE);
	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer[0] = &img[cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	fclose(fp);
	return SUCCESS;
}

int disp_data(const char *const fn_x, const char *const fn_y, const char *const fn_out) {
	unsigned int k, i, j, px, py;
	unsigned char *img;
	FILE *const fp_x = fopen(fn_x, "r"), *const fp_y = fopen(fn_y, "r");
	double y, x[2], min_x[2] = {DBL_MAX, DBL_MAX}, max_x[2] = {DBL_MIN, DBL_MIN}, rx[2];
	if (!fp_x) {
		fprintf(stderr, "Error opening data file %s\n!", fn_x);
		return FAIL;
	}
	img = (unsigned char *)malloc(WIDTH * HEIGHT * NC * sizeof(char));
	memset(img, 0, WIDTH * HEIGHT * NC * sizeof(char));
	while (fscanf(fp_x, "%lf%lf", &x[0], &x[1]) != EOF) {
		for (k = 0; k < 2; ++k) {
			if (x[k] < min_x[k]) min_x[k] = x[k];
			if (x[k] > max_x[k]) max_x[k] = x[k];
		}
	}
	for (k = 0; k < 2; ++k) rx[k] = MIN_R + max_x[k] - min_x[k];
	fseek(fp_x, 0, SEEK_SET);
	while (fscanf(fp_x, "%lf%lf", &x[0], &x[1]) != EOF) {
		fscanf(fp_y, "%lf", &y);
		px = ((x[0] - min_x[0]) / rx[0] * (WIDTH - PT_SIZE));
		py = ((x[1] - min_x[1]) / rx[1] * (HEIGHT - PT_SIZE));
		for (j = py; j < py + PT_SIZE && j < HEIGHT; ++j) for (i = px; i < px + PT_SIZE && i < WIDTH; ++i) for (k = 0; k < NC; ++k) img[j * WIDTH * NC + i * NC + k] = k == 0 ? M * y : k == 2 ? M * (1.0 - y) : 0;
	}
	write_jpeg_file(fn_out, img);
	fclose(fp_x);
	fclose(fp_y);
	free(img);
	return SUCCESS; 
}

int lwlr(const char *const fn_x, const char *const fn_y, const char *const fn_out, const char *const fn_pred_out, const double tau) {
	unsigned int k, i, j, r, c, px, py, num_pt, num_iter, pred;
	unsigned char *img, *pred_img;
	FILE *const fp_x = fopen(fn_x, "r"), *const fp_y = fopen(fn_y, "r");
	double det, y, x[2], dx[2], min_x[2] = {DBL_MAX, DBL_MAX}, max_x[2] = {DBL_MIN, DBL_MIN}, rx[2], theta[2] = {0, 0}, _theta[2], d[2], g[2], H[3], *X, *Y, *W, *h, *outer_prod;
	if (!fp_x) {
		fprintf(stderr, "Error opening data file %s\n!", fn_x);
		return FAIL;
	}
	img = (unsigned char *)malloc(WIDTH * HEIGHT * NC * sizeof(char));
	pred_img = (unsigned char *)malloc(WIDTH * HEIGHT * NC * sizeof(char));
	memset(img, 0, WIDTH * HEIGHT * NC * sizeof(char));
	num_pt = 0;
	while (fscanf(fp_x, "%lf%lf", &x[0], &x[1]) != EOF) {
		for (k = 0; k < 2; ++k) {
			if (x[k] < min_x[k]) min_x[k] = x[k];
			if (x[k] > max_x[k]) max_x[k] = x[k];
		}
		++num_pt;
	}
	for (k = 0; k < 2; ++k) rx[k] = MIN_R + max_x[k] - min_x[k];
	X = (double *)malloc((num_pt << 1) * sizeof(double));   
	Y = (double *)malloc(num_pt * sizeof(double));   
	W = (double *)malloc(num_pt * sizeof(double));
	h = (double *)malloc(num_pt * sizeof(double));
	outer_prod = (double *)malloc(num_pt * 3 * sizeof(double)); 
	fseek(fp_x, 0, SEEK_SET);
	i = 0;
	while (fscanf(fp_x, "%lf%lf", &x[0], &x[1]) != EOF) {
		X[i << 1] = x[0];
		X[(i << 1) + 1] = x[1];
		outer_prod[i * 3] = x[0] * x[0];
		outer_prod[i * 3 + 1] = x[0] * x[1];
		outer_prod[i * 3 + 2] = x[1] * x[1];
		fscanf(fp_y, "%lf", &Y[i]);
		++i;
	}
	for (r = 0; r < HEIGHT; r += PT_SIZE) {
		x[1] = min_x[1] + r * rx[1] / (HEIGHT - PT_SIZE);
		for (c = 0; c < WIDTH; c += PT_SIZE) {
			x[0] = min_x[0] + c * rx[0] / (WIDTH - PT_SIZE);
			theta[1] = theta[0] = 0;
			for (i = 0; i < num_pt; ++i) {
				for (k = 0; k < 2; ++k) dx[k] = x[k] - X[(i << 1) + k]; 
				W[i] = exp(-(dx[0] * dx[0] + dx[1] * dx[1]) / (2 * tau * tau)); 
			}
			g[1] = g[0] = EPS;
			num_iter = 0;
			while (g[0] * g[0] + g[1] * g[1] >= EPS * EPS && num_iter < MAX_NUM_ITER) {
				for (j = 0; j < 2; ++j) _theta[j] = theta[j], g[j] = 0;
				for (j = 0; j < 3; ++j) H[j] = 0; 
				for (i = 0; i < num_pt; ++i) {
					h[i] = ((double)1) / (1 + exp(-theta[0] * X[i << 1] - theta[1] * X[(i << 1) + 1])); 
					for (j = 0; j < 2; ++j) g[j] += X[(i << 1) + j] * W[i] * (Y[i] - h[i]); 
					for (j = 0; j < 3; ++j) H[j] -= W[i] * h[i] * (1 - h[i]) * outer_prod[i * 3 + j]; 
				}
				for (j = 0; j < 2; ++j) g[j] -= LAMBDA * theta[j];
				H[0] -= LAMBDA; H[2] -= LAMBDA;
				det = H[0] * H[2] - H[1] * H[1];
				theta[0] -= ETA * (H[2] * g[0] - H[1] * g[1]) / det; theta[1] -= ETA * (-H[1] * g[0] + H[0] * g[1]) / det; 
				for (j = 0; j < 2; ++j) d[j] = theta[j] - _theta[j]; 
				++num_iter;
			}
			y = ((double)1) / (1 + exp(-theta[0] * x[0] - theta[1] * x[1]));
			for (j = r; j < r + PT_SIZE; ++j) for (i = c; i < c + PT_SIZE; ++i) for (k = 0; k < NC; ++k) {
				pred = ((theta[0] * x[0] + theta[1] * x[1]) > 0);
				img[j * WIDTH * NC + i * NC + k] = k == 0 ? M_CLS * y : k == 2 ? M_CLS * (1.0 - y) : 0;
				pred_img[j * WIDTH * NC + i * NC + k] = k == 0 ? M_CLS * pred : k == 2 ? M_CLS * (1 - pred) : 0;
			}
		}
	}
	fseek(fp_x, 0, SEEK_SET);
	fseek(fp_y, 0, SEEK_SET);
	while (fscanf(fp_x, "%lf%lf", &x[0], &x[1]) != EOF) {
		fscanf(fp_y, "%lf", &y);
		px = ((x[0] - min_x[0]) / rx[0] * (WIDTH - PT_SIZE));
		py = ((x[1] - min_x[1]) / rx[1] * (HEIGHT - PT_SIZE));
		for (j = py; j < py + PT_SIZE && j < HEIGHT; ++j) for (i = px; i < px + PT_SIZE && i < WIDTH; ++i) for (k = 0; k < NC; ++k) pred_img[j * WIDTH * NC + i * NC + k] = img[j * WIDTH * NC + i * NC + k] = k == 0 ? M * y : k == 2 ? M * (1.0 - y) : 0;
	}
	write_jpeg_file(fn_out, img);
	write_jpeg_file(fn_pred_out, pred_img);
	fclose(fp_x);
	fclose(fp_y);
	free(X);
	free(Y);
	free(W);
	free(h);
	free(outer_prod);
	free(img);
	free(pred_img);
	return SUCCESS; 
}

int main(int argc, char *argv[]) {
	disp_data(X_DAT, Y_DAT, DISP_DATA_OUTPUT);
	lwlr(X_DAT, Y_DAT, LWLR_OUTPUT, LWLR_PRED_OUTPUT, 0.1);
	return 0;
}
