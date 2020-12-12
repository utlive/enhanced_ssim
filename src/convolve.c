/*
 * Copyright (c) 2011, Tom Distler (http://tdistler.com)
 * All rights reserved.
 *
 * The BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the tdistler.com nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * (06/10/2016) Updated by zli-nflx (zli@netflix.com) to optimize _iqa_convolve.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "convolve.h"
#include "iqa_options.h"

float KBND_SYMMETRIC(const float *img, int w, int h, int x, int y, float bnd_const)
{
	if (x<0) x=-1-x;
	else if (x>=w) x=(w-(x-w))-1;
	if (y<0) y=-1-y;
	else if (y>=h) y=(h-(y-h))-1;
	return img[y*w + x];
}

float KBND_REPLICATE(const float *img, int w, int h, int x, int y, float bnd_const)
{
	if (x<0) x=0;
	if (x>=w) x=w-1;
	if (y<0) y=0;
	if (y>=h) y=h-1;
	return img[y*w + x];
}

float KBND_CONSTANT(const float *img, int w, int h, int x, int y, float bnd_const)
{
	if (x<0) x=0;
	if (y<0) y=0;
	if (x>=w || y>=h)
		return bnd_const;
	return img[y*w + x];
}

void _iqa_integral_image_mean(float *img, int w, int h, const struct _kernel *k, float *result, int *rw, int *rh)
{
	int window_len = k->w; /* Assumes square. */
	int window_stride = k->stride;

	/* Compute the dimensions of the 'mean map' */
	int dst_w = (w - window_len + 1)/window_stride;
	int dst_h = (h - window_len + 1)/window_stride;

	long double *integral_img = (long double*)malloc((w+1)*(h+1)*sizeof(long double));
	if (!integral_img){
		fprintf(stderr, "error: Could not allocate memory for integral image.\n");
		fflush(stderr);
		return;
	}

	float *dst = 0;
	if (result)
		dst = result;
	else{
		dst = (float*)malloc((size_t)dst_w*dst_h*sizeof(float)); /* If img is to be rewritten, allocate temporary memory for the result */
		if (!dst){
			fprintf(stderr, "error: Could not allocate memory for temporary result image.\n");
			fflush(stderr);
			return;
		}
	}

	/* Initialize borders of integral image */
	for (int i = 0; i <= h; ++i) integral_img[i*(w+1)] = 0.0;
	for (int j = 0; j <= w; ++j) integral_img[j] = 0.0;

	/* Compute cumulative sums along each axis to obtain integral image */
	for (int i = 1; i <= h; ++i)
		for (int j = 1;j <= w; ++j)
			integral_img[i*(w+1) + j] = (long double)img[(i-1)*w + (j-1)] + integral_img[(i-1)*(w+1) + j] + integral_img[i*(w+1) + (j-1)] - integral_img[(i-1)*(w+1) + (j-1)];

	/* Calculate local means using the integral image */
	int i_strided, j_strided;
	for (int i = 0; i < dst_h; ++i){
		for (int j = 0; j < dst_w; ++j){
			i_strided = i*window_stride;
			j_strided = j*window_stride;
			dst[i*dst_w + j] = (integral_img[i_strided*(w+1) + j_strided] -
								integral_img[i_strided*(w+1) + j_strided + window_len] -
								integral_img[(i_strided + window_len)*(w+1) + j_strided] +
								integral_img[(i_strided + window_len)*(w+1) + j_strided + window_len])/(window_len*window_len);
		}
	}

	/* Free memory used to store the integral image */
	if (integral_img){
		free(integral_img);
		integral_img = 0;
	}

	/* If rw and rh exist, update their values */
	if (rw) *rw = dst_w;
	if (rh) *rh = dst_h;

	if (dst != result){
		for (int i = 0; i < dst_w*dst_h; ++i) img[i] = dst[i]; /* Overwrite img with the result of computing means */
		free(dst); /* Free temporary memory */
		dst = 0;
	}
}

static float _calc_scale(const struct _kernel *k)
{
    int ii,k_len;
    double sum=0.0;

    if (k->normalized)
        return 1.0f;
    else {
        assert(0); /* zli-nflx: TODO: generalize to make _calc_scale work on 1D separable filtering */

        k_len = k->w * k->h;
        for (ii=0; ii<k_len; ++ii)
            sum += k->kernel[ii];
        if (sum != 0.0)
            return (float)(1.0 / sum);
        return 1.0f;
    }
}

void _iqa_convolve(float *img, int w, int h, const struct _kernel *k, float *result, int *rw, int *rh)
{

#ifdef IQA_CONVOLVE_1D

	/* use 1D separable filter */

	int x,y,kx,ky;
	int dst_w = w - k->w + 1;
	int dst_h = h - k->h + 1;
	float *dst;
	float *img_cache;

	/* 1D separable filtering requires a normalized filter */
	if (!k->normalized)
		assert(0);

	/* create cache */
	img_cache = (float *)calloc(w*h, sizeof(float));
	if (!img_cache)
		assert(0);

	dst = result;
	if (!dst)
		dst = img; /* Convolve in-place */

	/* filter horizontally */
	for (y = 0; y < h; ++y){
		for (x = 0; x < dst_w; ++x){
			for (kx = 0; kx < k->w; ++kx){
				img_cache[y*dst_w + x] += img[y*w + x + kx] * k->kernel_h[kx];
			}
		}
	}

	for (x = 0; x < dst_w; ++x){
		for (y = 0; y < dst_h; ++y){
			dst[y*dst_w + x] = 0;
			for (ky = 0; ky < k->h; ++ky)
				dst[y*dst_w + x] += img_cache[(y + ky)*dst_w + x] * k->kernel_v[ky];
		}
	}

	/* free cache */
	free(img_cache);
	img_cache = 0;

#else /* use 2D filter */

	int x,y,kx,ky,u,v;
	int uc = k->w/2;
	int vc = k->h/2;
	int kw_even = (k->w&1)?0:1;
	int kh_even = (k->h&1)?0:1;
	int dst_w = w - k->w + 1;
	int dst_h = h - k->h + 1;
	int img_offset,k_offset;
	float sum;
	float scale, *dst=result;

	if (!dst)
		dst = img; /* Convolve in-place */

	/* Kernel is applied to all positions where the kernel is fully contained
	 * in the image */
	scale = _calc_scale(k);

	for (y = 0; y < dst_h; ++y){
		for (x = 0; x < dst_w; ++x){
			sum = 0.0;
			for (ky = 0; ky < k->h; ++ky)
				for (kx = 0; kx < k->w; ++kx)
					sum += img[(y + ky)*w + (x + kx)] * k->kernel[ky*k->w + kx];
			img[y*dst_w + x] = (float)(sum * scale);
		}
	}

#endif

	if (rw) *rw = dst_w;
	if (rh) *rh = dst_h;

}

int _iqa_img_filter(float *img, int w, int h, const struct _kernel *k, float *result)
{
	int x,y;
	int img_offset;
	float scale, *dst=result;

	if (!k || !k->bnd_opt)
		return 1;

	if (!dst) {
		dst = (float*)malloc(w*h*sizeof(float));
		if (!dst)
			return 2;
	}

	scale = _calc_scale(k);

	/* Kernel is applied to all positions where top-left corner is in the image */
	for (y=0; y < h; ++y) {
		for (x=0; x < w; ++x) {
			dst[y*w + x] = _iqa_filter_pixel(img, w, h, x, y, k, scale);
		}
	}

	/* If no result buffer given, copy results to image buffer */
	if (!result) {
		for (y=0; y<h; ++y) {
			img_offset = y*w;
			for (x=0; x<w; ++x, ++img_offset) {
				img[img_offset] = dst[img_offset];
			}
		}
		free(dst);
		dst = 0;
	}
	return 0;
}

float _iqa_filter_pixel(const float *img, int w, int h, int x, int y, const struct _kernel *k, const float kscale)
{
	int u,v,uc,vc;
	int kw_even,kh_even;
	int x_edge_left,x_edge_right,y_edge_top,y_edge_bottom;
	int edge,img_offset,k_offset;
	double sum;

	if (!k)
		return img[y*w + x];

	uc = k->w/2;
	vc = k->h/2;
	kw_even = (k->w&1)?0:1;
	kh_even = (k->h&1)?0:1;
	x_edge_left  = uc;
	x_edge_right = w-uc;
	y_edge_top = vc;
	y_edge_bottom = h-vc;

	edge = 0;
	if (x < x_edge_left || y < y_edge_top || x >= x_edge_right || y >= y_edge_bottom)
		edge = 1;

	sum = 0.0;
	k_offset = 0;
	for (v=-vc; v <= vc-kh_even; ++v) {
		img_offset = (y+v)*w + x;
		for (u=-uc; u <= uc-kw_even; ++u, ++k_offset) {
			if (!edge)
				sum += img[img_offset+u] * k->kernel[k_offset];
			else
				sum += k->bnd_opt(img, w, h, x+u, y+v, k->bnd_const) * k->kernel[k_offset];
		}
	}
	return (float)(sum * kscale);
}
