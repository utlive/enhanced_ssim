/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mem.h"
#include "math_utils.h"
#include "decimate.h"
#include "ssim_tools.h"

/* _ssim_map */
int _ssim_map(const struct _ssim_int *si, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    *ssim_sum += si->l * si->c * si->s;
    return 0;
}

/* _ssim_reduce */
float _ssim_reduce(int w, int h, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    return (float)(*ssim_sum / (double)(w*h));
}

/* map function for CoV pooling */
int _ssim_cov_map(const struct _ssim_int *si, void *ctx)
{
    struct _ssim_ctx *ssim_pair = (struct _ssim_ctx*) ctx;
	double ssim_val =  si->l * si->c * si->s;
	ssim_pair->ssim_sum += ssim_val;
    ssim_pair->ssim_sqd_sum += ssim_val*ssim_val;
	return 0;
}

/* reduce function for CoV pooling */
float _ssim_cov_reduce(int w, int h, void *ctx)
{
    struct _ssim_ctx *ssim_pair = (struct _ssim_ctx*) ctx;
	double ssim_std_sum = sqrt(w*h*ssim_pair->ssim_sqd_sum - ssim_pair->ssim_sum*ssim_pair->ssim_sum);
	return (float)(ssim_std_sum/ssim_pair->ssim_sum);
}

int compute_ssim(const float *ref, const float *cmp, int w, int h,
        int ref_stride, int cmp_stride, int window_type, int window_len, int window_stride, double *score, /* Abhinau added argument here. */
        double *l_score, double *c_score, double *s_score)
{

    int ret = 1;

    int scale;
    int x,y,src_offset,offset;
    float *ref_f,*cmp_f;
    struct _kernel low_pass;
    struct _kernel window;
    float result = INFINITY;
    float l, c, s;
    double ssim_sum=0.0;
    struct _map_reduce mr;
    const struct iqa_ssim_args *args = 0; /* 0 for default */
    float **g_custom_square_window = 0, *g_custom_square_window_h = 0, *g_custom_square_window_v = 0;
    
	/* check stride */
    int stride = ref_stride; /* stride in bytes */
    if (stride != cmp_stride)
    {
        printf("error: for ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride, cmp_stride);
        fflush(stdout);
        goto fail_or_end;
    }
    stride /= sizeof(float); /* stride_ in pixels */

    /* window_type is FFMPEG_SQUARE for 8x8 square window, GAUSSIAN for 11x11 circular-symmetric Gaussian window, CUSTOM_SQUARE for rectangular window of custom size (default) */
    
	if (window_type < 1 || window_type > 2){ /* Only values of 0, 1 and 2 are defined */
		printf("error: for ssim, window_type must be 0, 1 or 2, found %d", window_type);
		fflush(stdout);
		goto fail_or_end;
	}

    if (window_type == CUSTOM_SQUARE && window_len <= 0) {
        printf("error: for ssim, when using custom rectangular window, window_len = %d <= 0", window_len);
		fflush(stdout);
		goto fail_or_end;
	}

    if (window_type == CUSTOM_SQUARE && window_len > _min(w,h)) {
        printf("error: for ssim, when using custom rectangular window, window_len = %d > min(width, height) = %d", window_len, _min(w,h));
		fflush(stdout);
		goto fail_or_end;
	}

    // printf("Initializing algorithm parameters in compute_ssim.\n");
    /* initialize algorithm parameters */
    scale = _max( 1, _round( (float)_min(w,h) / 256.0f ) );
    if (args) {
        if(args->f) {
            scale = args->f;
        }
        mr.map     = _ssim_map;
        mr.reduce  = _ssim_reduce;
        mr.context = (void*)&ssim_sum;
    }

	// printf("window_type = %d. Creating windows.\n", window_type);
    if (window_type == FFMPEG_SQUARE){
		window.kernel = (float*)g_square_window;
		window.kernel_h = (float*)g_square_window_h;
		window.kernel_v = (float*)g_square_window_v;
		window.w = window.h = SQUARE_LEN;
		window.normalized = 1;
		window.bnd_opt = KBND_SYMMETRIC;
		window.stride = 1; /* FFMPEG uses stride 4 but I don't think that is being used in LIBVMAF */
    }
	else if (window_type == GAUSSIAN) {
        window.kernel = (float*)g_gaussian_window;
        window.kernel_h = (float*)g_gaussian_window_h;
        window.kernel_v = (float*)g_gaussian_window_v;
        window.w = window.h = GAUSSIAN_LEN;
		window.stride = 1;
    }
    else if (window_type == CUSTOM_SQUARE) {
		/* Declaring custom square windows */

		/* Windows only need to be created if using convolution.
		 * When using integral images, only the size needs to be specified. */

		/* To create the windows, you can allocate static memory like this. */

    #ifdef USE_IQA_CONVOLVE
		// static const float g_custom_square[window_len][window_len], g_custom_square_h[window_len], g_custom_square_v[window_len];

		/* Or you can allocate memory dynamically like this.
		 * Beware, memory allocated like this may not be contiguous. */

		g_custom_square_window = (float**)malloc(window_len*sizeof(float*));
		if (!g_custom_square_window){
			printf("error: unable to malloc g_custom_square_window.\n");
			goto fail_or_end;
		}
		for (int i = 0; i < window_len; ++i) g_custom_square_window[i] = (float*)malloc(window_len*sizeof(float));
		bool flag = 1;
		for (int i = 0; i < window_len; ++i) flag = flag && (g_custom_square_window[i]);

		if (!flag){
			for (int i = 0; i < window_len; ++i)
				if (g_custom_square_window[i]) free(g_custom_square_window[i]);
			free(g_custom_square_window);
			g_custom_square_window = 0;
			printf("error: unable to malloc g_custom_square_window.\n");
			goto fail_or_end;
		}
		for (int i = 0; i < window_len; ++i)
			for (int j = 0; j < window_len; ++j)
				g_custom_square_window[i][j] = 1.0f/(window_len*window_len);

		g_custom_square_window_h = (float*)malloc(window_len*sizeof(float));
		if (!g_custom_square_window_h){
			printf("error: unable to malloc g_custom_square_window_h.\n");
			goto fail_or_end;
		}
		for (int i = 0; i < window_len; ++i) g_custom_square_window_h[i] = 1.0f/window_len;

		g_custom_square_window_v = (float*)malloc(window_len*sizeof(float));
		if (!g_custom_square_window_v){
			printf("error: unable to malloc g_custom_square_window_h.\n");
			goto fail_or_end;
		}
		for (int i = 0; i < window_len; ++i) g_custom_square_window_v[i] = g_custom_square_window_h[i]; /* Memory copy is likely faster than recomputing 1.0/window_size */

	    window.kernel = (float*)g_custom_square_window;
		window.kernel_h = (float*)g_custom_square_window_h;
		window.kernel_v = (float*)g_custom_square_window_v;

		if (window_stride != 1){
		    printf("warning: can only use window_stride=1 when using _iqa_convolve. Ignoring input and setting window_stride=1");
			fflush(stdout);
			window_stride = 1;
		}

		window.stride = window_stride;
		window.bnd_opt = KBND_SYMMETRIC;
    #elif defined(USE_IQA_INTEGRAL_IMAGE_MEAN)
		// printf("Defining windows to use integral image.\n");
		/* When using integral images, windows do not need to be created explicitly */
		window.kernel = 0;
		window.kernel_h = 0;
		window.kernel_v = 0;
		window.stride = window_stride;
    #endif
		window.w = window.h = window_len;
		window.normalized = 1;
	}

    // printf("Finished creating windows.\n");
    /* convert image values to floats, forcing stride = width. */
    ref_f = (float*)malloc(w*h*sizeof(float));
    cmp_f = (float*)malloc(w*h*sizeof(float));
    if (!ref_f || !cmp_f) {
        if (ref_f) free(ref_f);
        if (cmp_f) free(cmp_f);
        printf("error: unable to malloc ref_f or cmp_f.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    for (y=0; y<h; ++y) {
        src_offset = y * stride;
        offset = y * w;
        for (x=0; x<w; ++x, ++offset, ++src_offset) {
            ref_f[offset] = (float)ref[src_offset];
            cmp_f[offset] = (float)cmp[src_offset];
        }
    }

    /* scale the images down if required */
    if (scale > 1) {
        /* generate simple low-pass filter */
        low_pass.kernel = (float*)malloc(scale*scale*sizeof(float));
        low_pass.kernel_h = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        low_pass.kernel_v = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        if (!(low_pass.kernel && low_pass.kernel_h && low_pass.kernel_v)) { /* zli-nflx */
            free(ref_f);
            free(cmp_f);
            if (low_pass.kernel) free(low_pass.kernel); /* zli-nflx */
            if (low_pass.kernel_h) free(low_pass.kernel_h); /* zli-nflx */
            if (low_pass.kernel_v) free(low_pass.kernel_v); /* zli-nflx */
            printf("error: unable to malloc low-pass filter kernel.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        low_pass.w = low_pass.h = scale;
        low_pass.normalized = 0;
        low_pass.bnd_opt = KBND_SYMMETRIC;
        for (offset=0; offset<scale*scale; ++offset)
            low_pass.kernel[offset] = 1.0f/(scale*scale);
        for (offset=0; offset<scale; ++offset)  /* zli-nflx */
            low_pass.kernel_h[offset] = 1.0f/(scale); /* zli-nflx */
        for (offset=0; offset<scale; ++offset) /* zli-nflx */
            low_pass.kernel_v[offset] = 1.0f/(scale); /* zli-nflx */

        /* resample */
        if (_iqa_decimate(ref_f, w, h, scale, &low_pass, 0, 0, 0) ||
            _iqa_decimate(cmp_f, w, h, scale, &low_pass, 0, &w, &h)) { /* update w/h */
            free(ref_f);
            free(cmp_f);
            free(low_pass.kernel);
            free(low_pass.kernel_h); /* zli-nflx */
            free(low_pass.kernel_v); /* zli-nflx */
            printf("error: decimation fails on ref_f or cmp_f.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        free(low_pass.kernel);
        free(low_pass.kernel_h); /* zli-nflx */
        free(low_pass.kernel_v); /* zli-nflx */
    }
	// printf("Calling _iqa_ssim.\n");
    result = _iqa_ssim(ref_f, cmp_f, w, h, &window, &mr, args, &l, &c, &s);
    // printf("Finished _iqa_ssim.\n");

	// printf("Freeing float versions of ref and cmp.\n");
	// printf("%f %f\n", ref_f[0], cmp_f[0]);
    free(ref_f);
    free(cmp_f);
    free(g_custom_square_window);
	free(g_custom_square_window_h);
	free(g_custom_square_window_v);
	if (!score){
		fprintf(stderr, "No destination to return result in compute_ssim.\n");
		fflush(stderr);
		goto fail_or_end;
	}
    *score = (double)result;
	if (l_score) *l_score = (double)l;
    if (c_score) *c_score = (double)c;
	if (s_score) *s_score = (double)s;
    
	ret = 0;
fail_or_end:
	// printf("Exiting compute_ssim.\n");
    return ret;
}

