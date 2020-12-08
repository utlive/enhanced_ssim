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

#include <cstdio>
#include <stdlib.h>
#include <stdexcept>
#include <exception>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>

extern "C" {
#include "read_frame.h"
#include "mem.h"
#include "convolve.h"
#include "ssim.h"
#include "iqa_options.h"
}

static char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

static bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

static void print_usage(int argc, char *argv[])
{
    fprintf(stderr, "Usage: %s fmt width height ref_path dis_path [--window-type window_type] [--window-len window_len] [--window-stride window_stride] [--distance-to-height-ratio d2h] [--spatial-aggregation-method] spatial_aggregation_method\n", argv[0]);
    fprintf(stderr, "fmt:\n\tyuv420p\n\tyuv422p\n\tyuv444p\n\tyuv420p10le\n\tyuv422p10le\n\tyuv444p10le\n\tyuv420p12le\n\tyuv422p12le\n\tyuv444p12le\n\tyuv420p16le\n\tyuv422p16le\n\tyuv444p16le\n\n");
    fprintf(stderr, "window_type: ffmpeg_square\n\tgaussian\n\tcustom_square (default)\n\n");
	fprintf(stderr, "window_len: 1 <= window_len <= min(width, height) (default: 11)\n\n");
	fprintf(stderr, "window_stride: window_stride >= 1 (default)\n\n");
	fprintf(stderr, "d2h: d2h > 0 (default: 6.0)\n\n");
	fprintf(stderr, "spatial_aggregation_method: mean, cov (default)\n\n");
}

#if MEM_LEAK_TEST_ENABLE
/*
 * Measures the current (and peak) resident and virtual memories
 * usage of your linux C process, in kB
 */
static void getMemory(int itr_ctr, int state)
{
	int currRealMem;
	int peakRealMem;
	int currVirtMem;
	int peakVirtMem;
	char state_str[10]="";
    // stores each word in status file
    char buffer[1024] = "";
	
	if(state ==1)
		strcpy(state_str,"start");
	else
		strcpy(state_str,"end");
		
    // linux file contains this-process info
    FILE* file = fopen("/proc/self/status", "r");

    // read the entire file
    while (fscanf(file, " %1023s", buffer) == 1)
	{
        if (strcmp(buffer, "VmRSS:") == 0)
		{
            fscanf(file, " %d", &currRealMem);
        }
        if (strcmp(buffer, "VmHWM:") == 0)
		{
            fscanf(file, " %d", &peakRealMem);
        }
        if (strcmp(buffer, "VmSize:") == 0)
		{
            fscanf(file, " %d", &currVirtMem);
        }
        if (strcmp(buffer, "VmPeak:") == 0)
		{
            fscanf(file, " %d", &peakVirtMem);
        }
    }
    fclose(file);
    printf("Iteration %d at %s of process: currRealMem: %6d, peakRealMem: %6d, currVirtMem: %6d, peakVirtMem: %6d\n",itr_ctr, state_str, currRealMem, peakRealMem, currVirtMem, peakVirtMem);
}
#endif

static int run_wrapper(char *fmt, int width, int height, char *ref_path, char *dis_path, int window_type, int window_len, int window_stride, float d2h, int spatial_aggregation_method)
{
    int ret = 0;
    struct data *s;
	size_t data_stride = 0;
	size_t data_sz = 0;
	float* ref_buf = 0;
    float* dis_buf = 0;
	float* temp_buf = 0;
    double* fr_scores = 0;
    float ssim_score = 0;
    s = (struct data *)malloc(sizeof(struct data));
    s->format = fmt;
    s->width = width;
    s->height = height;
    s->ref_rfile = NULL;
    s->dis_rfile = NULL;
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv420p12le") || !strcmp(fmt, "yuv420p16le"))
    {
        if ((width * height) % 2 != 0)
        {
            fprintf(stderr, "(width * height) %% 2 != 0, width = %d, height = %d.\n", width, height);
            ret = 1;
            goto fail_or_end;
        }
        s->offset = width * height / 2;
    }
    else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv422p12le") || !strcmp(fmt, "yuv422p16le"))
    {
        s->offset = width * height;
    }
    else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le") || !strcmp(fmt, "yuv444p12le") || !strcmp(fmt, "yuv444p16le"))
    {
        s->offset = width * height * 2;
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        ret = 1;
        goto fail_or_end;
    }

    if (!(s->ref_rfile = fopen(ref_path, "rb")))
    {
        fprintf(stderr, "fopen ref_path %s failed.\n", ref_path);
        ret = 1;
        goto fail_or_end;
    }


    if (!(s->dis_rfile = fopen(dis_path, "rb")))
    {
        fprintf(stderr, "fopen dis_path %s failed.\n", dis_path);
        ret = 1;
        goto fail_or_end;
    }

    if (strcmp(ref_path, "-"))
    {
#ifdef _WIN32
        struct _stat64 ref_stat;
        if (!_stat64(ref_path, &ref_stat))
#else
        struct stat ref_stat;
        if (!stat(ref_path, &ref_stat))
#endif
        {
            size_t frame_size = width * height + s->offset;
            if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le")
                || !strcmp(fmt, "yuv420p12le") || !strcmp(fmt, "yuv422p12le") || !strcmp(fmt, "yuv444p12le")
                || !strcmp(fmt, "yuv420p16le") || !strcmp(fmt, "yuv422p16le") || !strcmp(fmt, "yuv444p16le")
            )
            {
                frame_size *= 2;
            }

            s->num_frames = ref_stat.st_size / frame_size;
        }
        else
        {
            s->num_frames = -1;
        }
    }
    else
    {
        s->num_frames = -1;
    }

	if (s->num_frames <= 0)
		goto fail_or_end;

	data_stride = width * sizeof(float);
	data_sz = (size_t)data_stride * height;

	ref_buf = (float*)aligned_malloc(data_sz, MAX_ALIGN);
	dis_buf = (float*)aligned_malloc(data_sz, MAX_ALIGN);
	temp_buf = (float*)aligned_malloc(2*data_sz, MAX_ALIGN); // Used to store u and v components by read_frame
	fr_scores = (double*)aligned_malloc(s->num_frames*sizeof(double), MAX_ALIGN);

	if (!ref_buf || !dis_buf || !temp_buf || !fr_scores){
		fprintf(stderr, "aligned malloc failed to allocate memory for ref, dis and temp.\n");
		goto fail_or_end;
    }

	for (int frame = 0; frame < s->num_frames; ++frame){
	    ret = read_frame(ref_buf, dis_buf, temp_buf, data_stride, s);
		if (ret){
		    fprintf(stderr, "Error reading frame from file,\n");
			goto fail_or_end;
		}
		ret = compute_ssim(ref_buf, dis_buf, s->width, s->height, data_stride, data_stride, window_type, window_len, window_stride, d2h, spatial_aggregation_method, fr_scores + frame, 0, 0, 0, (frame == s->num_frames-1)); /* Clear windows after the last frame */
		if (ret){
		    fprintf(stderr, "compute ssim failed.\n");
			goto fail_or_end;
		}
	}

	ssim_score = 0;
    for (int frame = 0; frame < s->num_frames; ++frame) ssim_score += fr_scores[frame];
	ssim_score /= s->num_frames;
    printf("SSIM: %f\n", ssim_score);

fail_or_end:
    if (s->ref_rfile)
    {
        fclose(s->ref_rfile);
    }
    if (s->dis_rfile)
    {
        fclose(s->dis_rfile);
    }
    if (s)
    {
        free(s);
    }
	if (ref_buf)
    {
		aligned_free(ref_buf);
    }
	if (dis_buf)
    {
		aligned_free(dis_buf);
    }
	if (temp_buf)
    {
		aligned_free(temp_buf);
    }
	if (fr_scores)
    {
	    aligned_free(fr_scores);
	}
    return ret;
}


int main(int argc, char *argv[])
{
    char* fmt;
    int width;
    int height;
    char* ref_path;
    char* dis_path;
	int window_type;
	int window_len;
	int window_stride;
	float d2h;
	int spatial_aggregation_method;
#if MEM_LEAK_TEST_ENABLE	
	int itr_ctr;
	int ret = 0;
#endif
	char *temp;

    /* Check parameters */
    /* Usage: %s fmt width height ref_path dis_path [--window-type window_type] [--window-len window_len] [--window-stride window_stride] [--scale-method scale_method] */
    if (argc < 6)
    {
        print_usage(argc, argv);
        return -1;
    }

    fmt = argv[1];

    try
    {
        width = std::stoi(argv[2]);
        height = std::stoi(argv[3]);
    }
    catch (std::logic_error& e)
    {
        fprintf(stderr, "Error: Invalid width/height format: %s\n", e.what());
        print_usage(argc, argv);
        return -1;
    }

    if (width <= 0 || height <= 0)
    {
        fprintf(stderr, "Error: Invalid width/height value: %d, %d\n", width, height);
        print_usage(argc, argv);
        return -1;
    }

    ref_path = argv[4];
    dis_path = argv[5];

    temp = getCmdOption(argv + 6, argv + argc, "--window-type");
    if (temp)
    {
		if (!strcmp(temp, "ffmpeg_square")) window_type = FFMPEG_SQUARE;
		else if (!strcmp(temp, "gaussian")) window_type = GAUSSIAN;
		else if (!strcmp(temp, "custom_square")) window_type = CUSTOM_SQUARE;
		else
        {
            fprintf(stderr, "Error: Invalid window type: %s\n", temp);
            print_usage(argc, argv);
            return -1;
        }
    }
	else{
		window_type = CUSTOM_SQUARE;
    }

    temp = getCmdOption(argv + 6, argv + argc, "--window-len");
    if (temp)
    {
        try
        {
            window_len = std::stoi(temp);
        }
        catch (std::logic_error& e)
        {
            fprintf(stderr, "Error: Invalid window length format: %s\n", e.what());
            print_usage(argc, argv);
            return -1;
        }
    }
	else{
		window_len = (window_type == FFMPEG_SQUARE)? 8: 11;
    }

    if (window_len <= 0 || window_len > std::max(width, height))
    {
        fprintf(stderr, "Error: Invalid window length value: %d\n", window_len);
        print_usage(argc, argv);
        return -1;
    }
	
	if (window_type == FFMPEG_SQUARE && window_len != 8){
		fprintf(stderr, "Warning: FFMPEG window must be of size 8. Ignoring input.");	
		window_len = 8;
	}
	else if (window_type == GAUSSIAN && window_len != 11){
		fprintf(stderr, "Warning: Gaussian window must be of size 11. Ignoring input.");	
		window_len = 11;
	}

	temp = getCmdOption(argv + 6, argv + argc, "--window-stride");
    if (temp)
    {
        try
        {
            window_stride = std::stoi(temp);
        }
        catch (std::logic_error& e)
        {
            fprintf(stderr, "Error: Invalid window stride format: %s\n", e.what());
            print_usage(argc, argv);
            return -1;
        }
    }
	else{
		window_stride = 1;
    }

    if (window_stride <= 0)
    {
        fprintf(stderr, "Error: Invalid window stride value: %d\n", window_stride);
        print_usage(argc, argv);
        return -1;
    }

	if (window_type != CUSTOM_SQUARE && window_len != 8){
		fprintf(stderr, "Warning: Only custom window can have a stride greater than 1. Ignoring input.");	
		window_stride = 8;
	}

	temp = getCmdOption(argv + 6, argv + argc, "--distance-to-height-ratio");
    if (temp)
    {
        try
        {
            d2h = std::stof(temp);
        }
        catch (std::logic_error& e)
        {
            fprintf(stderr, "Error: Invalid distance to height ratio format: %s\n", e.what());
            print_usage(argc, argv);
            return -1;
        }
    }
	else{
		d2h = 6.0;
    }

    if (d2h <= 0)
    {
        fprintf(stderr, "Error: Invalid distance to height ratio value: %d\n", window_stride);
        print_usage(argc, argv);
        return -1;
    }

    temp = getCmdOption(argv + 6, argv + argc, "--spatial-aggregation-method");
    if (temp)
    {
		if (!strcmp(temp, "cov")) spatial_aggregation_method = COV_POOLING;
		else if (!strcmp(temp, "mean")) spatial_aggregation_method = MEAN_POOLING;
		else
        {
            fprintf(stderr, "Error: Invalid spatial aggregation method: %s\n", temp);
            print_usage(argc, argv);
            return -1;
        }
    }
	else{
		window_type = COV_POOLING;
    }

    try
    {
#if MEM_LEAK_TEST_ENABLE
		for(itr_ctr=0;itr_ctr<1000;itr_ctr++)
		{
			getMemory(itr_ctr,1);
			ret = run_wrapper(fmt, width, height, ref_path, dis_path, window_type, window_len, window_stride, d2h, spatial_aggregation_method);
			getMemory(itr_ctr,2);
		}
#else
		return run_wrapper(fmt, width, height, ref_path, dis_path, window_type, window_len, window_stride, d2h, spatial_aggregation_method);
#endif
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
        print_usage(argc, argv);
        return -1;
    }
    
}
