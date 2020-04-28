// test.cpp

/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _CRT_SECURE_NO_WARNINGS 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX
#include <windows.h>
#endif

#define USE_AVX2 0

// float4
#include "cppspmd_float4.h"
#include "test_kernel.h"
#include "mandelbrot.h"

// SSE4.1
#include "cppspmd_sse.h"
#include "test_kernel.h"
#include "mandelbrot.h"

// AVX2
#define CPPSPMD_USE_AVX2 0
#undef CPPSPMD_USE_FMA
#include "cppspmd_avx2.h"
#include "test_kernel.h"
#include "mandelbrot.h"

// AVX1 alt
#include "cppspmd_avx1.h"
#include "test_kernel.h"
#include "mandelbrot.h"

// AVX1
#undef CPPSPMD_USE_AVX2
#define CPPSPMD_USE_AVX2 1
#undef CPPSPMD_USE_FMA
#include "cppspmd_avx2.h"
#include "test_kernel.h"
#include "mandelbrot.h"

// AVX2 FMA
#undef CPPSPMD_USE_AVX2
#undef CPPSPMD_USE_FMA
#define CPPSPMD_USE_AVX2 1
#define CPPSPMD_USE_FMA 1
#include "cppspmd_avx2.h"
#include "test_kernel.h"
#include "mandelbrot.h"

typedef uint64_t timer_ticks;

class interval_timer
{
public:
	interval_timer();

	void start();
	void stop();

	double get_elapsed_secs() const;
	inline double get_elapsed_ms() const { return 1000.0f* get_elapsed_secs(); }
		
	static void init();
	static inline timer_ticks get_ticks_per_sec() { return g_freq; }
	static timer_ticks get_ticks();
	static double ticks_to_secs(timer_ticks ticks);
	static inline double ticks_to_ms(timer_ticks ticks) {	return ticks_to_secs(ticks) * 1000.0f; }

private:
	static timer_ticks g_init_ticks, g_freq;
	static double g_timer_freq;

	timer_ticks m_start_time, m_stop_time;

	bool m_started, m_stopped;
};

uint64_t interval_timer::g_init_ticks, interval_timer::g_freq;
double interval_timer::g_timer_freq;

#if defined(_WIN32)
inline void query_counter(timer_ticks* pTicks)
{
	QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(pTicks));
}
inline void query_counter_frequency(timer_ticks* pTicks)
{
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(pTicks));
}
#elif defined(__APPLE__)
#include <sys/time.h>
inline void query_counter(timer_ticks* pTicks)
{
	struct timeval cur_time;
	gettimeofday(&cur_time, NULL);
	*pTicks = static_cast<unsigned long long>(cur_time.tv_sec) * 1000000ULL + static_cast<unsigned long long>(cur_time.tv_usec);
}
inline void query_counter_frequency(timer_ticks* pTicks)
{
	*pTicks = 1000000;
}
#elif defined(__GNUC__)
#include <sys/timex.h>
inline void query_counter(timer_ticks* pTicks)
{
	struct timeval cur_time;
	gettimeofday(&cur_time, NULL);
	*pTicks = static_cast<unsigned long long>(cur_time.tv_sec) * 1000000ULL + static_cast<unsigned long long>(cur_time.tv_usec);
}
inline void query_counter_frequency(timer_ticks* pTicks)
{
	*pTicks = 1000000;
}
#else
#error TODO
#endif
				
interval_timer::interval_timer() : m_start_time(0), m_stop_time(0), m_started(false), m_stopped(false)
{
	if (!g_timer_freq)
		init();
}

void interval_timer::start()
{
	query_counter(&m_start_time);
	m_started = true;
	m_stopped = false;
}

void interval_timer::stop()
{
	assert(m_started);
	query_counter(&m_stop_time);
	m_stopped = true;
}

double interval_timer::get_elapsed_secs() const
{
	assert(m_started);
	if (!m_started)
		return 0;

	timer_ticks stop_time = m_stop_time;
	if (!m_stopped)
		query_counter(&stop_time);

	timer_ticks delta = stop_time - m_start_time;
	return delta * g_timer_freq;
}
		
void interval_timer::init()
{
	if (!g_timer_freq)
	{
		query_counter_frequency(&g_freq);
		g_timer_freq = 1.0f / g_freq;
		query_counter(&g_init_ticks);
	}
}

timer_ticks interval_timer::get_ticks()
{
	if (!g_timer_freq)
		init();
	timer_ticks ticks;
	query_counter(&ticks);
	return ticks - g_init_ticks;
}

double interval_timer::ticks_to_secs(timer_ticks ticks)
{
	if (!g_timer_freq)
		init();
	return ticks * g_timer_freq;
}


/* Write a PPM image file with the image of the Mandelbrot set */
inline void writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

/* Write a PPM image file with the image */
inline void writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0.f) v = 0.f;
        else if (v > 255.f) v = 255.f;
        unsigned char c = (unsigned char)v;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

static int mandel_c(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

void mandelbrot_c(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel_c(x, y, maxIterations);
        }
    }
}

int test_mandel()
{
	unsigned int width = 768;
	unsigned int height = 512;
	float x0 = -2;
	float x1 = 1;
	float y0 = -1;
	float y1 = 1;

	int maxIterations = 256;
	int* buf = new int[width * height];

	int num_runs = 10;

	double cheapest_run = 1e+10f;
	interval_timer otm;

	printf("test_mandel:\n");
		
	double t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		cppspmd_float4::spmd_call<mandelbrot_float4>(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("float4 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_float4.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		cppspmd_avx1::spmd_call<mandelbrot_avx1>(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX1 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX1.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		cppspmd_avx1_alt::spmd_call<mandelbrot_avx1_alt>(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX1 alt time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX1_alt.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		cppspmd_sse41::spmd_call<mandelbrot_sse41>(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("SSE4.1 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_SSE41.ppm");

#if USE_AVX2		
	t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		cppspmd_avx2::spmd_call<mandelbrot_avx2>(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX2 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX2.ppm");
#endif

	t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		mandelbrot_c(x0, y0, x1, y1,
			         width, height, maxIterations,
				      buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("C time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_C.ppm");
		
	return true;
}

bool test()
{
   FILE *pFile = fopen("float4.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "float4:\n");
	cppspmd_float4::spmd_call<test_kernel_float4>(pFile);
	fclose(pFile);
	printf("Wrote file float4.txt\n");
	
	pFile = fopen("sse41.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "sse41:\n");
	cppspmd_sse41::spmd_call<test_kernel_sse41>(pFile);
	fclose(pFile);
	printf("Wrote file sse41.txt\n");

	pFile = fopen("avx1_alt.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx1_alt:\n");
	cppspmd_avx1_alt::spmd_call<test_kernel_avx1_alt>(pFile);
	fclose(pFile);
	printf("Wrote file avx1_alt.txt\n");

	pFile = fopen("avx1.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx1:\n");
	cppspmd_avx1::spmd_call<test_kernel_avx1>(pFile);
	fclose(pFile);
	printf("Wrote file avx1.txt\n");

#if USE_AVX2
	pFile = fopen("avx2.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx2:\n");
	cppspmd_avx2::spmd_call<test_kernel_avx2>(pFile);
	fclose(pFile);
	printf("Wrote file avx2.txt\n");
#endif

	return true;
}

int main(int argc, char *arg_v[])
{
	test();

	test_mandel();

	return EXIT_SUCCESS;
}