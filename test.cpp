// test.cpp
// Important: This test app requires AVX-512. It does not yet automatically dispatch.

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

#ifndef CPPSPMD_GLUER
#define CPPSPMD_GLUER(a, b) a##b
#endif
#ifndef CPPSPMD_GLUER2
#define CPPSPMD_GLUER2(a, b) CPPSPMD_GLUER(a, b)
#endif

// float4
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _float4)

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

// SSE2
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _sse2)
#define CPPSPMD_SSE2 1

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"
#undef CPPSPMD_SSE2

// SSE4.1
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _sse41)
#define CPPSPMD_SSE41 1
#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

#undef CPPSPMD_SSE41

// AVX1
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _avx1)

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

// AVX1 alt
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _avx1_alt)

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

// AVX2 FMA
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _avx2_fma)

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

// int16 AVX2 FMA
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _int16_avx2_fma)

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"

// AVX-512
#undef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, _avx512)
#undef CPPSPMD_AVX512
#define CPPSPMD_AVX512 1

#include "test_kernel_declares.h"
#include "mandelbrot_declares.h"
#include "rt_kernel_declares.h"
#include "simple_declares.h"
#include "volume_kernel_declares.h"
#include "noise_kernel_declares.h"
#include "options_declares.h"
#include "ao_bench_declares.h"

#undef CPPSPMD_AVX512

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

//------------------------------------------------------------------------------------------------
// Mandelbrot

/* Write a PPM image file with the image of the Mandelbrot set */
inline void writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (char)((buf[i] & 0x1) ? 240 : 20);
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

	interval_timer otm;

	printf("test_mandel:\n");
		
	double t = 1e+10f;
	for (uint32_t i = 0; i < 10; i++)
	{
		otm.start();
		mandelbrot_float4(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("float4 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_float4.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_avx1(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX1 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX1.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_avx1_alt(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX1 alt time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX1_alt.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_sse2(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("SSE2 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_SSE2.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_sse41(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("SSE4.1 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_SSE41.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_avx2_fma(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX2 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX2.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
	{
		otm.start();
		mandelbrot_avx512(x0, y0, x1, y1, width, height, maxIterations, buf);
		t = std::min<double>(t, otm.get_elapsed_secs()); 
	}
	printf("AVX512 time: %f\n", t);
	writePPM(buf, width, height, "mandelbrot_AVX512.ppm");

	t = 1e+10f;
	for (uint32_t i = 0; i < num_runs; i++)
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

//------------------------------------------------------------------------------------------------
// Low-level test

bool test()
{
   FILE *pFile = fopen("float4.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "float4:\n");
	cppspmd_lowlevel_test_float4(pFile);
	fclose(pFile);
	printf("Wrote file float4.txt\n");
	
	pFile = fopen("sse2.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "sse2:\n");
	cppspmd_lowlevel_test_sse2(pFile);
	fclose(pFile);
	printf("Wrote file sse2.txt\n");

	pFile = fopen("sse41.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "sse41:\n");
	cppspmd_lowlevel_test_sse41(pFile);
	fclose(pFile);
	printf("Wrote file sse41.txt\n");

	pFile = fopen("avx1_alt.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx1_alt:\n");
	cppspmd_lowlevel_test_avx1_alt(pFile);
	fclose(pFile);
	printf("Wrote file avx1_alt.txt\n");

	pFile = fopen("avx1.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx1:\n");
	cppspmd_lowlevel_test_avx1(pFile);
	fclose(pFile);
	printf("Wrote file avx1.txt\n");
		
	pFile = fopen("avx2_fma.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx2_fma:\n");
	cppspmd_lowlevel_test_avx2_fma(pFile);
	fclose(pFile);
	printf("Wrote file avx2_fma.txt\n");

	pFile = fopen("avx512.txt", "w");
	if (!pFile) return false;
	fprintf(pFile, "avx512:\n");
	cppspmd_lowlevel_test_avx512(pFile);
	fclose(pFile);
	printf("Wrote file avx512.txt\n");

	return true;
}

//------------------------------------------------------------------------------------------------
// Ray trace test

static void writeImage(int* idImage, float* depthImage, int width, int height, const char* filename) {
	FILE* f = fopen(filename, "wb");
	if (!f) {
		perror(filename);
		exit(1);
	}

	fprintf(f, "P6\n%d %d\n255\n", width, height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			// use the bits from the object id of the hit object to make a
			// random color
			int id = idImage[y * width + x];
			unsigned char r = 0, g = 0, b = 0;

			for (int i = 0; i < 8; ++i) {
				// extract bit 3*i for red, 3*i+1 for green, 3*i+2 for blue
				int rbit = (id & (1 << (3 * i))) >> (3 * i);
				int gbit = (id & (1 << (3 * i + 1))) >> (3 * i + 1);
				int bbit = (id & (1 << (3 * i + 2))) >> (3 * i + 2);
				// and then set the bits of the colors starting from the
				// high bits...
				r |= rbit << (7 - i);
				g |= gbit << (7 - i);
				b |= bbit << (7 - i);
			}
			fputc(r, f);
			fputc(g, f);
			fputc(b, f);
		}
	}
	fclose(f);
	printf("Wrote image file %s\n", filename);
}

///--------------
// Just enough of a float3 class to do what we need in this file.
#ifdef _MSC_VER
__declspec(align(16))
#endif
struct float3 {
	float3() {}
	float3(float xx, float yy, float zz) {
		x = xx;
		y = yy;
		z = zz;
	}

	float3 operator*(float f) const { return float3(x * f, y * f, z * f); }
	float3 operator-(const float3& f2) const { return float3(x - f2.x, y - f2.y, z - f2.z); }
	float3 operator*(const float3& f2) const { return float3(x * f2.x, y * f2.y, z * f2.z); }
	float x, y, z;
	float pad; // match padding/alignment of ispc version
}
#ifndef _MSC_VER
__attribute__((aligned(16)))
#endif
;

struct Ray {
	float3 origin, dir, invDir;
	unsigned int dirIsNeg[3];
	float mint, maxt;
	int hitId;
};

inline float3 Cross(const float3& v1, const float3& v2) {
	float v1x = v1.x, v1y = v1.y, v1z = v1.z;
	float v2x = v2.x, v2y = v2.y, v2z = v2.z;
	float3 ret;
	ret.x = (v1y * v2z) - (v1z * v2y);
	ret.y = (v1z * v2x) - (v1x * v2z);
	ret.z = (v1x * v2y) - (v1y * v2x);
	return ret;
}

inline float Dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static void generateRay(const float raster2camera[4][4], const float camera2world[4][4], float x, float y, Ray& ray) {
	ray.mint = 0.f;
	ray.maxt = 1e30f;

	ray.hitId = 0;

	// transform raster coordinate (x, y, 0) to camera space
	float camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
	float camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
	float camz = raster2camera[2][3];
	float camw = raster2camera[3][3];
	camx /= camw;
	camy /= camw;
	camz /= camw;

	ray.dir.x = camera2world[0][0] * camx + camera2world[0][1] * camy + camera2world[0][2] * camz;
	ray.dir.y = camera2world[1][0] * camx + camera2world[1][1] * camy + camera2world[1][2] * camz;
	ray.dir.z = camera2world[2][0] * camx + camera2world[2][1] * camy + camera2world[2][2] * camz;

	ray.origin.x = camera2world[0][3] / camera2world[3][3];
	ray.origin.y = camera2world[1][3] / camera2world[3][3];
	ray.origin.z = camera2world[2][3] / camera2world[3][3];

	ray.invDir.x = 1.f / ray.dir.x;
	ray.invDir.y = 1.f / ray.dir.y;
	ray.invDir.z = 1.f / ray.dir.z;

	ray.dirIsNeg[0] = (ray.invDir.x < 0) ? 1 : 0;
	ray.dirIsNeg[1] = (ray.invDir.y < 0) ? 1 : 0;
	ray.dirIsNeg[2] = (ray.invDir.z < 0) ? 1 : 0;
}

static inline bool BBoxIntersect(const float bounds[2][3], const Ray& ray) {
	float3 bounds0(bounds[0][0], bounds[0][1], bounds[0][2]);
	float3 bounds1(bounds[1][0], bounds[1][1], bounds[1][2]);
	float t0 = ray.mint, t1 = ray.maxt;

	float3 tNear = (bounds0 - ray.origin) * ray.invDir;
	float3 tFar = (bounds1 - ray.origin) * ray.invDir;
	if (tNear.x > tFar.x) {
		float tmp = tNear.x;
		tNear.x = tFar.x;
		tFar.x = tmp;
	}
	t0 = std::max(tNear.x, t0);
	t1 = std::min(tFar.x, t1);

	if (tNear.y > tFar.y) {
		float tmp = tNear.y;
		tNear.y = tFar.y;
		tFar.y = tmp;
	}
	t0 = std::max(tNear.y, t0);
	t1 = std::min(tFar.y, t1);

	if (tNear.z > tFar.z) {
		float tmp = tNear.z;
		tNear.z = tFar.z;
		tFar.z = tmp;
	}
	t0 = std::max(tNear.z, t0);
	t1 = std::min(tFar.z, t1);

	return (t0 <= t1);
}

inline bool TriIntersect(const Triangle& tri, Ray& ray) {
	float3 p0(tri.p[0][0], tri.p[0][1], tri.p[0][2]);
	float3 p1(tri.p[1][0], tri.p[1][1], tri.p[1][2]);
	float3 p2(tri.p[2][0], tri.p[2][1], tri.p[2][2]);
	float3 e1 = p1 - p0;
	float3 e2 = p2 - p0;

	float3 s1 = Cross(ray.dir, e2);
	float divisor = Dot(s1, e1);

	if (divisor == 0.)
		return false;
	float invDivisor = 1.f / divisor;

	// Compute first barycentric coordinate
	float3 d = ray.origin - p0;
	float b1 = Dot(d, s1) * invDivisor;
	if (b1 < 0. || b1 > 1.)
		return false;

	// Compute second barycentric coordinate
	float3 s2 = Cross(d, e1);
	float b2 = Dot(ray.dir, s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
		return false;

	// Compute _t_ to intersection point
	float t = Dot(e2, s2) * invDivisor;
	if (t < ray.mint || t > ray.maxt)
		return false;

	ray.maxt = t;
	ray.hitId = tri.id;
	return true;
}

bool BVHIntersect(const LinearBVHNode nodes[], const Triangle tris[], Ray& r) {
	Ray ray = r;
	bool hit = false;
	// Follow ray through BVH nodes to find primitive intersections
	int todoOffset = 0, nodeNum = 0;
	int todo[64];

	while (true) {
		// Check ray against BVH node
		const LinearBVHNode& node = nodes[nodeNum];
		if (BBoxIntersect(node.bounds, ray)) {
			unsigned int nPrimitives = node.nPrimitives;
			if (nPrimitives > 0) {
				// Intersect ray with primitives in leaf BVH node
				unsigned int primitivesOffset = node.offset;
				for (unsigned int i = 0; i < nPrimitives; ++i) {
					if (TriIntersect(tris[primitivesOffset + i], ray))
						hit = true;
				}
				if (todoOffset == 0)
					break;
				nodeNum = todo[--todoOffset];
			}
			else {
				// Put far BVH node on _todo_ stack, advance to near node
				if (r.dirIsNeg[node.splitAxis]) {
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node.offset;
				}
				else {
					todo[todoOffset++] = node.offset;
					nodeNum = nodeNum + 1;
				}
			}
		}
		else {
			if (todoOffset == 0)
				break;
			nodeNum = todo[--todoOffset];
		}
	}
	r.maxt = ray.maxt;
	r.hitId = ray.hitId;

	return hit;
}

void raytrace_serial(int width, int height, int baseWidth, int baseHeight, const float raster2camera[4][4],
	const float camera2world[4][4], float image[], int id[], const LinearBVHNode nodes[],
	const Triangle triangles[]) {
	float widthScale = float(baseWidth) / float(width);
	float heightScale = float(baseHeight) / float(height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Ray ray;
			generateRay(raster2camera, camera2world, x * widthScale, y * heightScale, ray);
			BVHIntersect(nodes, triangles, ray);

			int offset = y * width + x;
			image[offset] = ray.maxt;
			id[offset] = ray.hitId;
		}
	}
}

//---------------

int test_rt() 
{
	float scale = 1.f;
	//const char* filename = "teapot";
	const char* filename = "sponza";
	//const char* filename = "cornell";

#define READ(var, n)                                                                                                   \
    if (fread(&(var), sizeof(var), n, f) != (unsigned int)n) {                                                         \
        fprintf(stderr, "Unexpected EOF reading scene file\n");                                                        \
        return 1;                                                                                                      \
    } else /* eat ; */

	//
	// Read the camera specification information from the camera file
	//
	char fnbuf[1024];
	sprintf(fnbuf, "%s.camera", filename);
	FILE* f = fopen(fnbuf, "rb");
	if (!f) {
		perror(fnbuf);
		return 1;
	}

	//
	// Nothing fancy, and trouble if we run on a big-endian system, just
	// fread in the bits
	//
	int baseWidth, baseHeight;
	float camera2world[4][4], raster2camera[4][4];
	READ(baseWidth, 1);
	READ(baseHeight, 1);
	READ(camera2world[0][0], 16);
	READ(raster2camera[0][0], 16);

	//
	// Read in the serialized BVH
	//
	sprintf(fnbuf, "%s.bvh", filename);
	f = fopen(fnbuf, "rb");
	if (!f) {
		perror(fnbuf);
		return 1;
	}

	// The BVH file starts with an int that gives the total number of BVH
	// nodes
	uint32_t nNodes;
	READ(nNodes, 1);

	LinearBVHNode* nodes = new LinearBVHNode[nNodes];
	for (unsigned int i = 0; i < nNodes; ++i) {
		// Each node is 6x floats for a boox, then an integer for an offset
		// to the second child node, then an integer that encodes the type
		// of node, the total number of int it if a leaf node, etc.
		float b[6];
		READ(b[0], 6);
		nodes[i].bounds[0][0] = b[0];
		nodes[i].bounds[0][1] = b[1];
		nodes[i].bounds[0][2] = b[2];
		nodes[i].bounds[1][0] = b[3];
		nodes[i].bounds[1][1] = b[4];
		nodes[i].bounds[1][2] = b[5];
		READ(nodes[i].offset, 1);
		READ(nodes[i].nPrimitives, 1);
		READ(nodes[i].splitAxis, 1);
		READ(nodes[i].pad, 1);
	}

	// And then read the triangles
	uint32_t nTris;
	READ(nTris, 1);
	Triangle* triangles = new Triangle[nTris];
	for (uint32_t i = 0; i < nTris; ++i) {
		// 9x floats for the 3 vertices
		float v[9];
		READ(v[0], 9);
		float* vp = v;
		for (int j = 0; j < 3; ++j) {
			triangles[i].p[j][0] = *vp++;
			triangles[i].p[j][1] = *vp++;
			triangles[i].p[j][2] = *vp++;
		}
		// And create an object id
		triangles[i].id = i + 1;
	}
	fclose(f);

	int height = int(baseHeight * scale);
	int width = int(baseWidth * scale);

	// allocate images; one to hold hit object ids, one to hold depth to
	// the first interseciton
	int* id = new int[width * height];
	float* image = new float[width * height];

	interval_timer tm;
	tm.start();

#ifdef _DEBUG
	const int T = 1;
#else
	const int T = 16;
#endif

	for (int i = 0; i < T; i++)
	{
#if 0
		raytrace_serial(width, height, width, height, raster2camera, camera2world, image, id, nodes, triangles);
#else
		for (int y = 0; y < height; y += TILE_SIZE)
		{
			const int tile_height = std::min(TILE_SIZE, height - y);

			for (int x = 0; x < width; x += TILE_SIZE)
			{
				const int tile_width = std::min(TILE_SIZE, width - x);

				raytrace_tile_avx512(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
				//raytrace_tile_avx2_fma(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
				//raytrace_tile_avx1(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
				//raytrace_tile_avx1_alt(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
				//raytrace_tile_sse41(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
				//raytrace_tile_sse2(x, std::min(x + tile_width, width), y, std::min(y + tile_height, height), width, height, raster2camera, camera2world, image, id, nodes, triangles);
			}
		}
#endif
	}

	printf("Elapsed: %f\n", tm.get_elapsed_secs());

	writeImage(id, image, width, height, "rt.ppm");

	return 0;
}

//------------------------------------------------------------------------------------------------

int test_simple() 
{
	printf("simple:\n");

    float vin[16], vout[16];

    // Initialize input buffer
    for (int i = 0; i < 16; ++i)
	{
        vin[i] = (float)i;
	}

	memset(vout, 0xDE, sizeof(vout));
    simple_float4(vin, vout, 16);

	// Print results
	printf("float4:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
    simple_sse41(vin, vout, 16);

	// Print results
	printf("SSE4.1:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
	simple_sse2(vin, vout, 16);

    // Print results
	printf("SSE2:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
	simple_avx2_fma(vin, vout, 16);

    // Print results
	printf("AVX2:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
	simple_avx1(vin, vout, 16);

    // Print results
	printf("AVX1:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
	simple_avx1_alt(vin, vout, 16);

    // Print results
	printf("AVX1_alt:\n");
    for (int i = 0; i < 16; ++i)
	{
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

	memset(vout, 0xDE, sizeof(vout));
	simple_avx512(vin, vout, 16);

    // Print results
	printf("AVX512:\n");
    for (int i = 0; i < 16; ++i)
	{
		printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
	}

    return 0;
}

//------------------------------------------------------------------------------------------------
// volume

/* Load image and viewing parameters from a camera data file.
   FIXME: we should add support to be able to specify viewing parameters
   in the program here directly. */
static void
loadCamera(const char* fn, int* width, int* height, float raster2camera[4][4],
	float camera2world[4][4]) {
	FILE* f = fopen(fn, "r");
	if (!f) {
		perror(fn);
		exit(1);
	}
	if (fscanf(f, "%d %d", width, height) != 2) {
		fprintf(stderr, "Unexpected end of file in camera file\n");
		exit(1);
	}

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (fscanf(f, "%f", &raster2camera[i][j]) != 1) {
				fprintf(stderr, "Unexpected end of file in camera file\n");
				exit(1);
			}
		}
	}
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (fscanf(f, "%f", &camera2world[i][j]) != 1) {
				fprintf(stderr, "Unexpected end of file in camera file\n");
				exit(1);
			}
		}
	}
	fclose(f);
}

/* Load a volume density file.  Expects the number of x, y, and z samples
   as the first three values (as integer strings), then x*y*z
   floating-point values (also as strings) to give the densities.  */
static float*
loadVolume(const char* fn, int n[3]) {
	FILE* f = fopen(fn, "r");
	if (!f) {
		perror(fn);
		exit(1);
	}

	if (fscanf(f, "%d %d %d", &n[0], &n[1], &n[2]) != 3) {
		fprintf(stderr, "Couldn't find resolution at start of density file\n");
		exit(1);
	}

	int count = n[0] * n[1] * n[2];
	float* v = new float[count];
	for (int i = 0; i < count; ++i) {
		if (fscanf(f, "%f", &v[i]) != 1) {
			fprintf(stderr, "Unexpected end of file at %d'th density value\n", i);
			exit(1);
		}
	}

	return v;
}

void test_volume()
{
	// Load viewing data and the volume density data
	int width, height;
	float raster2camera[4][4], camera2world[4][4];
	loadCamera("volume_assets/camera.dat", &width, &height, raster2camera, camera2world);
	float* image = new float[width * height];

	int n[3];
	float* density = loadVolume("volume_assets/density_highres.vol", n);

	interval_timer tm;
	tm.start();

	cppspmd_volume_avx512(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_avx2_fma(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_sse41(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_sse2(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_avx1(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_avx1_alt(density, n, raster2camera, camera2world, width, height, image);
	//cppspmd_volume_float4(density, n, raster2camera, camera2world, width, height, image);

	printf("Elapsed: %3.6f\n", tm.get_elapsed_secs());
		
	writePPM(image, width, height, "volume.ppm");
	
	printf("Wrote volume.ppm\b");
}

//------------------------------------------------------------------------------------------------
// noise

void test_noise()
{
	unsigned int width = 768;
	unsigned int height = 768;
	float x0 = -10;
	float x1 = 10;
	float y0 = -10;
	float y1 = 10;

	float* buf = new float[width * height];

	//
	// Compute the image using the ispc implementation; report the minimum
	// time of three runs.
	//
	noise_avx512(x0, y0, x1, y1, width, height, buf);
	//noise_sse41(x0, y0, x1, y1, width, height, buf);
	//noise_float4(x0, y0, x1, y1, width, height, buf);
	//noise_avx2_fma(x0, y0, x1, y1, width, height, buf);
	//noise_avx1(x0, y0, x1, y1, width, height, buf);
	//noise_avx1_alt(x0, y0, x1, y1, width, height, buf);

	writePPM(buf, width, height, "noise-cppspmd.ppm");

	delete[] buf;
}

//------------------------------------------------------------------------------------------------
// Options

#define BINOMIAL_NUM 64

// Cumulative normal distribution function
static inline float
CND(float X) {
    float L = fabsf(X);

    float k = 1.f / (1.f + 0.2316419f * L);
    float k2 = k*k;
    float k3 = k2*k;
    float k4 = k2*k2;
    float k5 = k3*k2;

    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * expf(-L * L * .5f);

    if (X > 0.f)
        w = 1.f - w;
    return w;
}


void
black_scholes(float Sa[], float Xa[], float Ta[],
              float ra[], float va[],
              float result[], int count) {
    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i];
        float T = Ta[i], r = ra[i];
        float v = va[i];

        float d1 = (logf(S/X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);

        result[i] = S * CND(d1) - X * expf(-r * T) * CND(d2);
    }
}


void
binomial_put(float Sa[], float Xa[], float Ta[],
             float ra[], float va[],
             float result[], int count) {
    float V[BINOMIAL_NUM];

    for (int i = 0; i < count; ++i) {
        float S = Sa[i], X = Xa[i];
        float T = Ta[i], r = ra[i];
        float v = va[i];

        float dt = T / BINOMIAL_NUM;
        float u = expf(v * sqrtf(dt));
        float d = 1.f / u;
        float disc = expf(r * dt);
        float Pu = (disc - d) / (u - d);

        for (int j = 0; j < BINOMIAL_NUM; ++j) {
            float upow = powf(u, (float)(2*j-BINOMIAL_NUM));
            V[j] = std::max(0.f, X - S * upow);
        }

        for (int j = BINOMIAL_NUM-1; j >= 0; --j)
            for (int k = 0; k < j; ++k)
                V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;

        result[i] = V[0];
    }
}

int test_options(bool scalar) 
{
	printf("Test options, scalar: %u\n", scalar);

    int nOptions = 128*1024;

    float *S = new float[nOptions];
    float *X = new float[nOptions];
    float *T = new float[nOptions];
    float *r = new float[nOptions];
    float *v = new float[nOptions];
    float *result = new float[nOptions];
		
	// Note: I've modified this from the original sample, which set all inputs to the same values (why?).
    for (int i = 0; i < nOptions; ++i) 
	{
		float f = (i / (float)nOptions);

        S[i] = 100 + f * 50;  // stock price
        X[i] = 98 + f * 10;   // option strike price
        T[i] = 2 + f * 1;    // time (years)
        r[i] = .02;  // risk-free interest rate
        v[i] = 5;    // volatility
    }

    int num_runs = 10;

    interval_timer tm;
	tm.start();

    // Binomial options pricing model
    for (int i = 0; i < num_runs; i++)
    {
		if (scalar)
			binomial_put(S, X, T, r, v, result, nOptions);
		else
		{
			binomial_put_avx512(S, X, T, r, v, result, nOptions);
			//binomial_put_float4(S, X, T, r, v, result, nOptions);
		}
    }
		
    printf("Binomial options:\n");
    printf("%3.6f secs\n", tm.get_elapsed_secs());

    FILE* binomial_csv = fopen(scalar ? "binomial_scalar.txt" : "binomial_simd.txt", "w");
    for (int i = 0; binomial_csv && i < nOptions; i++)
    {
        fprintf(binomial_csv, "%f\n", result[i]);
    }
    fclose(binomial_csv);

    num_runs = 10;

    tm.start();

    // Black-Scholes options pricing model
    for (int i = 0; i < num_runs; i++)
    {
		if (scalar)
			black_scholes(S, X, T, r, v, result, nOptions);
		else
		{
			black_scholes_avx512(S, X, T, r, v, result, nOptions);
			//black_scholes_float4(S, X, T, r, v, result, nOptions);
		}
    }
		
    printf("Black-Scholes:\n");
    printf("%3.6f secs\n", tm.get_elapsed_secs());

    FILE* black_scholes_csv = fopen(scalar ? "black_scholes_scalar.txt" : "black_scholes_simd.txt", "w");
    for (int i = 0; black_scholes_csv && i < nOptions; i++)
    {
        fprintf(black_scholes_csv, "%f\n", result[i]);
    }
    fclose(black_scholes_csv);
		
	delete [] S;
    delete [] X;
    delete [] T;
    delete [] r;
    delete [] v;
    delete [] result;

	return 0;
}

//------------------------------------------------------------------------------------------------
// ao bench

#define NSUBSAMPLES 2

extern void ao_serial(int w, int h, int nsubsamples, float image[]);

static unsigned int test_iterations[] = {3, 7, 1};
static unsigned int width = 800, height = 800;
static unsigned char *img;
static float *fimg;

static unsigned char clamp(float f) {
    int i = (int)(f * 255.5);

    if (i < 0)
        i = 0;
    if (i > 255)
        i = 255;

    return (unsigned char)i;
}

static void savePPM(const char *fname, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img[3 * (y * w + x) + 0] = clamp(fimg[3 * (y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 * (y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 * (y * w + x) + 2]);
        }
    }

    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror(fname);
        exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
    printf("Wrote image file %s\n", fname);
}

int test_ao()
{
	printf("test_ao:\n");
		
#ifdef _DEBUG
	//width = 64;
	//height = 64;
	test_iterations[0] = 1;
	test_iterations[1] = 1;
	test_iterations[2] = 1;
#endif

#if 0
    if (argc < 3) {
        printf("%s\n", argv[0]);
        printf("Usage: ao [width] [height] [ispc iterations] [tasks iterations] [serial iterations]\n");
        getchar();
        exit(-1);
    } else {
        if (argc == 6) {
            for (int i = 0; i < 3; i++) {
                test_iterations[i] = atoi(argv[3 + i]);
            }
        }
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
#endif
	
    // Allocate space for output images
    img = new unsigned char[width * height * 3];
    fimg = new float[width * height * 3];

	interval_timer tm;

    //
    // Run the ispc path, test_iterations times, and report the minimum
    // time for any of them.
    //
    double minTimeSIMD = 1e30;
    for (unsigned int i = 0; i < test_iterations[0]; i++) 
	{
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        assert(NSUBSAMPLES == 2);

        tm.start();

		//ao_sse2(width, height, NSUBSAMPLES, fimg);
		ao_sse41(width, height, NSUBSAMPLES, fimg);
		//ao_avx1(width, height, NSUBSAMPLES, fimg);
		//ao_avx1_alt(width, height, NSUBSAMPLES, fimg);
		//ao_avx2_fma(width, height, NSUBSAMPLES, fimg);
        //ao_avx512(width, height, NSUBSAMPLES, fimg);
		
        double t = tm.get_elapsed_secs();
        printf("@time of SIMD run:\t\t\t[%.3f] secs\n", t);
        minTimeSIMD = std::min(minTimeSIMD, t);
    }

    // Report results and save image
    printf("[aobench SIMD]:\t\t\t[%.3f] secs (%d x %d image)\n", minTimeSIMD, width, height);
    savePPM("ao-simd.ppm", width, height);

#if 0
    //
    // Run the ispc + tasks path, test_iterations times, and report the
    // minimum time for any of them.
    //
    double minTimeISPCTasks = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        assert(NSUBSAMPLES == 2);

        reset_and_start_timer();
        ao_ispc_tasks(width, height, NSUBSAMPLES, fimg);
        double t = get_elapsed_mcycles();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", t);
        minTimeISPCTasks = std::min(minTimeISPCTasks, t);
    }

    // Report results and save image
    printf("[aobench ispc + tasks]:\t\t[%.3f] million cycles (%d x %d image)\n", minTimeISPCTasks, width, height);
    savePPM("ao-ispc-tasks.ppm", width, height);
#endif

    //
    // Run the serial path, again test_iteration times, and report the
    // minimum time.
    //
    double minTimeSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations[2]; i++) 
	{
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        tm.start();
        ao_serial(width, height, NSUBSAMPLES, fimg);
        double t = tm.get_elapsed_secs();
        printf("@time of serial run:\t\t\t\t[%.3f] secs\n", t);
        minTimeSerial = std::min(minTimeSerial, t);
    }

    // Report more results, save another image...
    printf("[aobench serial]:\t\t[%.3f] secs (%d x %d image)\n", minTimeSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from SIMD)\n", minTimeSerial / minTimeSIMD);
    savePPM("ao-serial.ppm", width, height);

    return 0;
}


int main(int argc, char *arg_v[])
{
	test();

	test_ao();
		
	test_options(false);
	
	test_options(true);
				
	test_simple();

	test_rt();

	test_volume();

	test_noise();
				
	test_mandel();

	return EXIT_SUCCESS;
}