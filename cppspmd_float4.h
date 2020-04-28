// cppspmd_float4.h
// This is *very* slow, but it's useful for debugging, verification, and porting.
// Originally written by Nicolas Guillemot, Jefferson Amstutz in the "CppSPMD" project.
// 4/20: Richard Geldreich: Macro control flow, more SIMD instruction sets, optimizations, supports using multiple SIMD instruction sets in same executable. Still a work in progress!

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
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <utility>
#include <algorithm>

// Set to 1 to enable small but key x86 SSE helpers that work around run-time library slowness.
#if defined(_M_X64) || defined(_M_IX86)
	#define CPPSPMD_USE_SSE_INTRINSIC_HELPERS 1
#endif

// Set to 1 to use std::fmaf()
#define CPPSPMD_USE_FMAF 0

#if CPPSPMD_USE_SSE_INTRINSIC_HELPERS
	#include <immintrin.h>
#endif

#undef CPPSPMD_SSE
#undef CPPSPMD_AVX1
#undef CPPSPMD_AVX2
#undef CPPSPMD_AVX
#undef CPPSPMD_FLOAT4

#define CPPSPMD_SSE 0
#define CPPSPMD_AVX 0
#define CPPSPMD_AVX1 0
#define CPPSPMD_AVX2 0
#define CPPSPMD_FLOAT4 1

#ifdef _MSC_VER
	#ifndef CPPSPMD_DECL
		#define CPPSPMD_DECL(type, name) __declspec(align(32)) type name
	#endif

	#ifndef CPPSPMD_ALIGN
		#define CPPSPMD_ALIGN(v) __declspec(align(v))
	#endif
#else
	#ifndef CPPSPMD_DECL
		#define CPPSPMD_DECL(type, name) type name __attribute__((aligned(32)))
	#endif

	#ifndef CPPSPMD_ALIGN
		#define CPPSPMD_ALIGN(v) __attribute__((aligned(v)))
	#endif
#endif

#ifndef CPPSPMD_FORCE_INLINE
	#define CPPSPMD_FORCE_INLINE __forceinline
#endif

#undef CPPSPMD
#undef CPPSPMD_ARCH

#define CPPSPMD cppspmd_float4
#define CPPSPMD_ARCH _float4

#ifndef CPPSPMD_GLUER
	#define CPPSPMD_GLUER(a, b) a##b
#endif

#ifndef CPPSPMD_GLUER2
	#define CPPSPMD_GLUER2(a, b) CPPSPMD_GLUER(a, b)
#endif

#ifndef CPPSPMD_MAKE_NAME
	#define CPPSPMD_MAKE_NAME(a) CPPSPMD_GLUER2(a, CPPSPMD_ARCH)
#endif

namespace CPPSPMD
{

const int PROGRAM_COUNT_SHIFT = 2;
const int PROGRAM_COUNT = 1 << PROGRAM_COUNT_SHIFT;

CPPSPMD_DECL(uint32_t, g_allones_128[4]) = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
CPPSPMD_DECL(float, g_onef_128[4]) = { 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(uint32_t, g_oneu_128[4]) = { 1, 1, 1, 1 };

struct int4;

CPPSPMD_ALIGN(16)
struct float4
{
	float c[4];

	CPPSPMD_FORCE_INLINE float4() = default;
	CPPSPMD_FORCE_INLINE float4(const float4 &a) { c[0] = a.c[0]; c[1] = a.c[1]; c[2] = a.c[2]; c[3] = a.c[3]; }
	CPPSPMD_FORCE_INLINE float4(float x, float y, float z, float w) { c[0] = x; c[1] = y; c[2] = z; c[3] = w; }

	CPPSPMD_FORCE_INLINE float4& operator=(const float4 &a) { c[0] = a.c[0]; c[1] = a.c[1]; c[2] = a.c[2]; c[3] = a.c[3]; return *this; }
};

CPPSPMD_ALIGN(16)
struct int4
{
	int32_t c[4];

	CPPSPMD_FORCE_INLINE int4() = default;
	CPPSPMD_FORCE_INLINE int4(const int4 &a) { c[0] = a.c[0]; c[1] = a.c[1]; c[2] = a.c[2]; c[3] = a.c[3]; }
	CPPSPMD_FORCE_INLINE int4(int x, int y, int z, int w) { c[0] = x; c[1] = y; c[2] = z; c[3] = w; }

	CPPSPMD_FORCE_INLINE int4& operator=(const int4 &a) { c[0] = a.c[0]; c[1] = a.c[1]; c[2] = a.c[2]; c[3] = a.c[3]; return *this; }
};

#define CAST_F_TO_I(f) *(int *)(&(f))
#define CAST_I_TO_F(i) *(float *)(&(i))

CPPSPMD_FORCE_INLINE int4 cast_float4_to_int4(const float4 &a) { return int4(CAST_F_TO_I(a.c[0]), CAST_F_TO_I(a.c[1]), CAST_F_TO_I(a.c[2]), CAST_F_TO_I(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 cast_int4_to_float4(const int4 &a) { return float4(CAST_I_TO_F(a.c[0]), CAST_I_TO_F(a.c[1]), CAST_I_TO_F(a.c[2]), CAST_I_TO_F(a.c[3])); }

CPPSPMD_FORCE_INLINE float4 create_float4_rev(float w, float z, float y, float x) { return float4(x, y, z, w); }
CPPSPMD_FORCE_INLINE float4 create_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
CPPSPMD_FORCE_INLINE float4 set1_float4(float x) { return float4(x, x, x, x); }

CPPSPMD_FORCE_INLINE float minf(float a, float b) { return a < b ? a : b; }
CPPSPMD_FORCE_INLINE float maxf(float a, float b) { return a > b ? a : b; }

CPPSPMD_FORCE_INLINE int mini(int a, int b) { return a < b ? a : b; }
CPPSPMD_FORCE_INLINE int maxi(int a, int b) { return a > b ? a : b; }

#if CPPSPMD_USE_FMAF
CPPSPMD_FORCE_INLINE float my_fmaf(float a, float b, float c) { return std::fmaf(a, b, c); }
#else
CPPSPMD_FORCE_INLINE float my_fmaf(float a, float b, float c) { return (a * b) + c; }
#endif

// Work around MSVC run-time library slowness.
// std::roundf(): halfway casea re rounded away from zero, while with _mm_round_ss() it rounds halfway cases towards zero.
#if CPPSPMD_USE_SSE_INTRINSIC_HELPERS
CPPSPMD_FORCE_INLINE float my_roundf(float a) { __m128 k = _mm_load1_ps(&a); int r = _mm_extract_ps( _mm_round_ss(k, k,  _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC), 0 ); return *(const float *)&r; }
CPPSPMD_FORCE_INLINE float my_truncf(float a) { __m128 k = _mm_load1_ps(&a); int r = _mm_extract_ps( _mm_round_ss(k, k, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC), 0 ); return *(const float *)&r; }
#else
CPPSPMD_FORCE_INLINE float my_roundf(float a) { return std::roundf(a); }
CPPSPMD_FORCE_INLINE float my_truncf(float a) { return std::truncf(a); }
#endif

CPPSPMD_FORCE_INLINE float4 setzero_float4() { return float4(0.0f, 0.0f, 0.0f, 0.0f); }
CPPSPMD_FORCE_INLINE float4 add_float4(const float4 &a, const float4 &b) { return float4(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2], a.c[3] + b.c[3]); }
CPPSPMD_FORCE_INLINE float4 sub_float4(const float4 &a, const float4 &b) { return float4(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2], a.c[3] - b.c[3]); }
CPPSPMD_FORCE_INLINE float4 mul_float4(const float4 &a, const float4 &b) { return float4(a.c[0] * b.c[0], a.c[1] * b.c[1], a.c[2] * b.c[2], a.c[3] * b.c[3]); }
CPPSPMD_FORCE_INLINE float4 div_float4(const float4 &a, const float4 &b) { return float4(a.c[0] / b.c[0], a.c[1] / b.c[1], a.c[2] / b.c[2], a.c[3] / b.c[3]); }
CPPSPMD_FORCE_INLINE float4 fma_float4(const float4 &a, const float4 &b, const float4 &c) { return float4(my_fmaf(a.c[0], b.c[0], c.c[0]), my_fmaf(a.c[1], b.c[1], c.c[1]), my_fmaf(a.c[2], b.c[2], c.c[2]), my_fmaf(a.c[3], b.c[3], c.c[3])); }
CPPSPMD_FORCE_INLINE float4 fms_float4(const float4 &a, const float4 &b, const float4 &c) { return float4(my_fmaf(a.c[0], b.c[0], -c.c[0]), my_fmaf(a.c[1], b.c[1], -c.c[1]), my_fmaf(a.c[2], b.c[2], -c.c[2]), my_fmaf(a.c[3], b.c[3], -c.c[3])); }
CPPSPMD_FORCE_INLINE float4 fnma_float4(const float4 &a, const float4 &b, const float4 &c) { return float4(c.c[0] - (a.c[0] * b.c[0]), c.c[1] - (a.c[1] * b.c[1]), c.c[2] - (a.c[2] * b.c[2]), c.c[3] - (a.c[3] * b.c[3])); }
CPPSPMD_FORCE_INLINE float4 fnms_float4(const float4 &a, const float4 &b, const float4 &c) { return float4(-c.c[0] - (a.c[0] * b.c[0]), -c.c[1] - (a.c[1] * b.c[1]), -c.c[2] - (a.c[2] * b.c[2]), -c.c[3] - (a.c[3] * b.c[3])); }
// Emulate what we do in SSE, so -0 does the same thing
CPPSPMD_FORCE_INLINE float4 neg_float4(const float4 &a) { return float4(0.0f - a.c[0], 0.0f - a.c[1], 0.0f - a.c[2], 0.0f - a.c[3]); }
CPPSPMD_FORCE_INLINE int4 convert_float4_to_int4(const float4 &a) { return int4((int)a.c[0], (int)a.c[1], (int)a.c[2], (int)a.c[3]); }
CPPSPMD_FORCE_INLINE float4 convert_int4_to_float4(const int4 &a) { return float4((float)a.c[0], (float)a.c[1], (float)a.c[2], (float)a.c[3]); }
CPPSPMD_FORCE_INLINE float4 floor_float4(const float4 &a) { return float4(floorf(a.c[0]), floorf(a.c[1]), floorf(a.c[2]), floorf(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 ceil_float4(const float4 &a) { return float4(ceilf(a.c[0]), ceilf(a.c[1]), ceilf(a.c[2]), ceilf(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 min_float4(const float4 &a, const float4 &b) { return float4(minf(a.c[0], b.c[0]), minf(a.c[1], b.c[1]), minf(a.c[2], b.c[2]), minf(a.c[3], b.c[3])); }
CPPSPMD_FORCE_INLINE float4 max_float4(const float4 &a, const float4 &b) { return float4(maxf(a.c[0], b.c[0]), maxf(a.c[1], b.c[1]), maxf(a.c[2], b.c[2]), maxf(a.c[3], b.c[3])); }
CPPSPMD_FORCE_INLINE float4 sqrt_float4(const float4 &a) { return float4(sqrtf(a.c[0]), sqrtf(a.c[1]), sqrtf(a.c[2]), sqrtf(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 abs_float4(const float4 &a) { return float4(fabsf(a.c[0]), fabsf(a.c[1]), fabsf(a.c[2]), fabsf(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 round_float4(const float4 &a) { return float4(my_roundf(a.c[0]), my_roundf(a.c[1]), my_roundf(a.c[2]), my_roundf(a.c[3])); }
CPPSPMD_FORCE_INLINE float4 truncate_float4(const float4 &a) { return float4(my_truncf(a.c[0]), my_truncf(a.c[1]), my_truncf(a.c[2]), my_truncf(a.c[3])); }

CPPSPMD_FORCE_INLINE int4 cmp_eq_float4(const float4 &a, const float4 &b) { return int4(-(a.c[0] == b.c[0]), -(a.c[1] == b.c[1]), -(a.c[2] == b.c[2]), -(a.c[3] == b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_lt_float4(const float4 &a, const float4 &b) { return int4(-(a.c[0] < b.c[0]), -(a.c[1] < b.c[1]), -(a.c[2] < b.c[2]), -(a.c[3] < b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_le_float4(const float4 &a, const float4 &b) { return int4(-(a.c[0] <= b.c[0]), -(a.c[1] <= b.c[1]), -(a.c[2] <= b.c[2]), -(a.c[3] <= b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_gt_float4(const float4 &a, const float4 &b) { return int4(-(a.c[0] > b.c[0]), -(a.c[1] > b.c[1]), -(a.c[2] > b.c[2]), -(a.c[3] > b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_ge_float4(const float4 &a, const float4 &b) { return int4(-(a.c[0] >= b.c[0]), -(a.c[1] >= b.c[1]), -(a.c[2] >= b.c[2]), -(a.c[3] >= b.c[3])); }
	
CPPSPMD_FORCE_INLINE int4 create_int4_rev(int w, int z, int y, int x) { return int4(x, y, z, w); }
CPPSPMD_FORCE_INLINE int4 create_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }
CPPSPMD_FORCE_INLINE int4 set1_int4(int x) { return int4(x, x, x, x); }
CPPSPMD_FORCE_INLINE int4 setzero_int4() { return int4(0, 0, 0, 0); }

CPPSPMD_FORCE_INLINE int4 add_int4(const int4 &a, const int4 &b) { return int4(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2], a.c[3] + b.c[3]); }
CPPSPMD_FORCE_INLINE int4 sub_int4(const int4 &a, const int4 &b) { return int4(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2], a.c[3] - b.c[3]); }
CPPSPMD_FORCE_INLINE int4 mul_int4(const int4 &a, const int4 &b) { return int4(a.c[0] * b.c[0], a.c[1] * b.c[1], a.c[2] * b.c[2], a.c[3] * b.c[3]); }
CPPSPMD_FORCE_INLINE int4 div_int4(const int4 &a, const int4 &b) { return int4(b.c[0] ? (a.c[0] / b.c[0]) : 0, b.c[1] ? (a.c[1] / b.c[1]) : 0, b.c[2] ? (a.c[2] / b.c[2]) : 0, b.c[3] ? (a.c[3] / b.c[3]) : 0); }
CPPSPMD_FORCE_INLINE int4 div_int4(const int4 &a, int b) { return b ? int4(a.c[0] / b, a.c[1] / b, a.c[2] / b, a.c[3] / b) : int4(0, 0, 0, 0); }
CPPSPMD_FORCE_INLINE int4 mod_int4(const int4 &a, const int4 &b) { return int4(b.c[0] ? (a.c[0] % b.c[0]) : 0, b.c[1] ? (a.c[1] % b.c[1]) : 0, b.c[2] ? (a.c[2] % b.c[2]) : 0, b.c[3] ? (a.c[3] % b.c[3]) : 0); }
CPPSPMD_FORCE_INLINE int4 mod_int4(const int4 &a, int b) { return b ? int4(a.c[0] % b, a.c[1] % b, a.c[2] % b, a.c[3] % b) : int4(0, 0, 0, 0); }
CPPSPMD_FORCE_INLINE int4 neg_int4(const int4 &a) { return int4(-a.c[0], -a.c[1], -a.c[2], -a.c[3]); }		
	
CPPSPMD_FORCE_INLINE int4 and_int4(const int4 &a, const int4 &b) { return int4(a.c[0] & b.c[0], a.c[1] & b.c[1], a.c[2] & b.c[2], a.c[3] & b.c[3]); }
CPPSPMD_FORCE_INLINE int4 or_int4(const int4 &a, const int4 &b) { return int4(a.c[0] | b.c[0], a.c[1] | b.c[1], a.c[2] | b.c[2], a.c[3] | b.c[3]); }
CPPSPMD_FORCE_INLINE int4 xor_int4(const int4 &a, const int4 &b) { return int4(a.c[0] ^ b.c[0], a.c[1] ^ b.c[1], a.c[2] ^ b.c[2], a.c[3] ^ b.c[3]); }
CPPSPMD_FORCE_INLINE int4 andnot_int4(const int4 &a, const int4 &b) { return int4((~a.c[0]) & b.c[0], (~a.c[1]) & b.c[1], (~a.c[2]) & b.c[2], (~a.c[3]) & b.c[3]); }
	
CPPSPMD_FORCE_INLINE int4 shift_left_int4(const int4 &a, const int4 &b) { return int4(a.c[0] << b.c[0], a.c[1] << b.c[1], a.c[2] << b.c[2], a.c[3] << b.c[3]); }
CPPSPMD_FORCE_INLINE int4 shift_right_int4(const int4 &a, const int4 &b) { return int4(a.c[0] >> b.c[0], a.c[1] >> b.c[1], a.c[2] >> b.c[2], a.c[3] >> b.c[3]); }
CPPSPMD_FORCE_INLINE int4 unsigned_shift_right_int4(const int4 &a, const int4 &b) { return int4(((uint32_t)a.c[0]) >> b.c[0], ((uint32_t)a.c[1]) >> b.c[1], ((uint32_t)a.c[2]) >> b.c[2], ((uint32_t)a.c[3]) >> b.c[3]); }

CPPSPMD_FORCE_INLINE int4 shift_left_int4(const int4 &a, int b) { return int4(a.c[0] << b, a.c[1] << b, a.c[2] << b, a.c[3] << b); }
CPPSPMD_FORCE_INLINE int4 shift_right_int4(const int4 &a, int b) { return int4(a.c[0] >> b, a.c[1] >> b, a.c[2] >> b, a.c[3] >> b); }
CPPSPMD_FORCE_INLINE int4 unsigned_shift_right_int4(const int4 &a, int b) { return int4(((uint32_t)a.c[0]) >> b, ((uint32_t)a.c[1]) >> b, ((uint32_t)a.c[2]) >> b, ((uint32_t)a.c[3]) >> b); }

CPPSPMD_FORCE_INLINE int4 cmp_eq_int4(const int4 &a, const int4 &b) { return int4(-(a.c[0] == b.c[0]), -(a.c[1] == b.c[1]), -(a.c[2] == b.c[2]), -(a.c[3] == b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_gt_int4(const int4 &a, const int4 &b) { return int4(-(a.c[0] > b.c[0]), -(a.c[1] > b.c[1]), -(a.c[2] > b.c[2]), -(a.c[3] > b.c[3])); }

CPPSPMD_FORCE_INLINE int get_movemask_int4(const int4 &a) {	assert((a.c[0] == 0) || (a.c[0] == UINT32_MAX)); assert((a.c[1] == 0) || (a.c[1] == UINT32_MAX)); assert((a.c[2] == 0) || (a.c[2] == UINT32_MAX));	assert((a.c[3] == 0) || (a.c[3] == UINT32_MAX));	return (a.c[0] & 1) | (a.c[1] & 2) | (a.c[2] & 4) | (a.c[3] & 8); }

CPPSPMD_FORCE_INLINE int4 min_int4(const int4 &a, const int4 &b) { return int4(mini(a.c[0], b.c[0]), mini(a.c[1], b.c[1]), mini(a.c[2], b.c[2]), mini(a.c[3], b.c[3])); }
CPPSPMD_FORCE_INLINE int4 max_int4(const int4 &a, const int4 &b) { return int4(maxi(a.c[0], b.c[0]), maxi(a.c[1], b.c[1]), maxi(a.c[2], b.c[2]), maxi(a.c[3], b.c[3])); }

CPPSPMD_FORCE_INLINE int4 blend_int4(const int4 &a, const int4 &b, const int4 &c) { return int4( (b.c[0] & c.c[0]) | (a.c[0] & (~c.c[0])), (b.c[1] & c.c[1]) | (a.c[1] & (~c.c[1])), (b.c[2] & c.c[2]) | (a.c[2] & (~c.c[2])), (b.c[3] & c.c[3]) | (a.c[3] & (~c.c[3])) ); }
CPPSPMD_FORCE_INLINE float4 blend_float4(const float4 &a, const float4 &b, const float4 &c) { return cast_int4_to_float4(blend_int4(cast_float4_to_int4(a), cast_float4_to_int4(b), cast_float4_to_int4(c))); }
CPPSPMD_FORCE_INLINE float4 blend_float4(const float4 &a, const float4 &b, const int4 &c) { return cast_int4_to_float4(blend_int4(cast_float4_to_int4(a), cast_float4_to_int4(b), c)); }

CPPSPMD_FORCE_INLINE void store_float4(float *pDst, const float4 &v, const uint32_t stride = 1) { pDst[0] = v.c[0]; pDst[stride] = v.c[1]; pDst[stride*2] = v.c[2]; pDst[stride*3] = v.c[3]; }
CPPSPMD_FORCE_INLINE float4 load_float4(const float *pSrc, const uint32_t stride = 1) { return float4(pSrc[0], pSrc[stride], pSrc[stride*2], pSrc[stride*3]); }

CPPSPMD_FORCE_INLINE void store_int4(int *pDst, const int4 &v, const uint32_t stride = 1) { pDst[0] = v.c[0]; pDst[stride] = v.c[1]; pDst[stride*2] = v.c[2]; pDst[stride*3] = v.c[3]; }
CPPSPMD_FORCE_INLINE int4 load_int4(const int *pSrc, const uint32_t stride = 1) { return int4(pSrc[0], pSrc[stride], pSrc[stride*2], pSrc[stride*3]); }

CPPSPMD_FORCE_INLINE void store_mask_float4(float *pDst, const float4 &v, const int4 &m, const uint32_t stride = 1) { if (m.c[0]) pDst[0] = v.c[0]; if (m.c[1]) pDst[stride] = v.c[1]; if (m.c[2]) pDst[stride*2] = v.c[2]; if (m.c[3]) pDst[stride*3] = v.c[3]; }
CPPSPMD_FORCE_INLINE float4 load_mask_float4(const float *pSrc, const int4 &m, const uint32_t stride = 1) {	float4 result(set1_float4(0.0f)); if (m.c[0]) result.c[0] = pSrc[0]; if (m.c[1]) result.c[1] = pSrc[stride]; if (m.c[2]) result.c[2] = pSrc[stride*2]; if (m.c[3]) result.c[3] = pSrc[stride*3]; return result; }

CPPSPMD_FORCE_INLINE void store_mask_int4(int *pDst, const int4 &v, const int4 &m, const uint32_t stride = 1) { if (m.c[0]) pDst[0] = v.c[0]; if (m.c[1]) pDst[stride] = v.c[1]; if (m.c[2]) pDst[stride*2] = v.c[2]; if (m.c[3]) pDst[stride*3] = v.c[3]; }
CPPSPMD_FORCE_INLINE int4 load_mask_int4(const int *pSrc, const int4 &m, const uint32_t stride = 1) { int4 result(set1_int4(0)); if (m.c[0]) result.c[0] = pSrc[0]; if (m.c[1]) result.c[1] = pSrc[stride]; if (m.c[2]) result.c[2] = pSrc[stride*2]; if (m.c[3]) result.c[3] = pSrc[stride*3]; return result; }

CPPSPMD_FORCE_INLINE void scatter_mask_float4(float *pDst, const float4 &v, const int4 &ofs, const int4 &m) { if (m.c[0]) pDst[ofs.c[0]] = v.c[0]; if (m.c[1]) pDst[ofs.c[1]] = v.c[1]; if (m.c[2]) pDst[ofs.c[2]] = v.c[2]; if (m.c[3]) pDst[ofs.c[3]] = v.c[3]; }
CPPSPMD_FORCE_INLINE float4 gather_mask_float4(const float *pSrc, const int4 &ofs, const int4 &m) { float4 result(set1_float4(0.0f)); if (m.c[0]) result.c[0] = pSrc[ofs.c[0]]; if (m.c[1]) result.c[1] = pSrc[ofs.c[1]]; if (m.c[2]) result.c[2] = pSrc[ofs.c[2]]; if (m.c[3]) result.c[3] = pSrc[ofs.c[3]];	return result; }

CPPSPMD_FORCE_INLINE void scatter_mask_int4(int *pDst, const int4 &v, const int4 &ofs, const int4 &m) { if (m.c[0]) pDst[ofs.c[0]] = v.c[0]; if (m.c[1]) pDst[ofs.c[1]] = v.c[1]; if (m.c[2]) pDst[ofs.c[2]] = v.c[2]; if (m.c[3]) pDst[ofs.c[3]] = v.c[3]; }
CPPSPMD_FORCE_INLINE int4 gather_mask_int4(const int *pSrc, const int4 &ofs, const int4 &m) { int4 result(set1_int4(0)); if (m.c[0]) result.c[0] = pSrc[ofs.c[0]]; if (m.c[1]) result.c[1] = pSrc[ofs.c[1]]; if (m.c[2]) result.c[2] = pSrc[ofs.c[2]]; if (m.c[3]) result.c[3] = pSrc[ofs.c[3]]; return result; }

CPPSPMD_FORCE_INLINE void scatter_float4(float *pDst, const float4 &v, const int4 &ofs) { pDst[ofs.c[0]] = v.c[0]; pDst[ofs.c[1]] = v.c[1]; pDst[ofs.c[2]] = v.c[2]; pDst[ofs.c[3]] = v.c[3]; }
CPPSPMD_FORCE_INLINE float4 gather_float4(const float *pSrc, const int4 &ofs) { float4 result(set1_float4(0.0f)); result.c[0] = pSrc[ofs.c[0]]; result.c[1] = pSrc[ofs.c[1]]; result.c[2] = pSrc[ofs.c[2]]; result.c[3] = pSrc[ofs.c[3]];	return result; }

CPPSPMD_FORCE_INLINE void scatter_int4(int *pDst, const int4 &v, const int4 &ofs) { pDst[ofs.c[0]] = v.c[0]; pDst[ofs.c[1]] = v.c[1]; pDst[ofs.c[2]] = v.c[2]; pDst[ofs.c[3]] = v.c[3]; }
CPPSPMD_FORCE_INLINE int4 gather_int4(const int *pSrc, const int4 &ofs) { int4 result(set1_int4(0)); result.c[0] = pSrc[ofs.c[0]]; result.c[1] = pSrc[ofs.c[1]]; result.c[2] = pSrc[ofs.c[2]]; result.c[3] = pSrc[ofs.c[3]]; return result; }

const uint32_t ALL_ON_MOVEMASK = 0xF;

struct spmd_kernel
{
	struct vint;
	struct vbool;
	struct vfloat;
		
	// Exec mask
	struct exec_mask
	{
		int4 m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE exec_mask(const exec_mask& b) { m_mask.c[0] = b.m_mask.c[0]; m_mask.c[1] = b.m_mask.c[1]; m_mask.c[2] = b.m_mask.c[2]; m_mask.c[3] = b.m_mask.c[3]; }

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const int4& mask) : m_mask(mask) { }
				
		static CPPSPMD_FORCE_INLINE exec_mask all_on()	{ return exec_mask{ set1_int4(UINT32_MAX) }; }
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ set1_int4(0) }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return get_movemask_int4(m_mask); }
	};

	friend CPPSPMD_FORCE_INLINE bool all(const exec_mask& e);
	friend CPPSPMD_FORCE_INLINE bool any(const exec_mask& e);

	friend CPPSPMD_FORCE_INLINE exec_mask operator^ (const exec_mask& a, const exec_mask& b);
	friend CPPSPMD_FORCE_INLINE exec_mask operator& (const exec_mask& a, const exec_mask& b);
	friend CPPSPMD_FORCE_INLINE exec_mask operator| (const exec_mask& a, const exec_mask& b);
		
	exec_mask m_exec;
	exec_mask m_internal_exec;
	exec_mask m_kernel_exec;
	exec_mask m_continue_mask;
		
	void init(const exec_mask& kernel_exec);
	
	// Varying bool
		
	struct vbool
	{
		int4 m_value;

		CPPSPMD_FORCE_INLINE vbool(bool value) : m_value(set1_int4(value ? UINT32_MAX : 0)) { }

		CPPSPMD_FORCE_INLINE vbool(const vbool& b) { m_value.c[0] = b.m_value.c[0]; m_value.c[1] = b.m_value.c[1]; m_value.c[2] = b.m_value.c[2]; m_value.c[3] = b.m_value.c[3]; }

		CPPSPMD_FORCE_INLINE explicit vbool(const int4& value) : m_value(value) { assert((m_value.c[0] == 0) || (m_value.c[0] == UINT32_MAX)); assert((m_value.c[1] == 0) || (m_value.c[1] == UINT32_MAX)); assert((m_value.c[2] == 0) || (m_value.c[2] == UINT32_MAX)); assert((m_value.c[3] == 0) || (m_value.c[3] == UINT32_MAX)); }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const;
		CPPSPMD_FORCE_INLINE explicit operator vint() const;
								
	private:
		vbool& operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = blend_int4(dst.m_value, src.m_value, m_exec.m_mask);
		return dst;
	}
		
	CPPSPMD_FORCE_INLINE vbool& store_all(vbool& dst, const vbool& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
	
	// Varying float
	struct vfloat
	{
		float4 m_value;

		vfloat() = default;

		CPPSPMD_FORCE_INLINE vfloat(const vfloat& b) { m_value.c[0] = b.m_value.c[0]; m_value.c[1] = b.m_value.c[1]; m_value.c[2] = b.m_value.c[2]; m_value.c[3] = b.m_value.c[3]; }

		CPPSPMD_FORCE_INLINE explicit vfloat(const float4& v) : m_value(v) { }

		CPPSPMD_FORCE_INLINE vfloat(float value) : m_value(set1_float4(value)) { }

		CPPSPMD_FORCE_INLINE explicit vfloat(int value) : m_value(set1_float4((float)value)) { }

	private:
		vfloat& operator=(const vfloat&);
	};

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat& dst, const vfloat& src)
	{
		dst.m_value = blend_float4(dst.m_value, src.m_value, m_exec.m_mask);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat&& dst, const vfloat& src)
	{
		dst.m_value = blend_float4(dst.m_value, src.m_value, m_exec.m_mask);
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vfloat& store_all(vfloat& dst, const vfloat& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store_all(vfloat&& dst, const vfloat& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}

	// Linear ref to floats
	struct float_lref
	{
		float* m_pValue;

	private:
		float_lref& operator=(const float_lref&);
	};

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref& dst, const vfloat& src)
	{
		store_mask_float4(dst.m_pValue, src.m_value, m_exec.m_mask);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref&& dst, const vfloat& src)
	{
		store_mask_float4(dst.m_pValue, src.m_value, m_exec.m_mask);
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref& dst, const vfloat& src)
	{
		store_float4(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref&& dst, const vfloat& src)
	{
		store_float4(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const float_lref& src)
	{
		return vfloat{ load_mask_float4(src.m_pValue, m_exec.m_mask) };
	}
		
	// Varying ref to floats
	struct float_vref
	{
		int4 m_vindex;
		float* m_pValue;
		
	private:
		float_vref& operator=(const float_vref&);
	};

	// Varying ref to varying float
	struct vfloat_vref
	{
		int4 m_vindex;
		vfloat* m_pValue;
		
	private:
		vfloat_vref& operator=(const vfloat_vref&);
	};

	// Varying ref to varying int
	struct vint_vref
	{
		int4 m_vindex;
		vint* m_pValue;
		
	private:
		vint_vref& operator=(const vint_vref&);
	};

	CPPSPMD_FORCE_INLINE const float_vref& store(const float_vref& dst, const vfloat& src);
	CPPSPMD_FORCE_INLINE const float_vref& store(const float_vref&& dst, const vfloat& src);
		
	CPPSPMD_FORCE_INLINE const float_vref& store_all(const float_vref& dst, const vfloat& src);
	CPPSPMD_FORCE_INLINE const float_vref& store_all(const float_vref&& dst, const vfloat& src);

	CPPSPMD_FORCE_INLINE vfloat load(const float_vref& src)
	{
		return vfloat{ gather_mask_float4(src.m_pValue, src.m_vindex, m_exec.m_mask) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
		return vfloat{ gather_float4(src.m_pValue, src.m_vindex) };
	}

	// Linear ref to ints
	struct int_lref
	{
		int* m_pValue;

	private:
		int_lref& operator=(const int_lref&);
	};
		
	CPPSPMD_FORCE_INLINE const int_lref& store(const int_lref& dst, const vint& src)
	{
		store_mask_int4(dst.m_pValue, src.m_value, m_exec.m_mask);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_lref& src)
	{
		return vint{ load_mask_int4(src.m_pValue, m_exec.m_mask) };
	}

	// Linear ref to a varying int16
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};

	CPPSPMD_FORCE_INLINE const int16_lref& store(const int16_lref& dst, const vint& src)
	{
		if (m_exec.m_mask.c[0]) dst.m_pValue[0] = (int16_t)src.m_value.c[0];
		if (m_exec.m_mask.c[1]) dst.m_pValue[1] = (int16_t)src.m_value.c[1];
		if (m_exec.m_mask.c[2]) dst.m_pValue[2] = (int16_t)src.m_value.c[2];
		if (m_exec.m_mask.c[3]) dst.m_pValue[3] = (int16_t)src.m_value.c[3];
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_lref& store_all(const int16_lref& dst, const vint& src)
	{
		dst.m_pValue[0] = (int16_t)src.m_value.c[0];
		dst.m_pValue[1] = (int16_t)src.m_value.c[1];
		dst.m_pValue[2] = (int16_t)src.m_value.c[2];
		dst.m_pValue[3] = (int16_t)src.m_value.c[3];
		return dst;
	}
		
	CPPSPMD_FORCE_INLINE vint load(const int16_lref& src)
	{
		int4 result(set1_int4(0));
		if (m_exec.m_mask.c[0]) result.c[0] = src.m_pValue[0];
		if (m_exec.m_mask.c[1]) result.c[1] = src.m_pValue[1];
		if (m_exec.m_mask.c[2]) result.c[2] = src.m_pValue[2];
		if (m_exec.m_mask.c[3]) result.c[3] = src.m_pValue[3];
		return vint{ result };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const int16_lref& src)
	{
		return vint{ int4( src.m_pValue[0], src.m_pValue[1], src.m_pValue[2], src.m_pValue[3]) };
	}
		
	// Constant linear ref to a varying int
	struct cint_lref
	{
		const int* m_pValue;

	private:
		cint_lref& operator=(const cint_lref&);
	};

	CPPSPMD_FORCE_INLINE vint load(const cint_lref& src)
	{
		return vint{ load_mask_int4(src.m_pValue, m_exec.m_mask) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_lref& src)
	{
		return vint{ load_int4(src.m_pValue) };
	}
	
	// Varying ref to ints
	struct int_vref
	{
		int4 m_vindex;
		int* m_pValue;

	private:
		int_vref& operator=(const int_vref&);
	};

	// Varying ref to constant ints
	struct cint_vref
	{
		int4 m_vindex;
		const int* m_pValue;

	private:
		cint_vref& operator=(const cint_vref&);
	};

	// Varying int
	struct vint
	{
		int4 m_value;

		vint() = default;

		CPPSPMD_FORCE_INLINE vint(const vint& b) { m_value.c[0] = b.m_value.c[0]; m_value.c[1] = b.m_value.c[1]; m_value.c[2] = b.m_value.c[2]; m_value.c[3] = b.m_value.c[3]; }

		CPPSPMD_FORCE_INLINE explicit vint(const int4& value) : m_value(value)	{ }

		CPPSPMD_FORCE_INLINE vint(int value) : m_value(set1_int4(value)) { }

		CPPSPMD_FORCE_INLINE explicit vint(float value) : m_value(set1_int4((int)value))	{ }

		CPPSPMD_FORCE_INLINE explicit vint(const vfloat& other) : m_value(convert_float4_to_int4(other.m_value)) { }

		CPPSPMD_FORCE_INLINE explicit operator vbool() const 
		{
			return vbool{ int4(-(m_value.c[0] != 0), -(m_value.c[1] != 0), -(m_value.c[2] != 0), -(m_value.c[3] != 0)) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ convert_int4_to_float4(m_value) };
		}

		CPPSPMD_FORCE_INLINE int_vref operator[](int* ptr) const
		{
			return int_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE cint_vref operator[](const int* ptr) const
		{
			return cint_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE float_vref operator[](float* ptr) const
		{
			return float_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE vfloat_vref operator[](vfloat* ptr) const
		{
			return vfloat_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE vint_vref operator[](vint* ptr) const
		{
			return vint_vref{ m_value, ptr };
		}

	private:
		vint& operator=(const vint&);
	};

	// Load/store linear integer
	CPPSPMD_FORCE_INLINE void storeu_linear(int *pDst, const vint& src)
	{
		store_mask_int4(pDst, src.m_value, m_exec.m_mask);
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int *pDst, const vint& src)
	{
		store_int4(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int *pDst, const vint& src)
	{
		store_int4(pDst, src.m_value);
	}
		
	CPPSPMD_FORCE_INLINE vint loadu_linear(const int *pSrc)
	{
		return vint{ load_mask_int4(pSrc, m_exec.m_mask) };
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear_all(const int *pSrc)
	{
		return vint{ load_int4(pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint load_linear_all(const int *pSrc)
	{
		return vint{ load_int4(pSrc) };
	}
		
	// Load/store linear float
	CPPSPMD_FORCE_INLINE void storeu_linear(float *pDst, const vfloat& src)
	{
		store_mask_float4(pDst, src.m_value, m_exec.m_mask);
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(float *pDst, const vfloat& src)
	{
		store_float4(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(float *pDst, const vfloat& src)
	{
		store_float4(pDst, src.m_value);
	}
		
	CPPSPMD_FORCE_INLINE vfloat loadu_linear(const float *pSrc)
	{
		return vfloat{ load_mask_float4(pSrc, m_exec.m_mask) };
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear_all(const float *pSrc)
	{
		return vfloat{ load_float4(pSrc) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_linear_all(const float *pSrc)
	{
		return vfloat{ load_float4(pSrc) };
	}
		
	CPPSPMD_FORCE_INLINE vint& store(vint& dst, const vint& src)
	{
		store_mask_int4(dst.m_value.c, src.m_value, m_exec.m_mask);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store(const int_vref& dst, const vint& src)
	{
		scatter_mask_int4(dst.m_pValue, src.m_value, dst.m_vindex, m_exec.m_mask);
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vint& store_all(vint& dst, const vint& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
				
	CPPSPMD_FORCE_INLINE const int_vref& store_all(const int_vref& dst, const vint& src)
	{
		scatter_int4(dst.m_pValue, src.m_value, dst.m_vindex);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_vref& src)
	{
		return vint{ gather_mask_int4(src.m_pValue, src.m_vindex, m_exec.m_mask) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const int_vref& src)
	{
		return vint{ gather_int4(src.m_pValue, src.m_vindex) };
	}
		
	CPPSPMD_FORCE_INLINE vint load(const cint_vref& src)
	{
		return vint{ gather_mask_int4(src.m_pValue, src.m_vindex, m_exec.m_mask) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const cint_vref& src)
	{
		return vint{ gather_int4(src.m_pValue, src.m_vindex) };
	}

	CPPSPMD_FORCE_INLINE void store_strided(int *pDst, uint32_t stride, const vint &v)
	{
		store_mask_int4(pDst, v.m_value, m_exec.m_mask, stride);
	}

	CPPSPMD_FORCE_INLINE void store_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		store_mask_float4(pDstF, v.m_value, m_exec.m_mask, stride);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(int *pDst, uint32_t stride, const vint &v)
	{
		store_int4(pDst, v.m_value, stride);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		store_float4(pDstF, v.m_value, stride);
	}

	CPPSPMD_FORCE_INLINE vint load_strided(const int *pSrc, uint32_t stride)
	{
		return vint{ load_mask_int4(pSrc, m_exec.m_mask, stride) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_strided(const float *pSrc, uint32_t stride)
	{
		return vfloat{ load_mask_float4(pSrc, m_exec.m_mask, stride) };
	}

	CPPSPMD_FORCE_INLINE vint load_all_strided(const int *pSrc, uint32_t stride)
	{
		return vint{ load_int4(pSrc, stride) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all_strided(const float *pSrc, uint32_t stride)
	{
		return vfloat{ load_float4(pSrc, stride) };
	}

	CPPSPMD_FORCE_INLINE const vfloat_vref& store(const vfloat_vref& dst, const vfloat& src)
	{
		int mask = get_movemask_int4(m_exec.m_mask);
		
		if (mask & 1) dst.m_pValue[dst.m_vindex.c[0]].m_value.c[0] = src.m_value.c[0];
		if (mask & 2) dst.m_pValue[dst.m_vindex.c[1]].m_value.c[1] = src.m_value.c[1];
		if (mask & 4) dst.m_pValue[dst.m_vindex.c[2]].m_value.c[2] = src.m_value.c[2];
		if (mask & 8) dst.m_pValue[dst.m_vindex.c[3]].m_value.c[3] = src.m_value.c[3];

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		int mask = get_movemask_int4(m_exec.m_mask);

		float4 k = set1_float4(0.0f);

		if (mask & 1) k.c[0] = src.m_pValue[src.m_vindex.c[0]].m_value.c[0];
		if (mask & 2) k.c[1] = src.m_pValue[src.m_vindex.c[1]].m_value.c[1];
		if (mask & 4) k.c[2] = src.m_pValue[src.m_vindex.c[2]].m_value.c[2];
		if (mask & 8) k.c[3] = src.m_pValue[src.m_vindex.c[3]].m_value.c[3];

		return vfloat{ k };
	}

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		int mask = get_movemask_int4(m_exec.m_mask);
		
		if (mask & 1) dst.m_pValue[dst.m_vindex.c[0]].m_value.c[0] = src.m_value.c[0];
		if (mask & 2) dst.m_pValue[dst.m_vindex.c[1]].m_value.c[1] = src.m_value.c[1];
		if (mask & 4) dst.m_pValue[dst.m_vindex.c[2]].m_value.c[2] = src.m_value.c[2];
		if (mask & 8) dst.m_pValue[dst.m_vindex.c[3]].m_value.c[3] = src.m_value.c[3];

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const vint_vref& src)
	{
		int mask = get_movemask_int4(m_exec.m_mask);

		int4 k = set1_int4(0);

		if (mask & 1) k.c[0] = src.m_pValue[src.m_vindex.c[0]].m_value.c[0];
		if (mask & 2) k.c[1] = src.m_pValue[src.m_vindex.c[1]].m_value.c[1];
		if (mask & 4) k.c[2] = src.m_pValue[src.m_vindex.c[2]].m_value.c[2];
		if (mask & 8) k.c[3] = src.m_pValue[src.m_vindex.c[3]].m_value.c[3];

		return vint{ k };
	}
			
	// Linear integer
	struct lint
	{
		int4 m_value;

		CPPSPMD_FORCE_INLINE lint(const lint& b) { m_value.c[0] = b.m_value.c[0]; m_value.c[1] = b.m_value.c[1]; m_value.c[2] = b.m_value.c[2]; m_value.c[3] = b.m_value.c[3]; }

		CPPSPMD_FORCE_INLINE explicit lint(int4 value)
			: m_value(value)
		{ }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ convert_int4_to_float4(m_value) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vint() const
		{
			return vint{ m_value };
		}

		CPPSPMD_FORCE_INLINE int get_first_value() const 
		{
			return m_value.c[0];
		}

		CPPSPMD_FORCE_INLINE float_lref operator[](float* ptr) const
		{
			return float_lref{ ptr + get_first_value() };
		}

		CPPSPMD_FORCE_INLINE int_lref operator[](int* ptr) const
		{
			return int_lref{ ptr + get_first_value() };
		}

		CPPSPMD_FORCE_INLINE int16_lref operator[](int16_t* ptr) const
		{
			return int16_lref{ ptr + get_first_value() };
		}

		CPPSPMD_FORCE_INLINE cint_lref operator[](const int* ptr) const
		{
			return cint_lref{ ptr + get_first_value() };
		}

	private:
		lint& operator=(const lint&);
	};
	
	const lint program_index = lint{ int4(0, 1, 2, 3) };
	
	// SPMD condition helpers

	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_if(const vbool& cond, const IfBody& ifBody);

	CPPSPMD_FORCE_INLINE void spmd_if_break(const vbool& cond);

	// No breaks, continues, etc. allowed
	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_simple_if(const vbool& cond, const IfBody& ifBody);

	// No breaks, continues, etc. allowed
	template<typename IfAnyBody, typename IfAllBody>
	CPPSPMD_FORCE_INLINE void spmd_simple_if_all(const vbool& cond, const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody);

	// No breaks, continues, etc. allowed
	template<typename IfBody, typename ElseBody>
	CPPSPMD_FORCE_INLINE void spmd_simple_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody &elseBody);

	template<typename IfAnyBody, typename IfAllBody>
	CPPSPMD_FORCE_INLINE void spmd_if_all(const vbool& cond, const IfAnyBody& ifBody, const IfAllBody& ifAllBody);
		
	template<typename IfBody, typename ElseBody>
	CPPSPMD_FORCE_INLINE void spmd_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody);

	template<typename IfAnyBody, typename IfAllBody, typename ElseAnyBody, typename ElseAllBody>
	CPPSPMD_FORCE_INLINE void spmd_ifelse_all(const vbool& cond, 
		const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody, 
		const ElseAnyBody& elseAnyBody, const ElseAllBody &elseAllBody);

	template<typename IfAnyBody, typename IfAllBody, typename ElseAnyBody>
	CPPSPMD_FORCE_INLINE void spmd_ifelse_all(const vbool& cond, 
		const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody, 
		const ElseAnyBody& elseAnyBody);

	template<typename WhileCondBody, typename WhileBody>
	CPPSPMD_FORCE_INLINE void spmd_while(const WhileCondBody& whileCondBody, const WhileBody& whileBody)
	{
		exec_mask orig_internal_exec = m_internal_exec;

		exec_mask orig_continue_mask = m_continue_mask;
		m_continue_mask = exec_mask::all_off();

#ifdef _DEBUG
		const bool prev_in_loop = m_in_loop;
		m_in_loop = true;
#endif

		while(true)
		{
			exec_mask cond_exec = exec_mask(whileCondBody());
			m_internal_exec = m_internal_exec & cond_exec;
			m_exec = m_exec & cond_exec;

			if (!any(m_exec))
				break;

			whileBody();

			m_internal_exec = m_internal_exec | m_continue_mask;
			m_exec = m_internal_exec & m_kernel_exec;
			m_continue_mask = exec_mask::all_off();
		}

#ifdef _DEBUG
		m_in_loop = prev_in_loop;
#endif

		m_internal_exec = orig_internal_exec;
		m_exec = m_internal_exec & m_kernel_exec;

		m_continue_mask = orig_continue_mask;
	}

	struct scoped_while_restorer
	{
		spmd_kernel *m_pKernel;
		exec_mask m_orig_internal_exec, m_orig_continue_mask;
#ifdef _DEBUG
		bool m_prev_in_loop;
#endif
				
		CPPSPMD_FORCE_INLINE scoped_while_restorer(spmd_kernel *pKernel) : 
			m_pKernel(pKernel), 
			m_orig_internal_exec(pKernel->m_internal_exec),
			m_orig_continue_mask(pKernel->m_continue_mask)
		{
			pKernel->m_continue_mask.all_off();

#ifdef _DEBUG
			m_prev_in_loop = pKernel->m_in_loop;
			pKernel->m_in_loop = true;
#endif
		}

		CPPSPMD_FORCE_INLINE ~scoped_while_restorer() 
		{ 
#ifdef _DEBUG
			m_pKernel->m_in_loop = false;
#endif
			m_pKernel->m_internal_exec = m_orig_internal_exec;
			m_pKernel->m_exec = m_pKernel->m_kernel_exec & m_pKernel->m_internal_exec;
			m_pKernel->m_continue_mask = m_orig_continue_mask;
		}
	};

#undef SPMD_WHILE
#undef SPMD_WEND

#define SPMD_WHILE(cond) { scoped_while_restorer CPPSPMD_GLUER2(_while_restore_, __LINE__)(this); while(true) { exec_mask CPPSPMD_GLUER2(cond_exec, __LINE__) = exec_mask(vbool(cond)); m_internal_exec = m_internal_exec & CPPSPMD_GLUER2(cond_exec, __LINE__); m_exec = m_exec & CPPSPMD_GLUER2(cond_exec, __LINE__); if (!any(m_exec)) break;
#define SPMD_WEND m_internal_exec = m_internal_exec | m_continue_mask; m_exec = m_internal_exec & m_kernel_exec; m_continue_mask = exec_mask::all_off(); } }
		
	template<typename ForInitBody, typename ForCondBody, typename ForIncrBody, typename ForBody>
	CPPSPMD_FORCE_INLINE void spmd_for(const ForInitBody& forInitBody, const ForCondBody& forCondBody, const ForIncrBody& forIncrBody, const ForBody& forBody)
	{
		exec_mask orig_internal_exec = m_internal_exec;

		forInitBody();

		exec_mask orig_continue_mask = m_continue_mask;
		m_continue_mask = exec_mask::all_off();

#ifdef _DEBUG
		const bool prev_in_loop = m_in_loop;
		m_in_loop = true;
#endif

		while(true)
		{
			exec_mask cond_exec = exec_mask(forCondBody());
			m_internal_exec = m_internal_exec & cond_exec;
			m_exec = m_exec & cond_exec;

			if (!any(m_exec))
				break;

			forBody();

			m_internal_exec = m_internal_exec | m_continue_mask;
			m_exec = m_internal_exec & m_kernel_exec;
			m_continue_mask = exec_mask::all_off();
			
			forIncrBody();
		}

#ifdef _DEBUG
		m_in_loop = prev_in_loop;
#endif

		m_internal_exec = orig_internal_exec;
		m_exec = m_internal_exec & m_kernel_exec;

		m_continue_mask = orig_continue_mask;
	}

	template<typename ForeachBody>
	CPPSPMD_FORCE_INLINE void spmd_foreach(int begin, int end, const ForeachBody& foreachBody);
		
#ifdef _DEBUG
	bool m_in_loop;
#endif

	CPPSPMD_FORCE_INLINE void spmd_break()
	{
#ifdef _DEBUG
		assert(m_in_loop);
#endif

		m_internal_exec = exec_mask::all_off();
		m_exec = exec_mask::all_off();
	}

	CPPSPMD_FORCE_INLINE void spmd_continue()
	{
#ifdef _DEBUG
		assert(m_in_loop);
#endif

		m_continue_mask = m_continue_mask | m_internal_exec;
		m_internal_exec = exec_mask::all_off();
		m_exec = exec_mask::all_off();
	}

	CPPSPMD_FORCE_INLINE void spmd_return();
		
	template<typename UnmaskedBody>
	CPPSPMD_FORCE_INLINE void spmd_unmasked(const UnmaskedBody& unmaskedBody)
	{
		exec_mask orig_exec = m_exec, orig_kernel_exec = m_kernel_exec, orig_internal_exec = m_internal_exec;

		m_kernel_exec = exec_mask::all_on();
		m_internal_exec = exec_mask::all_on();
		m_exec = exec_mask::all_on();

		unmaskedBody();

		m_kernel_exec = m_kernel_exec & orig_kernel_exec;
		m_internal_exec = m_internal_exec & orig_internal_exec;
		m_exec = m_exec & orig_exec;
	}

	template<typename SPMDKernel, typename... Args>
	CPPSPMD_FORCE_INLINE decltype(auto) spmd_call(Args&&... args)
	{
		SPMDKernel kernel;
		kernel.init(m_exec);
		return kernel._call(std::forward<Args>(args)...);
	}

	CPPSPMD_FORCE_INLINE void swap(vint &a, vint &b) { vint temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vfloat &a, vfloat &b) { vfloat temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vbool &a, vbool &b) { vbool temp = a; store(a, b); store(b, temp); }
};

using exec_mask = spmd_kernel::exec_mask;
using vint = spmd_kernel::vint;
using int_lref = spmd_kernel::int_lref;
using cint_vref = spmd_kernel::cint_vref;
using cint_lref = spmd_kernel::cint_lref;
using int_vref = spmd_kernel::int_vref;
using lint = spmd_kernel::lint;
using vbool = spmd_kernel::vbool;
using vfloat = spmd_kernel::vfloat;
using float_lref = spmd_kernel::float_lref;
using float_vref = spmd_kernel::float_vref;
using vfloat_vref = spmd_kernel::vfloat_vref;
using vint_vref = spmd_kernel::vint_vref;

CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vfloat() const 
{ 
	return vfloat { float4(m_value.c[0] ? 1.0f : 0.0f, m_value.c[1] ? 1.0f : 0.0f, m_value.c[2] ? 1.0f : 0.0f, m_value.c[3] ? 1.0f : 0.0f) };
}

// Returns UINT32_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint() const 
{ 
	return vint { m_value };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
	return vbool{ int4( -(v.m_value.c[0] == 0), -(v.m_value.c[1] == 0), -(v.m_value.c[2] == 0), -(v.m_value.c[3] == 0)) };
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ xor_int4(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) {	return exec_mask{ and_int4(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ or_int4(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return (e.m_mask.c[0] & e.m_mask.c[1] & e.m_mask.c[2] & e.m_mask.c[3]) != 0; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return (e.m_mask.c[0] | e.m_mask.c[1] | e.m_mask.c[2] | e.m_mask.c[3]) != 0; }

CPPSPMD_FORCE_INLINE bool all(const vbool& e) { return (e.m_value.c[0] & e.m_value.c[1] & e.m_value.c[2] & e.m_value.c[3]) != 0; }
CPPSPMD_FORCE_INLINE bool any(const vbool& e) { return (e.m_value.c[0] | e.m_value.c[1] | e.m_value.c[2] | e.m_value.c[3]) != 0; }

CPPSPMD_FORCE_INLINE exec_mask andnot(const exec_mask& a, const exec_mask& b) { return exec_mask{ andnot_int4(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE vbool operator||(const vbool& a, const vbool& b) { return vbool{ or_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator&&(const vbool& a, const vbool& b) { return vbool{ and_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, const vfloat& b) { return vfloat{ add_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vfloat& b) {	return vfloat{ sub_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const vfloat& b) { return vfloat(a) + b; }
CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, float b) { return a + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vint& b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vint& a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, int b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(int a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, float b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(float a, const vfloat& b) { return vfloat(a) - b; }

CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, const vfloat& b) { return vfloat{ mul_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, float b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float a, const vfloat& b) { return vfloat(a) * b; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, int b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(int a, const vfloat& b) { return vfloat(a) * b; }

CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, const vfloat& b) {	return vfloat{ div_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, int b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(int a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, float b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(float a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& v) { return vfloat{ neg_float4(v.m_value) }; }

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, const vfloat& b) { return vbool{ cmp_eq_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, float b) { return a == vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, const vfloat& b) { return !vbool{ cmp_eq_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, float b) { return a != vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, const vfloat& b) { return vbool{ cmp_lt_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, float b) { return a < vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, const vfloat& b) { return vbool{ cmp_gt_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, float b) { return a > vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, const vfloat& b) { return vbool{ cmp_le_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, float b) { return a <= vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, const vfloat& b) { return vbool{ cmp_ge_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, float b) { return a >= vfloat(b); }

CPPSPMD_FORCE_INLINE vfloat spmd_ternaryf(const vbool& cond, const vfloat& a, const vfloat& b) { return vfloat{ blend_float4(b.m_value, a.m_value, cond.m_value) }; }
CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) { return vint{ blend_int4(b.m_value, a.m_value, cond.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ sqrt_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ abs_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ max_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ min_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ ceil_float4(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ floor_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ round_float4(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ truncate_float4(a.m_value) }; }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b) { return vint{ max_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b) { return vint{ min_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ min_float4(b.m_value, max_float4(v.m_value, a.m_value)) };
}

CPPSPMD_FORCE_INLINE vint clamp(const vint& v, const vint& a, const vint& b)
{
	return vint{ max_int4(a.m_value, min_int4(v.m_value, b.m_value)) };
}

CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ fma_float4(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ fms_float4(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
return vfloat{ fnma_float4(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
return vfloat{ fnms_float4(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { return lint{ add_int4(set1_int4(a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { return lint{ add_int4(a.m_value, set1_int4(b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ and_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, const vint& b) { return vint{ or_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, int b) { return a | vint(b); }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, const vint& b) { return vint{ xor_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, int b) { return a ^ vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(const vint& a, const vint& b) { return vbool{ cmp_eq_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vint& a, const vint& b) { return !vbool{ cmp_eq_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vint& a, const vint& b) { return vbool{ cmp_gt_int4(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vint& a, const vint& b) { return !vbool{ cmp_gt_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vint& a, const vint& b) { return !vbool{ cmp_gt_int4(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vint& a, const vint& b) { return vbool{ cmp_gt_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, const vint& b) { return vint{ add_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, const vint& b) { return vint{ sub_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, int b) { return a + vint(b); }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, int b) { return a - vint(b); }
CPPSPMD_FORCE_INLINE vint operator+(int a, const vint& b) { return vint(a) + b; }
CPPSPMD_FORCE_INLINE vint operator-(int a, const vint& b) { return vint(a) - b; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, const vint& b) { return vint{ mul_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, int b) { return a * vint(b); }
CPPSPMD_FORCE_INLINE vint operator*(int a, const vint& b) { return vint(a) * b; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ neg_int4(v.m_value) }; }

// Div/mod both suppress divide by 0
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, const vint& b) { return vint{ div_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, int b) { return vint{ div_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b) { return vint{ mod_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b) { return vint{ mod_int4(a.m_value, b) }; }

CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b) { return vint{ shift_left_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b) { return vint{ shift_left_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b) { return vint{ shift_right_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b) { return vint{ unsigned_shift_right_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b) { return vint{ shift_right_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b) { return vint{ unsigned_shift_right_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint create_vint(const int4 &v) { return vint{ v }; }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) CPPSPMD::create_vint( shift_left_int4((a).m_value, (b)) )
#define VINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( shift_right_int4((a).m_value, (b)) )
#define VUINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( unsigned_shift_right_int4((a).m_value, (b)) )

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, const lint& b) { return vbool{ cmp_eq_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, int b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint& b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint& a, const lint& b) { return vbool{ cmp_gt_int4(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const lint& a, const lint& b) { return vbool{ cmp_gt_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const lint& a, const lint& b) { return !vbool{ cmp_gt_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const lint& a, const lint& b) { return !vbool{ cmp_gt_int4(b.m_value, a.m_value) }; }

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) { assert(instance < 4); return v.m_value.c[instance]; }
CPPSPMD_FORCE_INLINE int extract(const vint& v, int instance) { assert(instance < 4); return v.m_value.c[instance]; }
CPPSPMD_FORCE_INLINE int extract(const lint& v, int instance) { assert(instance < 4); return v.m_value.c[instance]; }
CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) { assert(instance < 4); return v.m_value.c[instance]; }

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

#define VINT_EXTRACT(v, instance) ((v).m_value.c[instance])
#define VBOOL_EXTRACT(v, instance) ((v).m_value.c[instance])
#define VFLOAT_EXTRACT(result, v, instance) ((v).m_value.c[instance])

CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_return()
{
	m_kernel_exec = andnot(m_exec, m_kernel_exec);
	m_exec = exec_mask::all_off();
}

template<typename SPMDKernel, typename... Args>
CPPSPMD_FORCE_INLINE decltype(auto) spmd_call(Args&&... args)
{
	SPMDKernel kernel;
	kernel.init(exec_mask::all_on());
	return kernel._call(std::forward<Args>(args)...);
}

CPPSPMD_FORCE_INLINE void spmd_kernel::init(const spmd_kernel::exec_mask& kernel_exec)
{
	m_exec = kernel_exec;
	m_internal_exec = exec_mask::all_on();
	m_kernel_exec = kernel_exec;
	m_continue_mask = exec_mask::all_off();

#ifdef _DEBUG
	m_in_loop = false;
#endif
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref& dst, const vfloat& src)
{
	scatter_mask_float4(dst.m_pValue, src.m_value, dst.m_vindex, m_exec.m_mask);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	scatter_float4(dst.m_pValue, src.m_value, dst.m_vindex);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	scatter_mask_float4(dst.m_pValue, src.m_value, dst.m_vindex, m_exec.m_mask);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	scatter_float4(dst.m_pValue, src.m_value, dst.m_vindex);
	return dst;
}

CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_if_break(const vbool& cond)
{
#ifdef _DEBUG
	assert(m_in_loop);
#endif
	
	exec_mask cond_exec(cond);
					
	m_internal_exec = andnot(m_internal_exec & cond_exec, m_internal_exec);
	m_exec = m_kernel_exec & m_internal_exec;
}

// No breaks, continues, etc. allowed
template<typename IfBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_simple_if(const vbool& cond, const IfBody& ifBody)
{
	const exec_mask orig_exec = m_exec;

	exec_mask im = m_exec & exec_mask(cond);

	if (any(im))
	{
		m_exec = im;
		ifBody();
		m_exec = orig_exec;
	}
}

// No breaks, continues, etc. allowed
template<typename IfAnyBody, typename IfAllBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_simple_if_all(const vbool& cond, const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody)
{
	const exec_mask orig_exec = m_exec;

	exec_mask im = m_exec & exec_mask(cond);

	uint32_t mask = im.get_movemask();
	if (mask == 0xF)
	{
		m_exec = im;
		ifAllBody();
		m_exec = orig_exec;
	}
	else if (mask != 0)
	{
		m_exec = im;
		ifAnyBody();
		m_exec = orig_exec;
	}
}

// No breaks, continues, etc. allowed
template<typename IfBody, typename ElseBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_simple_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody &elseBody)
{
	const exec_mask orig_exec = m_exec;

	exec_mask im = m_exec & exec_mask(cond);

	if (any(im))
	{
		m_exec = im;
		ifBody();
	}

	exec_mask em = orig_exec & exec_mask(!cond);

	if (any(em))
	{
		m_exec = em;
		elseBody();
	}
		
	m_exec = orig_exec;
}

template<typename IfBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_if(const vbool& cond, const IfBody& ifBody)
{
	exec_mask orig_internal_exec = m_internal_exec;

	exec_mask cond_exec(cond);
	exec_mask pre_if_internal_exec = m_internal_exec & cond_exec;

	m_internal_exec = pre_if_internal_exec;
	m_exec = m_exec & cond_exec;

	if (any(m_exec))
		ifBody();

	m_internal_exec = andnot(pre_if_internal_exec ^ m_internal_exec, orig_internal_exec);
	m_exec = m_kernel_exec & m_internal_exec;
}

template<typename IfAnyBody, typename IfAllBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_if_all(const vbool& cond, const IfAnyBody& ifAnyBody, const IfAllBody &ifAllBody)
{
	exec_mask orig_internal_exec = m_internal_exec;

	exec_mask cond_exec(cond);
	exec_mask pre_if_internal_exec = m_internal_exec & cond_exec;

	m_internal_exec = pre_if_internal_exec;
	m_exec = m_exec & cond_exec;

	const uint32_t mask = m_exec.get_movemask();

	if (mask == 0xF)
		ifAllBody();
	else if (mask != 0)
		ifAnyBody();

	m_internal_exec = andnot(pre_if_internal_exec ^ m_internal_exec, orig_internal_exec);
	m_exec = m_kernel_exec & m_internal_exec;
}

template<typename IfBody, typename ElseBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody)
{
	bool all_flag;

	{
		exec_mask cond_exec(cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		uint32_t mask = m_exec.get_movemask();
		all_flag = (mask == 0xF);

		if (mask != 0)
			ifBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}

	if (!all_flag)
	{
		exec_mask cond_exec(!cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		if (any(m_exec))
			elseBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}
}

template<typename IfAnyBody, typename IfAllBody, typename ElseAnyBody, typename ElseAllBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_ifelse_all(const vbool& cond, 
	const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody, 
	const ElseAnyBody& elseAnyBody, const ElseAllBody& elseAllBody)
{
	bool all_flag;

	{
		exec_mask cond_exec(cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		uint32_t mask = m_exec.get_movemask();

		all_flag = (mask == 0xF);
		if (all_flag)
			ifAllBody();
		else if (mask != 0)
			ifAnyBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}

	if (!all_flag)
	{
		exec_mask cond_exec(!cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		uint32_t mask = m_exec.get_movemask();

		if (mask == 0xF)
			elseAllBody();
		else if (mask != 0)
			elseAnyBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}
}

template<typename IfAnyBody, typename IfAllBody, typename ElseAnyBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_ifelse_all(const vbool& cond, 
	const IfAnyBody& ifAnyBody, const IfAllBody& ifAllBody, 
	const ElseAnyBody& elseAnyBody)
{
	bool all_flag;

	{
		exec_mask cond_exec(cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		uint32_t mask = m_exec.get_movemask();

		all_flag = (mask == 0xF);
		if (all_flag)
			ifAllBody();
		else if (mask != 0)
			ifAnyBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}

	if (!all_flag)
	{
		exec_mask cond_exec(!cond), orig_internal_exec(m_internal_exec), pre_if_internal_exec = m_internal_exec & cond_exec;

		m_internal_exec = pre_if_internal_exec;
		m_exec = m_exec & cond_exec;

		if (any(m_exec))
			elseAnyBody();

		m_internal_exec = andnot(m_internal_exec ^ pre_if_internal_exec, orig_internal_exec);
		m_exec = m_kernel_exec & m_internal_exec;
	}
}

struct scoped_exec_restorer
{
	exec_mask *m_pMask;
	exec_mask m_prev_mask;
	CPPSPMD_FORCE_INLINE scoped_exec_restorer(exec_mask *pExec_mask) : m_pMask(pExec_mask), m_prev_mask(*pExec_mask) { }
	CPPSPMD_FORCE_INLINE ~scoped_exec_restorer() { *m_pMask = m_prev_mask; }
};

#undef SPMD_SIMPLE_IF
#undef SPMD_SIMPLE_ELSE
#undef SPMD_SIMPLE_END_IF

// Cannot use break, continue, or return inside if/else
#define SPMD_SIMPLE_IF(cond) exec_mask CPPSPMD_GLUER2(_exec_temp, __LINE__)(m_exec & exec_mask(vbool(cond))); if (any(CPPSPMD_GLUER2(_exec_temp, __LINE__))) { CPPSPMD::scoped_exec_restorer CPPSPMD_GLUER2(_exec_restore_, __LINE__)(&m_exec); m_exec = CPPSPMD_GLUER2(_exec_temp, __LINE__);
#define SPMD_SIMPLE_ELSE(cond) } exec_mask CPPSPMD_GLUER2(_exec_temp, __LINE__)(m_exec & exec_mask(!vbool(cond))); if (any(CPPSPMD_GLUER2(_exec_temp, __LINE__))) { CPPSPMD::scoped_exec_restorer CPPSPMD_GLUER2(_exec_restore_, __LINE__)(&m_exec); m_exec = CPPSPMD_GLUER2(_exec_temp, __LINE__);
#define SPMD_SIMPLE_END_IF }

struct scoped_exec_restorer2
{
	spmd_kernel *m_pKernel;
	exec_mask m_orig_internal_mask;
	exec_mask m_pre_if_internal_exec;
		
	CPPSPMD_FORCE_INLINE scoped_exec_restorer2(spmd_kernel *pKernel, const vbool &cond) : 
		m_pKernel(pKernel), 
		m_orig_internal_mask(pKernel->m_internal_exec)
	{ 
		exec_mask cond_exec(cond);
		m_pre_if_internal_exec = pKernel->m_internal_exec & cond_exec;
		pKernel->m_internal_exec = m_pre_if_internal_exec;
		pKernel->m_exec = pKernel->m_exec & cond_exec;
	}

	CPPSPMD_FORCE_INLINE ~scoped_exec_restorer2() 
	{ 
		m_pKernel->m_internal_exec = andnot(m_pre_if_internal_exec ^ m_pKernel->m_internal_exec, m_orig_internal_mask);
		m_pKernel->m_exec = m_pKernel->m_kernel_exec & m_pKernel->m_internal_exec;
	}
};

#undef SPMD_IF
#undef SPMD_ELSE
#undef SPMD_END_IF

#define SPMD_IF(cond) { CPPSPMD::scoped_exec_restorer2 CPPSPMD_GLUER2(_exec_restore2_, __LINE__)(this, vbool(cond)); if (any(m_exec)) {
#define SPMD_ELSE(cond) } } { CPPSPMD::scoped_exec_restorer2 CPPSPMD_GLUER2(_exec_restore2_, __LINE__)(this, !vbool(cond)); if (any(m_exec)) {
#define SPMD_END_IF } }

template<typename ForeachBody>
CPPSPMD_FORCE_INLINE void spmd_kernel::spmd_foreach(int begin, int end, const ForeachBody& foreachBody)
{
	if (begin == end)
		return;
	
	if (!any(m_exec))
		return;

	// We don't support iterating backwards.
	if (begin > end)
		std::swap(begin, end);

	exec_mask prev_continue_mask = m_continue_mask, prev_internal_exec = m_internal_exec;
		
	m_continue_mask = exec_mask::all_off();

	int total_full = (end - begin) / PROGRAM_COUNT;
	int total_partial = (end - begin) % PROGRAM_COUNT;

	lint loop_index = begin + program_index;
	
	const int total_loops = total_full + (total_partial ? 1 : 0);

	for (int i = 0; i < total_loops; i++)
	{
		if (!any(m_exec))
			break;

		int n = PROGRAM_COUNT;
		if ((i == (total_loops - 1)) && (total_partial))
		{
			exec_mask partial_mask = exec_mask{ cmp_gt_int4(set1_int4(total_partial), program_index.m_value) };
			m_internal_exec = m_internal_exec & partial_mask;
			m_exec = m_exec & partial_mask;
			n = total_partial;
		}

		foreachBody(loop_index, n);

		m_internal_exec = m_internal_exec | m_continue_mask;
		m_exec = m_internal_exec & m_kernel_exec;
		m_continue_mask = exec_mask::all_off();

		loop_index.m_value = (loop_index + PROGRAM_COUNT).m_value;
	}

	m_internal_exec = prev_internal_exec;
	m_exec = m_internal_exec & m_kernel_exec;

	m_continue_mask = prev_continue_mask;
}

} // namespace cppspmd_float4

