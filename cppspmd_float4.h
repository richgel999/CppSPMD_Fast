// cppspmd_float4.h
// This is *very* slow, but it's useful for debugging, verification, and porting.
// Originally written by Nicolas Guillemot, Jefferson Amstutz in the "CppSPMD" project.
// 4/20: Richard Geldreich: Macro control flow, more SIMD instruction sets, optimizations, supports using multiple SIMD instruction sets in same executable. Still a work in progress!

// The original CppSPMD header, which this version distantly derives from, used the MIT license:
// Copyright 2016 Nicolas Guillemot
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without 
// restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom 
// the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
// AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <utility>
#include <algorithm>

// Set to 1 to use std::fmaf()
#define CPPSPMD_USE_FMAF 0

#undef CPPSPMD_SSE
#undef CPPSPMD_AVX1
#undef CPPSPMD_AVX2
#undef CPPSPMD_AVX
#undef CPPSPMD_FLOAT4
#undef CPPSPMD_INT16

#define CPPSPMD_SSE 0
#define CPPSPMD_AVX 0
#define CPPSPMD_AVX1 0
#define CPPSPMD_AVX2 0
#define CPPSPMD_FLOAT4 1
#define CPPSPMD_INT16 0

#ifdef _MSC_VER
	#ifndef CPPSPMD_DECL
		#define CPPSPMD_DECL(type, name) __declspec(align(16)) type name
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
#ifdef _DEBUG
#define CPPSPMD_FORCE_INLINE inline
#else
#define CPPSPMD_FORCE_INLINE __forceinline
#endif
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

#ifndef CPPSPMD_NAME
#define CPPSPMD_NAME(a) CPPSPMD_GLUER2(a, CPPSPMD_ARCH)
#endif

#undef VASSERT
#define VCOND(cond) ((exec_mask(vbool(cond)) & m_exec).get_movemask() == m_exec.get_movemask())
#define VASSERT(cond) assert( VCOND(cond) )

#undef CPPSPMD_ALIGNMENT
#define CPPSPMD_ALIGNMENT (16)

namespace CPPSPMD
{

const int PROGRAM_COUNT_SHIFT = 2;
const int PROGRAM_COUNT = 1 << PROGRAM_COUNT_SHIFT;

template <typename N> inline N* aligned_new() { void* p = _mm_malloc(sizeof(N), 64); new (p) N;	return static_cast<N*>(p); }
template <typename N> void aligned_delete(N* p) { if (p) { p->~N(); _mm_free(p); } }

CPPSPMD_DECL(const uint32_t, g_allones_128[4]) = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
CPPSPMD_DECL(const float, g_onef_128[4]) = { 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(const uint32_t, g_oneu_128[4]) = { 1, 1, 1, 1 };

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

CPPSPMD_FORCE_INLINE int minu(int a, int b) { return (uint32_t)a < (uint32_t)b ? a : b; }
CPPSPMD_FORCE_INLINE int maxu(int a, int b) { return (uint32_t)a > (uint32_t)b ? a : b; }

#if CPPSPMD_USE_FMAF
CPPSPMD_FORCE_INLINE float my_fmaf(float a, float b, float c) { return std::fmaf(a, b, c); }
#else
CPPSPMD_FORCE_INLINE float my_fmaf(float a, float b, float c) { return (a * b) + c; }
#endif

// Work around for std::roundf(): halfway cases rounded away from zero, while with _mm_round_ss() it rounds halfway cases towards zero.
CPPSPMD_FORCE_INLINE float my_roundf(float a) 
{ 
	float f = std::roundf(a);
	
	// Are we exactly halfway?
	if (fabs(a - f) == .5f)
	{
		int q = (int)f;
		if (q & 1)
		{
			// Fix rounding so it's like _mm_round_ss().
			f += ((a < 0.0f) ? 1.0f : -1.0f);
			if ((a < 0.0f) && (f == 0.0f))
			{
				f = -0.0f;
			}
		}
	}

	return f;
}

CPPSPMD_FORCE_INLINE float my_truncf(float a) { return std::truncf(a); }

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

CPPSPMD_FORCE_INLINE int4 mulhiu_int4(const int4 &a, const int4 &b) 
{ 
	return int4( 
		(int)(((uint64_t)((uint32_t)a.c[0]) * (uint64_t)((uint32_t)b.c[0])) >> 32U), 
		(int)(((uint64_t)((uint32_t)a.c[1]) * (uint64_t)((uint32_t)b.c[1])) >> 32U), 
		(int)(((uint64_t)((uint32_t)a.c[2]) * (uint64_t)((uint32_t)b.c[2])) >> 32U), 
		(int)(((uint64_t)((uint32_t)a.c[3]) * (uint64_t)((uint32_t)b.c[3])) >> 32U) );
}

CPPSPMD_FORCE_INLINE int4 div_int4(const int4 &a, const int4 &b) { return int4(b.c[0] ? (a.c[0] / b.c[0]) : INT_MIN, b.c[1] ? (a.c[1] / b.c[1]) : INT_MIN, b.c[2] ? (a.c[2] / b.c[2]) : INT_MIN, b.c[3] ? (a.c[3] / b.c[3]) : INT_MIN); }
CPPSPMD_FORCE_INLINE int4 div_int4(const int4 &a, int b) { return b ? int4(a.c[0] / b, a.c[1] / b, a.c[2] / b, a.c[3] / b) : int4(INT_MIN, INT_MIN, INT_MIN, INT_MIN); }
CPPSPMD_FORCE_INLINE int4 mod_int4(const int4 &a, const int4 &b) { return int4(b.c[0] ? (a.c[0] % b.c[0]) : 0, b.c[1] ? (a.c[1] % b.c[1]) : 0, b.c[2] ? (a.c[2] % b.c[2]) : 0, b.c[3] ? (a.c[3] % b.c[3]) : 0); }
CPPSPMD_FORCE_INLINE int4 mod_int4(const int4 &a, int b) { return b ? int4(a.c[0] % b, a.c[1] % b, a.c[2] % b, a.c[3] % b) : int4(0, 0, 0, 0); }
CPPSPMD_FORCE_INLINE int4 neg_int4(const int4 &a) { return int4(-a.c[0], -a.c[1], -a.c[2], -a.c[3]); }		
CPPSPMD_FORCE_INLINE int4 inv_int4(const int4 &a) { return int4(~a.c[0], ~a.c[1], ~a.c[2], ~a.c[3]); }
	
CPPSPMD_FORCE_INLINE int4 and_int4(const int4 &a, const int4 &b) { return int4(a.c[0] & b.c[0], a.c[1] & b.c[1], a.c[2] & b.c[2], a.c[3] & b.c[3]); }
CPPSPMD_FORCE_INLINE int4 or_int4(const int4 &a, const int4 &b) { return int4(a.c[0] | b.c[0], a.c[1] | b.c[1], a.c[2] | b.c[2], a.c[3] | b.c[3]); }
CPPSPMD_FORCE_INLINE int4 xor_int4(const int4 &a, const int4 &b) { return int4(a.c[0] ^ b.c[0], a.c[1] ^ b.c[1], a.c[2] ^ b.c[2], a.c[3] ^ b.c[3]); }
CPPSPMD_FORCE_INLINE int4 andnot_int4(const int4 &a, const int4 &b) { return int4((~a.c[0]) & b.c[0], (~a.c[1]) & b.c[1], (~a.c[2]) & b.c[2], (~a.c[3]) & b.c[3]); }

// C's shifts have undefined behavior, but SSE's doesn't, so we need to emulate.
CPPSPMD_FORCE_INLINE int4 shift_left_int4(const int4 &a, const int4 &b) 
{ 
	return int4((b.c[0] > 31) ? 0 : (a.c[0] << b.c[0]), (b.c[1] > 31) ? 0 : (a.c[1] << b.c[1]), (b.c[2] > 31) ? 0 : (a.c[2] << b.c[2]), (b.c[3] > 31) ? 0 : (a.c[3] << b.c[3])); 
}

CPPSPMD_FORCE_INLINE int4 shift_right_int4(const int4 &a, const int4 &b) 
{ 
	return int4(a.c[0] >> std::min(31, b.c[0]), a.c[1] >> std::min(31, b.c[1]), a.c[2] >> std::min(31, b.c[2]), a.c[3] >> std::min(31, b.c[3])); 
}

CPPSPMD_FORCE_INLINE int4 unsigned_shift_right_int4(const int4 &a, const int4 &b) 
{ 
	return int4( (b.c[0] > 31) ? 0 : ((uint32_t)a.c[0]) >> b.c[0], (b.c[1] > 31) ? 0 : ((uint32_t)a.c[1]) >> b.c[1], (b.c[2] > 31) ? 0 : ((uint32_t)a.c[2]) >> b.c[2], (b.c[3] > 31) ? 0 : ((uint32_t)a.c[3]) >> b.c[3]);
}

CPPSPMD_FORCE_INLINE int4 shift_left_int4(const int4 &a, int b) 
{ 
	if (b > 31)
		return int4(0, 0, 0, 0);
	else
		return int4(a.c[0] << b, a.c[1] << b, a.c[2] << b, a.c[3] << b); 
}

CPPSPMD_FORCE_INLINE int4 shift_right_int4(const int4 &a, int b) 
{ 
	b = std::min(b, 31);
	return int4(a.c[0] >> b, a.c[1] >> b, a.c[2] >> b, a.c[3] >> b); 
}

CPPSPMD_FORCE_INLINE int4 unsigned_shift_right_int4(const int4 &a, int b) 
{ 
	if (b > 31)
		return int4(0, 0, 0, 0);
	else
		return int4(((uint32_t)a.c[0]) >> b, ((uint32_t)a.c[1]) >> b, ((uint32_t)a.c[2]) >> b, ((uint32_t)a.c[3]) >> b); 
}

CPPSPMD_FORCE_INLINE int4 cmp_eq_int4(const int4 &a, const int4 &b) { return int4(-(a.c[0] == b.c[0]), -(a.c[1] == b.c[1]), -(a.c[2] == b.c[2]), -(a.c[3] == b.c[3])); }
CPPSPMD_FORCE_INLINE int4 cmp_gt_int4(const int4 &a, const int4 &b) { return int4(-(a.c[0] > b.c[0]), -(a.c[1] > b.c[1]), -(a.c[2] > b.c[2]), -(a.c[3] > b.c[3])); }

CPPSPMD_FORCE_INLINE int movemask_int4(const int4 &a) 
{	
	return ((a.c[0] >> 31) & 1) | (((a.c[1] >> 31) & 1) << 1)| (((a.c[2] >> 31) & 1) << 2)| (((a.c[3] >> 31) & 1) << 3); 
}

CPPSPMD_FORCE_INLINE int4 min_int4(const int4 &a, const int4 &b) { return int4(mini(a.c[0], b.c[0]), mini(a.c[1], b.c[1]), mini(a.c[2], b.c[2]), mini(a.c[3], b.c[3])); }
CPPSPMD_FORCE_INLINE int4 max_int4(const int4 &a, const int4 &b) { return int4(maxi(a.c[0], b.c[0]), maxi(a.c[1], b.c[1]), maxi(a.c[2], b.c[2]), maxi(a.c[3], b.c[3])); }

CPPSPMD_FORCE_INLINE int4 min_uint4(const int4& a, const int4& b) { return int4(minu(a.c[0], b.c[0]), minu(a.c[1], b.c[1]), minu(a.c[2], b.c[2]), minu(a.c[3], b.c[3])); }
CPPSPMD_FORCE_INLINE int4 max_uint4(const int4& a, const int4& b) { return int4(maxu(a.c[0], b.c[0]), maxu(a.c[1], b.c[1]), maxu(a.c[2], b.c[2]), maxu(a.c[3], b.c[3])); }

CPPSPMD_FORCE_INLINE int4 abs_int4(const int4 &a) { return int4(::abs(a.c[0]), ::abs(a.c[1]), ::abs(a.c[2]), ::abs(a.c[3])); }

CPPSPMD_FORCE_INLINE uint32_t byteswap_uint32(uint32_t v) { uint32_t b0 = v & 0xFF; uint32_t b1 = (v >> 8U) & 0xFF; uint32_t b2 = (v >> 16U) & 0xFF; uint32_t b3 = (v >> 24U) & 0xFF; return b3 | (b2 << 8) | (b1 << 16) | (b0 << 24); }
CPPSPMD_FORCE_INLINE int4 byteswap_int4(const int4& a) { return int4(byteswap_uint32(a.c[0]), byteswap_uint32(a.c[1]), byteswap_uint32(a.c[2]), byteswap_uint32(a.c[3])); }

CPPSPMD_FORCE_INLINE int4 blendv_int4(const int4 &a, const int4 &b, const int4 &c) 
{ 
	const int m0 = c.c[0] >> 31, m1 = c.c[1] >> 31, m2 = c.c[2] >> 31, m3 = c.c[3] >> 31;
	//return int4( (b.c[0] & c.c[0]) | (a.c[0] & (~c.c[0])), (b.c[1] & c.c[1]) | (a.c[1] & (~c.c[1])), (b.c[2] & c.c[2]) | (a.c[2] & (~c.c[2])), (b.c[3] & c.c[3]) | (a.c[3] & (~c.c[3])) ); 
	return int4( (b.c[0] & m0) | (a.c[0] & (~m0)), (b.c[1] & m1) | (a.c[1] & (~m1)), (b.c[2] & m2) | (a.c[2] & (~m2)), (b.c[3] & m3) | (a.c[3] & (~m3)) ); 
}

CPPSPMD_FORCE_INLINE int4 blendv_epi8_int4(const int4 &a, const int4 &b, const int4 &c) 
{ 
	int4 r;
	for (int i = 0; i < 16; i++)
	{
		uint8_t ab = ((const uint8_t *)&a)[i];
		uint8_t bb = ((const uint8_t *)&b)[i];
		uint8_t mask = (((const int8_t *)&c)[i]) >> 7;
		((uint8_t *)&r)[i] = (bb & mask) | (ab & (~mask));
	}
	return r;
}

CPPSPMD_FORCE_INLINE int clamp32(int a, int lo, int hi) { return std::max(lo, std::min(a, hi)); }
CPPSPMD_FORCE_INLINE int saturate_i8_32(int a) { return std::max(-128, std::min(a, 127)); }
CPPSPMD_FORCE_INLINE int clamp_lo_u8_32(int a) { return std::max(a, 0); }
CPPSPMD_FORCE_INLINE int clamp_hi_u8_32(int a) { return std::min(a, 255); }

CPPSPMD_FORCE_INLINE int4 add_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)(((const uint8_t*)&a)[i] + ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 sub_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)(((const uint8_t*)&a)[i] - ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 adds_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)clamp_hi_u8_32(((const uint8_t*)&a)[i] + ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 subs_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)clamp_lo_u8_32(((const uint8_t*)&a)[i] - ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 avg_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)((((const uint8_t*)&a)[i] + ((const uint8_t*)&b)[i] + 1) >> 1); return result; }
CPPSPMD_FORCE_INLINE int4 max_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)std::max<uint8_t>(((const uint8_t*)&a)[i], ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 min_epu8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (uint8_t)std::min<uint8_t>(((const uint8_t*)&a)[i], ((const uint8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 sad_epu8(const int4& a, const int4& b) 
{ 
	uint8_t vals[16];
	for (int i = 0; i < 16; i++)
		vals[i] = (uint8_t)std::abs((int)((const uint8_t*)&a)[i] - (int)((const uint8_t*)&b)[i]);
	
	int4 result;
	result.c[0] = (uint16_t)(vals[0] + vals[1] + vals[2] + vals[3] + vals[4] + vals[5] + vals[6] + vals[7]);
	result.c[1] = 0;
	result.c[2] = (uint16_t)(vals[8] + vals[9] + vals[10] + vals[11] + vals[12] + vals[13] + vals[14] + vals[15]);
	result.c[3] = 0;
	
	return result; 
}

CPPSPMD_FORCE_INLINE int4 unpacklo_epi8(const int4& a, const int4& b)
{
	int4 result;
	uint8_t* pDst = (uint8_t*)&result;
	const uint8_t* pA = (const uint8_t*)&a;
	const uint8_t* pB = (const uint8_t*)&b;
	
	for (int i = 0; i < 8; i++)
	{
		pDst[i * 2 + 0] = pA[i];
		pDst[i * 2 + 1] = pB[i];
	}

	return result;
}

CPPSPMD_FORCE_INLINE int4 unpackhi_epi8(const int4& a, const int4& b)
{
	int4 result;
	uint8_t* pDst = (uint8_t*)&result;
	const uint8_t* pA = (const uint8_t*)&a;
	const uint8_t* pB = (const uint8_t*)&b;

	for (int i = 0; i < 8; i++)
	{
		pDst[i * 2 + 0] = pA[8 + i];
		pDst[i * 2 + 1] = pB[8 + i];
	}

	return result;
}

CPPSPMD_FORCE_INLINE int movemask_epi8(const int4& a)
{
	const uint8_t* pA = (const uint8_t*)&a;
	int mask = 0;
	for (int i = 0; i < 16; i++)
		mask |= ((pA[i] >> 7) << i);
	return mask;
}

CPPSPMD_FORCE_INLINE int4 lane_shuffle_ps_float4(const float4& a, int control)
{
	assert(control >= 0 && control < 256);
	float4 result;
	result.c[0] = a.c[control & 3];
	result.c[1] = a.c[(control >> 2) & 3];
	result.c[2] = a.c[(control >> 4) & 3];
	result.c[3] = a.c[(control >> 6) & 3];
	return result;
}

CPPSPMD_FORCE_INLINE int4 lane_shuffle_epi32_int4(const int4& a, int control)
{
	assert(control >= 0 && control < 256);
	int4 result;
	result.c[0] = a.c[control & 3];
	result.c[1] = a.c[(control >> 2) & 3];
	result.c[2] = a.c[(control >> 4) & 3];
	result.c[3] = a.c[(control >> 6) & 3];
	return result;
}

CPPSPMD_FORCE_INLINE int4 lane_shufflelo_epi16_int4(const int4& a, int control)
{
	assert(control >= 0 && control < 256);
	int4 result;
	((int16_t*)&result)[0] = ((const int16_t*)&a)[control & 3];
	((int16_t*)&result)[1] = ((const int16_t*)&a)[(control >> 2) & 3];
	((int16_t*)&result)[2] = ((const int16_t*)&a)[(control >> 4) & 3];
	((int16_t*)&result)[3] = ((const int16_t*)&a)[(control >> 6) & 3];
	result.c[2] = a.c[2];
	result.c[3] = a.c[3];
	return result;
}

CPPSPMD_FORCE_INLINE int4 lane_shufflehi_epi16_int4(const int4& a, int control)
{
	assert(control >= 0 && control < 256);
	int4 result;
	result.c[0] = a.c[0];
	result.c[1] = a.c[1];
	((int16_t*)&result)[4] = ((const int16_t*)&a)[4 + (control & 3)];
	((int16_t*)&result)[5] = ((const int16_t*)&a)[4 + ((control >> 2) & 3)];
	((int16_t*)&result)[6] = ((const int16_t*)&a)[4 + ((control >> 4) & 3)];
	((int16_t*)&result)[7] = ((const int16_t*)&a)[4 + ((control >> 6) & 3)];
	return result;
}

CPPSPMD_FORCE_INLINE int4 lane_shift_left_bytes(const int4& a, int l)
{
	if ((uint32_t)l > 16)
		l = 16;

	int4 result;
	for (int i = 0; i < 16; i++)
	{
		uint8_t c = (i >= l) ? ((const uint8_t *)&a)[i - l] : 0;
		((uint8_t *)&result)[i] = c;
	}
	
	return result;
}

CPPSPMD_FORCE_INLINE int4 lane_shift_right_bytes(const int4& a, int l)
{
	if ((uint32_t)l > 16)
		l = 16;

	int4 result;
	for (int i = 0; i < 16; i++)
	{
		uint8_t c = ((i + l) <= 15) ? ((const uint8_t *)&a)[i + l] : 0;
		((uint8_t *)&result)[i] = c;
	}
	
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpacklo_epi8_int4(const int4& a, const int4& b)
{
	int4 result;
	const uint8_t* pA = (const uint8_t*)&a;
	const uint8_t* pB = (const uint8_t*)&b;
	uint8_t* pR = (uint8_t*)&result;
	for (int i = 0; i < 8; i++)
	{
		pR[i * 2 + 0] = pA[i];
		pR[i * 2 + 1] = pB[i];
	}
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpackhi_epi8_int4(const int4& a, const int4& b)
{
	int4 result;
	const uint8_t* pA = (const uint8_t*)&a;
	const uint8_t* pB = (const uint8_t*)&b;
	uint8_t* pR = (uint8_t*)&result;
	for (int i = 0; i < 8; i++)
	{
		pR[i * 2 + 0] = pA[8 + i];
		pR[i * 2 + 1] = pB[8 + i];
	}
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpacklo_epi16_int4(const int4& a, const int4& b)
{
	int4 result;
	const uint16_t* pA = (const uint16_t*)&a;
	const uint16_t* pB = (const uint16_t*)&b;
	uint16_t* pR = (uint16_t*)&result;
	for (int i = 0; i < 4; i++)
	{
		pR[i * 2 + 0] = pA[i];
		pR[i * 2 + 1] = pB[i];
	}
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpackhi_epi16_int4(const int4& a, const int4& b)
{
	int4 result;
	const uint16_t* pA = (const uint16_t*)&a;
	const uint16_t* pB = (const uint16_t*)&b;
	uint16_t* pR = (uint16_t*)&result;
	for (int i = 0; i < 4; i++)
	{
		pR[i * 2 + 0] = pA[4 + i];
		pR[i * 2 + 1] = pB[4 + i];
	}
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpacklo_epi32_int4(const int4& a, const int4& b)
{
	int4 result;
	result.c[0] = a.c[0];
	result.c[1] = b.c[0];
	result.c[2] = a.c[1];
	result.c[3] = b.c[1];
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpackhi_epi32_int4(const int4& a, const int4& b)
{
	int4 result;
	result.c[0] = a.c[2];
	result.c[1] = b.c[2];
	result.c[2] = a.c[3];
	result.c[3] = b.c[3];
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpacklo_epi64_int4(const int4& a, const int4& b)
{
	int4 result;
	result.c[0] = a.c[0];
	result.c[1] = a.c[1];
	result.c[2] = b.c[0];
	result.c[3] = b.c[1];
	return result;
}

CPPSPMD_FORCE_INLINE int4 unpackhi_epi64_int4(const int4& a, const int4& b)
{
	int4 result;
	result.c[0] = a.c[2];
	result.c[1] = a.c[3];
	result.c[2] = b.c[2];
	result.c[3] = b.c[3];
	return result;
}

CPPSPMD_FORCE_INLINE int4 add_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)(((const int8_t*)&a)[i] + ((const int8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 sub_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)(((const int8_t*)&a)[i] - ((const int8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 adds_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)saturate_i8_32(((const int8_t*)&a)[i] + ((const int8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 subs_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)saturate_i8_32(((const int8_t*)&a)[i] - ((const int8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 max_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)std::max<int8_t>(((const int8_t*)&a)[i], ((const int8_t*)&b)[i]); return result; }
CPPSPMD_FORCE_INLINE int4 min_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((int8_t*)&result)[i] = (int8_t)std::min<int8_t>(((const int8_t*)&a)[i], ((const int8_t*)&b)[i]); return result; }

CPPSPMD_FORCE_INLINE int4 cmpeq_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (((const int8_t*)&a)[i] == ((const int8_t*)&b)[i]) ? 0xFF : 00; return result; }
CPPSPMD_FORCE_INLINE int4 cmplt_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (((const int8_t*)&a)[i] < ((const int8_t*)&b)[i]) ? 0xFF : 00; return result; }
CPPSPMD_FORCE_INLINE int4 cmpgt_epi8(const int4& a, const int4& b) { int4 result; for (int i = 0; i < 16; i++) ((uint8_t*)&result)[i] = (((const int8_t*)&a)[i] > ((const int8_t*)&b)[i]) ? 0xFF : 00; return result; }

CPPSPMD_FORCE_INLINE float4 blend_float4(const float4 &a, const float4 &b, const float4 &c) { return cast_int4_to_float4(blendv_int4(cast_float4_to_int4(a), cast_float4_to_int4(b), cast_float4_to_int4(c))); }
CPPSPMD_FORCE_INLINE float4 blend_float4(const float4 &a, const float4 &b, const int4 &c) { return cast_int4_to_float4(blendv_int4(cast_float4_to_int4(a), cast_float4_to_int4(b), c)); }

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

CPPSPMD_FORCE_INLINE int4 shuffle_epi8_int4(const int4 &a, const int4 &b)
{
	int4 result;
	for (int i = 0; i < 16; i++)
	{
		int v = ((const uint8_t*)&b)[i];
		((uint8_t*)&result)[i] = (v & 128) ? 0 : ((const uint8_t*)&a)[v & 15];
	}
	return result;
}

CPPSPMD_FORCE_INLINE int4 add_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] + pB[i]);
	return r;
}

CPPSPMD_FORCE_INLINE int4 adds_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)clamp32(pA[i] + pB[i], INT16_MIN, INT16_MAX);
	return r;
}

CPPSPMD_FORCE_INLINE int4 adds_epu16_int4(const int4& a, const int4& b)
{
	int4 r;
	const uint16_t *pA = (const uint16_t *)&a;
	const uint16_t *pB = (const uint16_t *)&b;
	uint16_t *pR = (uint16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (uint16_t)std::min<uint32_t>(pA[i] + pB[i], UINT16_MAX);
	return r;
}

CPPSPMD_FORCE_INLINE int4 sub_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] - pB[i]);
	return r;
}

CPPSPMD_FORCE_INLINE int4 subs_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)clamp32((int)pA[i] - (int)pB[i], INT16_MIN, INT16_MAX);
	return r;
}

CPPSPMD_FORCE_INLINE int4 subs_epu16_int4(const int4& a, const int4& b)
{
	int4 r;
	const uint16_t *pA = (const uint16_t *)&a;
	const uint16_t *pB = (const uint16_t *)&b;
	uint16_t *pR = (uint16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (uint16_t)std::max<int>((int)pA[i] - (int)pB[i], 0);
	return r;
}

CPPSPMD_FORCE_INLINE int4 avg_epu16_int4(const int4& a, const int4& b)
{
	int4 r;
	const uint16_t *pA = (const uint16_t *)&a;
	const uint16_t *pB = (const uint16_t *)&b;
	uint16_t *pR = (uint16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (uint16_t)((pA[i] + pB[i] + 1) >> 1U);
	return r;
}

CPPSPMD_FORCE_INLINE int4 mullo_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)((int)pA[i] * (int)pB[i]);
	return r;
}

CPPSPMD_FORCE_INLINE int4 mulhi_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(((int)pA[i] * (int)pB[i]) >> 16);
	return r;
}

CPPSPMD_FORCE_INLINE int4 mulhi_epu16_int4(const int4& a, const int4& b)
{
	int4 r;
	const uint16_t *pA = (const uint16_t *)&a;
	const uint16_t *pB = (const uint16_t *)&b;
	uint16_t *pR = (uint16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (uint16_t)(((uint32_t)pA[i] * (uint32_t)pB[i]) >> 16U);
	return r;
}

CPPSPMD_FORCE_INLINE int4 min_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)std::min(pA[i], pB[i]);
	return r;
}

CPPSPMD_FORCE_INLINE int4 max_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)std::max(pA[i], pB[i]);
	return r;
}

CPPSPMD_FORCE_INLINE int4 madd_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int32_t *pR = (int32_t *)&r;
	for (int i = 0; i < 4; i++)
		pR[i] = pA[i * 2] * pB[i * 2] + pA[i * 2 + 1] * pB[i * 2 + 1];
	return r;
}

CPPSPMD_FORCE_INLINE int4 cmpeq_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)((pA[i] == pB[i]) ? 0xFFFF : 0);
	return r;
}

CPPSPMD_FORCE_INLINE int4 cmpgt_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)((pA[i] > pB[i]) ? 0xFFFF : 0);
	return r;
}

CPPSPMD_FORCE_INLINE int4 cmplt_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)((pA[i] < pB[i]) ? 0xFFFF : 0);
	return r;
}

CPPSPMD_FORCE_INLINE int4 packs_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	int8_t *pR = (int8_t *)&r;
	for (int i = 0; i < 8; i++)
	{
		pR[i] = (int8_t)clamp32(pA[i], INT8_MIN, INT8_MAX);
		pR[i + 8] = (int8_t)clamp32(pB[i], INT8_MIN, INT8_MAX);
	}
	return r;
}

CPPSPMD_FORCE_INLINE int4 packus_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int16_t *pB = (const int16_t *)&b;
	uint8_t *pR = (uint8_t *)&r;
	for (int i = 0; i < 8; i++)
	{
		pR[i] = (uint8_t)clamp32(pA[i], 0, 255);
		pR[i + 8] = (uint8_t)clamp32(pB[i], 0, 255);
	}
	return r;
}

CPPSPMD_FORCE_INLINE int4 sll_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int64_t *pB = (const int64_t *)&b;

	uint32_t c = pB[0];
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] << c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 sra_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int64_t *pB = (const int64_t *)&b;

	uint32_t c = pB[0];
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] >> c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 srl_epi16_int4(const int4& a, const int4& b)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	const int64_t *pB = (const int64_t *)&b;

	uint32_t c = pB[0];
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(((uint16_t)pA[i]) >> c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 slli_epi16_int4(const int4& a, uint32_t c)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
		
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] << c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 srai_epi16_int4(const int4& a, uint32_t c)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
	
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(pA[i] >> c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 srli_epi16_int4(const int4& a, uint32_t c)
{
	int4 r;
	const int16_t *pA = (const int16_t *)&a;
		
	c = std::min(c, 16U);

	int16_t *pR = (int16_t *)&r;
	for (int i = 0; i < 8; i++)
		pR[i] = (int16_t)(((uint16_t)pA[i]) >> c);
	return r;
}

CPPSPMD_FORCE_INLINE int4 mul_epu32_int4(const int4& a, const int4& b)
{
	int4 result;
	uint64_t *pR = (uint64_t *)&result;
	pR[0] = (uint64_t)a.c[0] * (uint32_t)b.c[0];
	pR[1] = (uint64_t)a.c[1] * (uint32_t)b.c[1];
	return result;
}

const uint32_t ALL_ON_MOVEMASK = 0xF;

struct spmd_kernel
{
	struct vint;
	struct lint;
	struct vbool;
	struct vfloat;

	typedef int int_t;
	typedef vint vint_t;
	typedef lint lint_t;
		
	// Exec mask
	struct exec_mask
	{
		int4 m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE exec_mask(const exec_mask& b) { m_mask.c[0] = b.m_mask.c[0]; m_mask.c[1] = b.m_mask.c[1]; m_mask.c[2] = b.m_mask.c[2]; m_mask.c[3] = b.m_mask.c[3]; }

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const int4& mask) : m_mask(mask) { }

		CPPSPMD_FORCE_INLINE void enable_lane(uint32_t lane) { memset(&m_mask, 0, sizeof(m_mask)); m_mask.c[lane] = UINT32_MAX; }
				
		static CPPSPMD_FORCE_INLINE exec_mask all_on()	{ return exec_mask{ set1_int4(UINT32_MAX) }; }
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ set1_int4(0) }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return movemask_int4(m_mask); }
	};

	friend CPPSPMD_FORCE_INLINE bool all(const exec_mask& e);
	friend CPPSPMD_FORCE_INLINE bool any(const exec_mask& e);

	// true if all lanes active
	CPPSPMD_FORCE_INLINE bool spmd_all() const { return all(m_exec); }
	// true if any lanes active
	CPPSPMD_FORCE_INLINE bool spmd_any() const { return any(m_exec); }
	// true if no lanes active
	CPPSPMD_FORCE_INLINE bool spmd_none() { return !any(m_exec); }

	// true if cond is true for all active lanes - false if no active lanes
	CPPSPMD_FORCE_INLINE bool spmd_all(const vbool& e) { uint32_t m = m_exec.get_movemask(); return (m != 0) && ((exec_mask(e) & m_exec).get_movemask() == m); }
	// true if cond is true for any active lanes
	CPPSPMD_FORCE_INLINE bool spmd_any(const vbool& e) { return (exec_mask(e) & m_exec).get_movemask() != 0; }
	// false if cond is true for any active lanes
	CPPSPMD_FORCE_INLINE bool spmd_none(const vbool& e) { return !spmd_any(e); }

	friend CPPSPMD_FORCE_INLINE exec_mask operator^ (const exec_mask& a, const exec_mask& b);
	friend CPPSPMD_FORCE_INLINE exec_mask operator& (const exec_mask& a, const exec_mask& b);
	friend CPPSPMD_FORCE_INLINE exec_mask operator| (const exec_mask& a, const exec_mask& b);
		
	exec_mask m_exec;
	exec_mask m_kernel_exec;
	exec_mask m_continue_mask;
#ifdef _DEBUG
	bool m_in_loop;
#endif
		
	CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return m_exec.get_movemask(); }
		
	void init(const exec_mask& kernel_exec);
	
	// Varying bool
	struct vbool
	{
		int4 m_value;

		vbool() = default;

		CPPSPMD_FORCE_INLINE vbool(bool value) : m_value(set1_int4(value ? UINT32_MAX : 0)) { }

		CPPSPMD_FORCE_INLINE vbool(const vbool& b) { m_value.c[0] = b.m_value.c[0]; m_value.c[1] = b.m_value.c[1]; m_value.c[2] = b.m_value.c[2]; m_value.c[3] = b.m_value.c[3]; }

		CPPSPMD_FORCE_INLINE explicit vbool(const int4& value) : m_value(value) { assert((m_value.c[0] == 0) || ((uint32_t)m_value.c[0] == UINT32_MAX)); assert((m_value.c[1] == 0) || ((uint32_t)m_value.c[1] == UINT32_MAX)); assert((m_value.c[2] == 0) || ((uint32_t)m_value.c[2] == UINT32_MAX)); assert((m_value.c[3] == 0) || ((uint32_t)m_value.c[3] == UINT32_MAX)); }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const;
		CPPSPMD_FORCE_INLINE explicit operator vint() const;
								
	private:
		vbool& operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = blendv_int4(dst.m_value, src.m_value, m_exec.m_mask);
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

	CPPSPMD_FORCE_INLINE vint load_bytes_all(const cint_vref& src)
	{
		vint result;
		for (int i = 0; i < 4; i++)
			result.m_value.c[i] = ((const int32_t*)((const uint8_t*)src.m_pValue + src.m_vindex.c[i]))[0];
		return result;
	}

	CPPSPMD_FORCE_INLINE vint load_words_all(const cint_vref& src)
	{
		vint result;
		for (int i = 0; i < 4; i++)
			result.m_value.c[i] = ((const int32_t*)((const uint8_t*)src.m_pValue + 2 * src.m_vindex.c[i]))[0];
		return result;
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
		int mask = movemask_int4(m_exec.m_mask);
		
		if (mask & 1) dst.m_pValue[dst.m_vindex.c[0]].m_value.c[0] = src.m_value.c[0];
		if (mask & 2) dst.m_pValue[dst.m_vindex.c[1]].m_value.c[1] = src.m_value.c[1];
		if (mask & 4) dst.m_pValue[dst.m_vindex.c[2]].m_value.c[2] = src.m_value.c[2];
		if (mask & 8) dst.m_pValue[dst.m_vindex.c[3]].m_value.c[3] = src.m_value.c[3];

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		int mask = movemask_int4(m_exec.m_mask);

		float4 k = set1_float4(0.0f);

		if (mask & 1) k.c[0] = src.m_pValue[src.m_vindex.c[0]].m_value.c[0];
		if (mask & 2) k.c[1] = src.m_pValue[src.m_vindex.c[1]].m_value.c[1];
		if (mask & 4) k.c[2] = src.m_pValue[src.m_vindex.c[2]].m_value.c[2];
		if (mask & 8) k.c[3] = src.m_pValue[src.m_vindex.c[3]].m_value.c[3];

		return vfloat{ k };
	}

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		int mask = movemask_int4(m_exec.m_mask);
		
		if (mask & 1) dst.m_pValue[dst.m_vindex.c[0]].m_value.c[0] = src.m_value.c[0];
		if (mask & 2) dst.m_pValue[dst.m_vindex.c[1]].m_value.c[1] = src.m_value.c[1];
		if (mask & 4) dst.m_pValue[dst.m_vindex.c[2]].m_value.c[2] = src.m_value.c[2];
		if (mask & 8) dst.m_pValue[dst.m_vindex.c[3]].m_value.c[3] = src.m_value.c[3];

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const vint_vref& src)
	{
		int mask = movemask_int4(m_exec.m_mask);

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

	CPPSPMD_FORCE_INLINE lint& store_all(lint& dst, const lint& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
	
	const lint program_index = lint{ int4(0, 1, 2, 3) };
	
	// SPMD condition helpers

	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_if(const vbool& cond, const IfBody& ifBody);

	CPPSPMD_FORCE_INLINE void spmd_if_break(const vbool& cond);

	// No breaks, continues, etc. allowed
	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_sif(const vbool& cond, const IfBody& ifBody);
		
	// No breaks, continues, etc. allowed
	template<typename IfBody, typename ElseBody>
	CPPSPMD_FORCE_INLINE void spmd_sifelse(const vbool& cond, const IfBody& ifBody, const ElseBody &elseBody);
				
	template<typename IfBody, typename ElseBody>
	CPPSPMD_FORCE_INLINE void spmd_ifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody);

	template<typename WhileCondBody, typename WhileBody>
	CPPSPMD_FORCE_INLINE void spmd_while(const WhileCondBody& whileCondBody, const WhileBody& whileBody);

	template<typename ForInitBody, typename ForCondBody, typename ForIncrBody, typename ForBody>
	CPPSPMD_FORCE_INLINE void spmd_for(const ForInitBody& forInitBody, const ForCondBody& forCondBody, const ForIncrBody& forIncrBody, const ForBody& forBody);

	template<typename ForeachBody>
	CPPSPMD_FORCE_INLINE void spmd_foreach(int begin, int end, const ForeachBody& foreachBody);

#ifdef _DEBUG
	CPPSPMD_FORCE_INLINE void check_masks();
#else
	CPPSPMD_FORCE_INLINE void check_masks() { }
#endif

	CPPSPMD_FORCE_INLINE void spmd_break();
	CPPSPMD_FORCE_INLINE void spmd_continue();
	CPPSPMD_FORCE_INLINE void spmd_return();
		
	template<typename UnmaskedBody>
	CPPSPMD_FORCE_INLINE void spmd_unmasked(const UnmaskedBody& unmaskedBody);

	template<typename SPMDKernel, typename... Args>
	CPPSPMD_FORCE_INLINE decltype(auto) spmd_call(Args&&... args);

	CPPSPMD_FORCE_INLINE void swap(vint &a, vint &b) { vint temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vfloat &a, vfloat &b) { vfloat temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vbool &a, vbool &b) { vbool temp = a; store(a, b); store(b, temp); }

	CPPSPMD_FORCE_INLINE float reduce_add(vfloat v)	{ float4 k = blend_float4(vfloat(0.0f).m_value, v.m_value, m_exec.m_mask); return k.c[0] + k.c[1] + k.c[2] + k.c[3]; }

	#include "cppspmd_math_declares.h"

}; // struct spmd_kernel

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

// Bad pattern - doesn't factor in the current exec mask. Prefer spmd_any() instead.
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
CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) { return vint{ blendv_int4(b.m_value, a.m_value, cond.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ sqrt_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ abs_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ max_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ min_float4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ ceil_float4(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ floor_float4(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ round_float4(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ truncate_float4(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat frac(const vfloat& a) { return a - floor(a); }
CPPSPMD_FORCE_INLINE vfloat fmod(vfloat a, vfloat b) { vfloat c = frac(abs(a / b)) * abs(b); return spmd_ternaryf(a < 0, -c, c); }
CPPSPMD_FORCE_INLINE vfloat sign(const vfloat& a) { return spmd_ternaryf(a < 0.0f, 1.0f, 1.0f); }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b) { return vint{ max_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b) { return vint{ min_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint maxu(const vint& a, const vint& b) { return vint{ max_uint4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint minu(const vint& a, const vint& b) { return vint{ min_uint4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint cast_vfloat_to_vint(const vfloat& v) { return vint{ create_int4(CAST_F_TO_I(v.m_value.c[0]), CAST_F_TO_I(v.m_value.c[1]), CAST_F_TO_I(v.m_value.c[2]), CAST_F_TO_I(v.m_value.c[3])) }; }
CPPSPMD_FORCE_INLINE vfloat cast_vint_to_vfloat(const vint & v) { return vfloat{ create_float4(CAST_I_TO_F(v.m_value.c[0]), CAST_I_TO_F(v.m_value.c[1]), CAST_I_TO_F(v.m_value.c[2]), CAST_I_TO_F(v.m_value.c[3])) }; }

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

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the floats in each 128-bit lane.
#define VFLOAT_LANE_SHUFFLE_PS(a, control) vfloat(lane_shuffle_ps_float4((a).m_value, control))

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { return lint{ add_int4(set1_int4(a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { return lint{ add_int4(a.m_value, set1_int4(b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ and_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint andnot(const vint& a, const vint& b) { return vint{ andnot_int4(a.m_value, b.m_value) }; }
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

CPPSPMD_FORCE_INLINE vint mulhiu(const vint& a, const vint& b) { return vint{ mulhiu_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ neg_int4(v.m_value) }; }

CPPSPMD_FORCE_INLINE vint operator~(const vint& a) { return vint{ inv_int4(a.m_value) }; }

// A few of these break the lane-based abstraction model. They are supported in SSE2, so it makes sense to support them and let the user figure it out.
CPPSPMD_FORCE_INLINE vint adds_epu8(const vint& a, const vint& b) { return vint{ adds_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epu8(const vint& a, const vint& b) { return vint{ subs_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint avg_epu8(const vint& a, const vint& b) { return vint{ avg_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint max_epu8(const vint& a, const vint& b) { return vint{ max_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min_epu8(const vint& a, const vint& b) { return vint{ min_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sad_epu8(const vint& a, const vint& b) { return vint{ sad_epu8(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint add_epi8(const vint& a, const vint& b) { return vint{ add_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epi8(const vint& a, const vint& b) { return vint{ adds_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sub_epi8(const vint& a, const vint& b) { return vint{ sub_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epi8(const vint& a, const vint& b) { return vint{ subs_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi8(const vint& a, const vint& b) { return vint{ cmpeq_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi8(const vint& a, const vint& b) { return vint{ cmpgt_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi8(const vint& a, const vint& b) { return vint{ cmplt_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint unpacklo_epi8(const vint& a, const vint& b) { return vint{ unpacklo_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint unpackhi_epi8(const vint& a, const vint& b) { return vint{ unpackhi_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE int movemask_epi8(const vint& a) { return movemask_epi8(a.m_value); }
CPPSPMD_FORCE_INLINE int movemask_epi32(const vint& a) { return movemask_int4(a.m_value); }

CPPSPMD_FORCE_INLINE vint cmple_epu8(const vint& a, const vint& b) { return vint{ cmpeq_epi8(min_epu8(a.m_value, b.m_value), a.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpge_epu8(const vint& a, const vint& b) { return vint{ cmple_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epu8(const vint& a, const vint& b) { return vint{ andnot_int4(cmpeq_epi8(a.m_value, b.m_value), cmp_eq_int4(max_epu8(a.m_value, b.m_value), a.m_value)) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epu8(const vint& a, const vint& b) { return vint{ cmpgt_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint absdiff_epu8(const vint& a, const vint& b) { return vint{ or_int4(subs_epu8(a.m_value, b.m_value), subs_epu8(b.m_value, a.m_value)) }; }

CPPSPMD_FORCE_INLINE vint blendv_epi8(const vint& a, const vint& b, const vint &mask) { return vint{ blendv_epi8_int4(a.m_value, b.m_value, mask.m_value) }; }
CPPSPMD_FORCE_INLINE vint blendv_epi32(const vint& a, const vint& b, const vint &mask) { return vint{ blendv_int4(a.m_value, b.m_value, mask.m_value) }; }

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int32's in each 128-bit lane.
#define VINT_LANE_SHUFFLE_EPI32(a, control) vint(lane_shuffle_epi32_int4(a.m_value, control))

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int16's in either the high or low 64-bit lane.
#define VINT_LANE_SHUFFLELO_EPI16(a, control) vint(lane_shufflelo_epi16_int4(a.m_value, control))
#define VINT_LANE_SHUFFLEHI_EPI16(a, control) vint(lane_shufflehi_epi16_int4(a.m_value, control))

#define VINT_LANE_SHUFFLE_MASK(a, b, c, d) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))
#define VINT_LANE_SHUFFLE_MASK_R(d, c, b, a) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))

// _mm_srli_si128/_mm_slli_si128
#define VINT_LANE_SHIFT_LEFT_BYTES(a, l) vint(lane_shift_left_bytes(a.m_value, l))
#define VINT_LANE_SHIFT_RIGHT_BYTES(a, l) vint(lane_shift_right_bytes(a.m_value, l))

// Unpack and interleave 8-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi8(const vint& a, const vint& b) { return vint(unpacklo_epi8_int4(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi8(const vint& a, const vint& b) { return vint(unpackhi_epi8_int4(a.m_value, b.m_value)); }

// Unpack and interleave 16-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi16(const vint& a, const vint& b) { return vint(unpacklo_epi16_int4(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi16(const vint& a, const vint& b) { return vint(unpackhi_epi16_int4(a.m_value, b.m_value)); }

// Unpack and interleave 32-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi32(const vint& a, const vint& b) { return vint(unpacklo_epi32_int4(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi32(const vint& a, const vint& b) { return vint(unpackhi_epi32_int4(a.m_value, b.m_value)); }

// Unpack and interleave 64-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi64(const vint& a, const vint& b) { return vint(unpacklo_epi64_int4(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi64(const vint& a, const vint& b) { return vint(unpackhi_epi64_int4(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint vint_set1_epi8(int8_t a) { int v = (uint8_t)a | ((uint8_t)a << 8) | ((uint8_t)a << 16) | ((uint8_t)a << 24); return vint(int4(v, v, v, v)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi16(int16_t a) { int v = (uint16_t)a | ((uint16_t)a << 16); return vint(int4(v, v, v, v)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi32(int32_t a) { return vint(int4(a, a, a, a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi64(int64_t a) { return vint(int4((int32_t)a, (int32_t)(a >> 32), (int32_t)a, (int32_t)(a >> 32))); }

CPPSPMD_FORCE_INLINE vint add_epi16(const vint& a, const vint& b) { return vint{ add_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epi16(const vint& a, const vint& b) { return vint{ adds_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epu16(const vint& a, const vint& b) { return vint{ adds_epu16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint avg_epu16(const vint& a, const vint& b) { return vint{ avg_epu16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sub_epi16(const vint& a, const vint& b) { return vint{ sub_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epi16(const vint& a, const vint& b) { return vint{ subs_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epu16(const vint& a, const vint& b) { return vint{ subs_epu16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mullo_epi16(const vint& a, const vint& b) { return vint{ mullo_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epi16(const vint& a, const vint& b) { return vint{ mulhi_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epu16(const vint& a, const vint& b) { return vint{ mulhi_epu16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min_epi16(const vint& a, const vint& b) { return vint{ min_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint max_epi16(const vint& a, const vint& b) { return vint{ max_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint madd_epi16(const vint& a, const vint& b) { return vint{ madd_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi16(const vint& a, const vint& b) { return vint{ cmpeq_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi16(const vint& a, const vint& b) { return vint{ cmpgt_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi16(const vint& a, const vint& b) { return vint{ cmplt_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint packs_epi16(const vint& a, const vint& b) { return vint{ packs_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint packus_epi16(const vint& a, const vint& b) { return vint{ packus_epi16_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint uniform_shift_left_epi16(const vint& a, const vint& b) { return vint{ sll_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint uniform_arith_shift_right_epi16(const vint& a, const vint& b) { return vint{ sra_epi16_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint uniform_shift_right_epi16(const vint& a, const vint& b) { return vint{ srl_epi16_int4(a.m_value, b.m_value) }; }

#define VINT_SHIFT_LEFT_EPI16(a, b) vint(slli_epi16_int4((a).m_value, b))
#define VINT_SHIFT_RIGHT_EPI16(a, b) vint(srai_epi16_int4((a).m_value, b))
#define VUINT_SHIFT_RIGHT_EPI16(a, b) vint(srli_epi16_int4((a).m_value, b))

CPPSPMD_FORCE_INLINE vint undefined_vint() { return vint{ set1_int4(0) }; }
CPPSPMD_FORCE_INLINE vfloat undefined_vfloat() { return vfloat{ set1_float4(0) }; }

CPPSPMD_FORCE_INLINE vint abs(const vint& v) { return vint{ abs_int4(v.m_value) }; }

CPPSPMD_FORCE_INLINE vint mul_epu32(const vint &a, const vint& b) { return vint(mul_epu32_int4(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint div_epi32(const vint &a, const vint& b) { return vint(div_int4(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint mod_epi32(const vint &a, const vint& b)
{
	vint aa = abs(a), ab = abs(b);
	vint q = div_epi32(aa, ab);
	vint r = aa - q * ab;
	return spmd_ternaryi(a < 0, -r, r);
}

// Div/mod both suppress divide by 0
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, const vint& b) { return vint{ div_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, int b) { return vint{ div_int4(a.m_value, b) }; }
//CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b) { return vint{ mod_int4(a.m_value, b.m_value) }; }
//CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b) { return vint{ mod_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b) { return mod_epi32(a, b); }
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b) { return mod_epi32(a, vint(b)); }

CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b) { return vint{ shift_left_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b) { return vint{ shift_left_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b) { return vint{ shift_right_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b) { return vint{ unsigned_shift_right_int4(a.m_value, b) }; }
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b) { return vint{ shift_right_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b) { return vint{ unsigned_shift_right_int4(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint vuint_shift_right_not_zero(const vint& a, const vint& b) { return vint{ unsigned_shift_right_int4(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint byteswap(const vint& v) { return vint{ byteswap_int4(v.m_value) }; }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) vint( shift_left_int4((a).m_value, (b)) )
#define VINT_SHIFT_RIGHT(a, b) vint( shift_right_int4((a).m_value, (b)) )
#define VUINT_SHIFT_RIGHT(a, b) vint( unsigned_shift_right_int4((a).m_value, (b)) )
#define VINT_ROT(x, k) (VINT_SHIFT_LEFT((x), (k)) | VUINT_SHIFT_RIGHT((x), 32 - (k)))

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
#define VFLOAT_EXTRACT(v, instance) ((v).m_value.c[instance])

CPPSPMD_FORCE_INLINE vfloat &insert(vfloat& v, int instance, float f)
{
	assert(instance < 4);
	v.m_value.c[instance] = f;
	return v;
}

CPPSPMD_FORCE_INLINE vint &insert(vint& v, int instance, int i)
{
	assert(instance < 4);
	v.m_value.c[instance] = i;
	return v;
}

CPPSPMD_FORCE_INLINE vint init_lookup4(const uint8_t pTab[16])
{
	return vint{ load_int4((const int*)pTab) };
}

CPPSPMD_FORCE_INLINE vint table_lookup4_8(const vint& a, const vint& table)
{
	return vint{ shuffle_epi8_int4(table.m_value, a.m_value) };
}

CPPSPMD_FORCE_INLINE void init_lookup5(const uint8_t pTab[32], vint& table_0, vint& table_1)
{
	int4 l = load_int4((const int*)pTab);
	int4 h = load_int4((const int*)(pTab + 16));
	table_0.m_value = l;
	table_1.m_value = h;
}

CPPSPMD_FORCE_INLINE vint table_lookup5_8(const vint& a, const vint& table_0, const vint& table_1)
{
	int4 l_0 = shuffle_epi8_int4(table_0.m_value, a.m_value);
	int4 h_0 = shuffle_epi8_int4(table_1.m_value, a.m_value);

	int4 m_0 = shift_left_int4(a.m_value, 31 - 4);

	int4 v_0 = blendv_int4(l_0, h_0, m_0);

	return vint{ v_0 };
}

CPPSPMD_FORCE_INLINE void init_lookup6(const uint8_t pTab[64], vint& table_0, vint& table_1, vint& table_2, vint& table_3)
{
	int4 a = load_int4((const int*)pTab);
	int4 b = load_int4((const int*)(pTab + 16));
	int4 c = load_int4((const int*)(pTab + 32));
	int4 d = load_int4((const int*)(pTab + 48));

	table_0.m_value = a;
	table_1.m_value = b;
	table_2.m_value = c;
	table_3.m_value = d;
}

CPPSPMD_FORCE_INLINE vint table_lookup6_8(const vint& a, const vint& table_0, const vint& table_1, const vint& table_2, const vint& table_3)
{
	int4 m_0 = shift_left_int4(a.m_value, 31 - 4);

	int4 av_0;
	{
		int4 al_0 = shuffle_epi8_int4(table_0.m_value, a.m_value);
		int4 ah_0 = shuffle_epi8_int4(table_1.m_value, a.m_value);
		av_0 = blendv_int4(al_0, ah_0, m_0);
	}

	int4 bv_0;
	{
		int4 bl_0 = shuffle_epi8_int4(table_2.m_value, a.m_value);
		int4 bh_0 = shuffle_epi8_int4(table_3.m_value, a.m_value);
		bv_0 = blendv_int4(bl_0, bh_0, m_0);
	}

	int4 m2_0 = shift_left_int4(a.m_value, 31 - 5);
	int4 v2_0 = blendv_int4(av_0, bv_0, m2_0);

	return vint{ v2_0 };
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

#include "cppspmd_flow.h"
#include "cppspmd_math.h"

} // namespace cppspmd_float4

