// cppspmd_int16_avx2_fma.h
// The module is intended for AVX2, but it also supports AVX1. Also supports optional FMA support.
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
#include <immintrin.h>
#include <algorithm>

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
#ifdef _DEBUG
#define CPPSPMD_FORCE_INLINE inline
#else
#define CPPSPMD_FORCE_INLINE __forceinline
#endif
#endif

#undef CPPSPMD
#undef CPPSPMD_ARCH

#undef CPPSPMD_SSE
#undef CPPSPMD_AVX1
#undef CPPSPMD_AVX2
#undef CPPSPMD_AVX
#undef CPPSPMD_FLOAT4
#undef CPPSPMD_INT16

#define CPPSPMD_SSE 0
#define CPPSPMD_AVX 1
#define CPPSPMD_FLOAT4 0

#define CPPSPMD cppspmd_int16_avx2_fma
#define CPPSPMD_ARCH _int16_avx2_fma
#define CPPSPMD_AVX1 0
#define CPPSPMD_AVX2 1
#define CPPSPMD_INT16 1

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

#define CPPSPMD_ALIGNMENT (32)

namespace CPPSPMD
{

const int PROGRAM_COUNT_SHIFT = 4;
const int PROGRAM_COUNT = 1 << PROGRAM_COUNT_SHIFT;

template <typename N> inline N* aligned_new() { void* p = _mm_malloc(sizeof(N), 64); new (p) N;	return static_cast<N*>(p); }
template <typename N> void aligned_delete(N* p) { if (p) { p->~N(); _mm_free(p); } }

CPPSPMD_DECL(const uint32_t, g_allones_256[8]) = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
CPPSPMD_DECL(const float, g_onef_256[8]) = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(const uint32_t, g_oneu_256[8]) = { 1, 1, 1, 1, 1, 1, 1, 1 };
CPPSPMD_DECL(const uint32_t, g_x_128[4]) = { UINT32_MAX, 0, 0, 0 };

CPPSPMD_DECL(const uint32_t, g_lane_masks_256[8][8]) = 
{ 
	{ UINT32_MAX, 0, 0, 0,  0, 0, 0, 0 },
	{ 0, UINT32_MAX, 0, 0,  0, 0, 0, 0 },
	{ 0, 0, UINT32_MAX, 0,  0, 0, 0, 0 },
	{ 0, 0, 0, UINT32_MAX,  0, 0, 0, 0 },
	{ 0, 0, 0, 0,			UINT32_MAX, 0, 0, 0 },
	{ 0, 0, 0, 0,			0, UINT32_MAX, 0, 0 },
	{ 0, 0, 0, 0,			0, 0, UINT32_MAX, 0 },
	{ 0, 0, 0, 0,			0, 0, 0, UINT32_MAX },
};

CPPSPMD_FORCE_INLINE __m128 get_lo(__m256i v) { return _mm256_castps256_ps128(_mm256_castsi256_ps(v)); }
CPPSPMD_FORCE_INLINE __m128 get_lo(__m256 v) { return _mm256_castps256_ps128(v); }

CPPSPMD_FORCE_INLINE __m128i get_lo_i(__m256i v) { return _mm_castps_si128(_mm256_castps256_ps128(_mm256_castsi256_ps(v))); }
CPPSPMD_FORCE_INLINE __m128i get_lo_i(__m256 v) { return _mm_castps_si128(_mm256_castps256_ps128(v)); }

CPPSPMD_FORCE_INLINE __m128 get_hi(__m256i v) { return _mm256_extractf128_ps(_mm256_castsi256_ps(v), 1); }
CPPSPMD_FORCE_INLINE __m128 get_hi(__m256 v) { return _mm256_extractf128_ps(v, 1); }

CPPSPMD_FORCE_INLINE __m128i get_hi_i(__m256i v) { return _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(v), 1)); }
CPPSPMD_FORCE_INLINE __m128i get_hi_i(__m256 v) { return _mm_castps_si128(_mm256_extractf128_ps(v, 1)); }

CPPSPMD_FORCE_INLINE __m256i combine_i(__m128 lo, __m128 hi) { return _mm256_castps_si256(_mm256_setr_m128(lo, hi)); }
CPPSPMD_FORCE_INLINE __m256i combine_i(__m128i lo, __m128i hi) { return _mm256_setr_m128i(lo, hi); }
CPPSPMD_FORCE_INLINE __m256 combine(__m128 lo, __m128 hi) { return _mm256_setr_m128(lo, hi); }
CPPSPMD_FORCE_INLINE __m256 combine(__m128i lo, __m128i hi) { return _mm256_castsi256_ps(_mm256_setr_m128i(lo, hi)); }

CPPSPMD_FORCE_INLINE __m256i and_si256(__m256i a, __m256i b) {	return _mm256_and_si256(a, b); }
CPPSPMD_FORCE_INLINE __m256i or_si256(__m256i a, __m256i b) { return _mm256_or_si256(a, b); }
CPPSPMD_FORCE_INLINE __m256i xor_si256(__m256i a, __m256i b) {	return _mm256_xor_si256(a, b); }
CPPSPMD_FORCE_INLINE __m256i andnot_si256(__m256i a, __m256i b) {	return _mm256_andnot_si256(a, b); }

CPPSPMD_FORCE_INLINE __m256i _mm256_cmple_epu16(__m256i x, __m256i y) { return _mm256_cmpeq_epi16(_mm256_subs_epu16(x, y), _mm256_setzero_si256()); }
CPPSPMD_FORCE_INLINE __m256i _mm256_cmpge_epu16(__m256i x, __m256i y) { return _mm256_cmple_epu16(y, x); }
CPPSPMD_FORCE_INLINE __m256i _mm256_cmpgt_epu16(__m256i x, __m256i y) { return _mm256_andnot_si256(_mm256_cmpeq_epi16(x, y), _mm256_cmple_epu16(y, x)); }
CPPSPMD_FORCE_INLINE __m256i _mm256_cmplt_epu16(__m256i x, __m256i y) { return _mm256_cmpgt_epu16(y, x); }
CPPSPMD_FORCE_INLINE __m256i _mm256_cmpge_epi16(__m256i x, __m256i y) { return _mm256_or_si256(_mm256_cmpeq_epi16(x, y), _mm256_cmpgt_epi16(x, y)); }

// Divide 8 16-bit uints by 255:
// x := ((x + 128) + (x >> 8)) >> 8:
CPPSPMD_FORCE_INLINE __m256i _mm256_div255_epu16(__m256i x)
{
	return _mm256_srli_epi16(_mm256_adds_epu16(_mm256_adds_epu16(x, _mm256_set1_epi16(128)), _mm256_srli_epi16(x, 8)), 8);
}

// Calculate absolute difference: abs(x - y):
CPPSPMD_FORCE_INLINE __m256i _mm256_absdiff_epu16(__m256i x, __m256i y)
{
	__m256i a = _mm256_subs_epu16(x, y);
	__m256i b = _mm256_subs_epu16(y, x);
	return _mm256_or_si256(a, b);
}

// Duplicates bytes
CPPSPMD_FORCE_INLINE void convert_16_to_32_alt(__m256i maski, __m256i &ai, __m256i &bi)
{
	__m256i l = _mm256_unpacklo_epi16(maski, maski);
	__m256i h = _mm256_unpackhi_epi16(maski, maski);

	ai = _mm256_permute2x128_si256(l, h, 0 | (2 << 4));
	bi = _mm256_permute2x128_si256(l, h, 1 | (3 << 4));
}

// With sign extension
CPPSPMD_FORCE_INLINE void convert_16_to_32(__m256i maski, __m256i &ai, __m256i &bi)
{
	ai = _mm256_cvtepi16_epi32(get_lo_i(maski));
	bi = _mm256_cvtepi16_epi32(get_hi_i(maski));
}

CPPSPMD_DECL(const uint8_t, g_convert_32_to_16_control[32]) =
{
	0x00, 0x01, 0x04, 0x05,  0x08, 0x09, 0x0C, 0x0D,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x00, 0x01, 0x04, 0x05,  0x08, 0x09, 0x0C, 0x0D
};

CPPSPMD_FORCE_INLINE __m256i convert_32_to_16(__m256i ai, __m256i bi)
{
	__m256i c = _mm256_load_si256((const __m256i *)g_convert_32_to_16_control);
	__m256i ai_s = _mm256_shuffle_epi8(ai, c);
	__m256i bi_s = _mm256_shuffle_epi8(bi, c);

#if 0
	__m128i l = _mm_or_si128(get_lo_i(ai_s), get_hi_i(ai_s));
	__m128i h = _mm_or_si128(get_lo_i(bi_s), get_hi_i(bi_s));

	return combine_i(l, h);
#else
	__m256i k0 = _mm256_inserti128_si256(ai_s, _mm256_castsi256_si128(bi_s), 1);
	__m256i k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(get_hi_i(ai_s)), get_hi_i(bi_s), 1);

	return _mm256_or_si256(k0, k1);
#endif
}

CPPSPMD_DECL(const uint8_t, g_convert_hi_32_to_16_control[32]) =
{
	0x02, 0x03, 0x06, 0x07,  0x0A, 0x0B, 0x0E, 0x0F,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x02, 0x03, 0x06, 0x07,  0x0A, 0x0B, 0x0E, 0x0F
};

CPPSPMD_FORCE_INLINE __m256i convert_hi_32_to_16(__m256i ai, __m256i bi)
{
	__m256i c = _mm256_load_si256((const __m256i *)g_convert_hi_32_to_16_control);
	__m256i ai_s = _mm256_shuffle_epi8(ai, c);
	__m256i bi_s = _mm256_shuffle_epi8(bi, c);

#if 0
	__m128i l = _mm_or_si128(get_lo_i(ai_s), get_hi_i(ai_s));
	__m128i h = _mm_or_si128(get_lo_i(bi_s), get_hi_i(bi_s));

	return combine_i(l, h);
#else
	__m256i k0 = _mm256_inserti128_si256(ai_s, _mm256_castsi256_si128(bi_s), 1);
	__m256i k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(get_hi_i(ai_s)), get_hi_i(bi_s), 1);

	return _mm256_or_si256(k0, k1);
#endif
}

CPPSPMD_DECL(const uint8_t, g_convert_r_32_to_16_control[32]) =
{
	0x00, 0x80, 0x04, 0x80,  0x08, 0x80, 0x0C, 0x80,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x00, 0x80, 0x04, 0x80,  0x08, 0x80, 0x0C, 0x80
};

CPPSPMD_FORCE_INLINE __m256i convert_r_32_to_16(__m256i ai, __m256i bi)
{
	__m256i c = _mm256_load_si256((const __m256i *)g_convert_r_32_to_16_control);
	__m256i ai_s = _mm256_shuffle_epi8(ai, c);
	__m256i bi_s = _mm256_shuffle_epi8(bi, c);
	__m256i k0 = _mm256_inserti128_si256(ai_s, _mm256_castsi256_si128(bi_s), 1);
	__m256i k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(get_hi_i(ai_s)), get_hi_i(bi_s), 1);
	return _mm256_or_si256(k0, k1);
}

CPPSPMD_DECL(const uint8_t, g_convert_g_32_to_16_control[32]) =
{
	0x01, 0x80, 0x05, 0x80,  0x09, 0x80, 0x0D, 0x80,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x01, 0x80, 0x05, 0x80,  0x09, 0x80, 0x0D, 0x80
};

CPPSPMD_FORCE_INLINE __m256i convert_g_32_to_16(__m256i ai, __m256i bi)
{
	__m256i c = _mm256_load_si256((const __m256i *)g_convert_g_32_to_16_control);
	__m256i ai_s = _mm256_shuffle_epi8(ai, c);
	__m256i bi_s = _mm256_shuffle_epi8(bi, c);
	__m256i k0 = _mm256_inserti128_si256(ai_s, _mm256_castsi256_si128(bi_s), 1);
	__m256i k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(get_hi_i(ai_s)), get_hi_i(bi_s), 1);
	return _mm256_or_si256(k0, k1);
}

CPPSPMD_DECL(const uint8_t, g_convert_b_32_to_16_control[32]) =
{
	0x02, 0x80, 0x06, 0x80,  0x0A, 0x80, 0x0E, 0x80,  0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80,  0x80, 0x80, 0x80, 0x80,  0x02, 0x80, 0x06, 0x80,  0x0A, 0x80, 0x0E, 0x80
};

CPPSPMD_FORCE_INLINE __m256i convert_b_32_to_16(__m256i ai, __m256i bi)
{
	__m256i c = _mm256_load_si256((const __m256i *)g_convert_b_32_to_16_control);
	__m256i ai_s = _mm256_shuffle_epi8(ai, c);
	__m256i bi_s = _mm256_shuffle_epi8(bi, c);
	__m256i k0 = _mm256_inserti128_si256(ai_s, _mm256_castsi256_si128(bi_s), 1);
	__m256i k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(get_hi_i(ai_s)), get_hi_i(bi_s), 1);
	return _mm256_or_si256(k0, k1);
}

CPPSPMD_FORCE_INLINE __m128i _mm_blendv_epi32(__m128i a, __m128i b, __m128i c) { return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _mm_castsi128_ps(c))); }

CPPSPMD_FORCE_INLINE __m256i blendv_epi32(__m256i a, __m256i b, __m256i c) 
{ 
#if CPPSPMD_USE_AVX2
	return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), _mm256_castsi256_ps(c))); 
#else
	return combine_i(_mm_blendv_epi32(get_lo_i(a), get_lo_i(b), get_lo_i(c)), _mm_blendv_epi32(get_hi_i(a), get_hi_i(b), get_hi_i(c)));
#endif
}

// Movemasks are 2-bits per lane due to how _mm256_movemask_epi8() works on bytes, not words.
const uint32_t ALL_ON_MOVEMASK = 0xFFFFFFFFU;

struct spmd_kernel
{
	struct vint16;
	struct lint16;
	struct vbool;
	struct vfloat;

	typedef int16_t int_t;
	typedef vint16 vint_t;
	typedef lint16 lint_t;

	// Exec mask
	struct exec_mask
	{
		__m256i m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const __m256i& mask) : m_mask(mask) { }

		CPPSPMD_FORCE_INLINE void enable_lane(uint32_t lane) { m_mask = _mm256_load_si256((const __m256i *)&g_lane_masks_256[lane][0]); }

		static CPPSPMD_FORCE_INLINE exec_mask all_on() { return exec_mask{ _mm256_load_si256((const __m256i*)g_allones_256) }; }
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ _mm256_setzero_si256() }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return _mm256_movemask_epi8(m_mask); }
	};

	friend CPPSPMD_FORCE_INLINE bool all(const exec_mask& e);
	friend CPPSPMD_FORCE_INLINE bool any(const exec_mask& e);

	CPPSPMD_FORCE_INLINE bool spmd_all() const { return all(m_exec); }
	CPPSPMD_FORCE_INLINE bool spmd_any() const { return any(m_exec); }
	CPPSPMD_FORCE_INLINE bool spmd_none() { return !any(m_exec); }

	// true if cond is true for all active lanes - false if no active lanes
	CPPSPMD_FORCE_INLINE bool spmd_all(const vbool& e) { uint32_t m = m_exec.get_movemask(); return (m != 0) && ((exec_mask(e) & m_exec).get_movemask() == m); }
	// true if cond is true for any active lanes
	CPPSPMD_FORCE_INLINE bool spmd_any(const vbool& e) { return (exec_mask(e) & m_exec).get_movemask() != 0; }
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
		__m256i m_value;

		vbool() = default;

		CPPSPMD_FORCE_INLINE vbool(bool value) : m_value(_mm256_set1_epi32(value ? UINT32_MAX : 0)) { }

		CPPSPMD_FORCE_INLINE explicit vbool(const __m256i& value) : m_value(value) { }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const;
		CPPSPMD_FORCE_INLINE explicit operator vint16() const;

	private:
		vbool & operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = _mm256_blendv_epi8(dst.m_value, src.m_value, m_exec.m_mask);
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
		__m256 m_value_l;
		__m256 m_value_h;

		vfloat() = default;

		CPPSPMD_FORCE_INLINE explicit vfloat(const __m256& l, const __m256& h) : m_value_l(l), m_value_h(h) { }

		CPPSPMD_FORCE_INLINE vfloat(float value) : m_value_l(_mm256_set1_ps(value)), m_value_h(_mm256_set1_ps(value)) { }

		CPPSPMD_FORCE_INLINE explicit vfloat(int value) : m_value_l(_mm256_set1_ps((float)value)), m_value_h(_mm256_set1_ps((float)value)) { }
				
	private:
		vfloat & operator=(const vfloat&);
	};

	CPPSPMD_FORCE_INLINE void get_movemask_32_i(__m256i &l, __m256i &h)
	{
		convert_16_to_32(m_exec.m_mask, l, h);
	}

	CPPSPMD_FORCE_INLINE void get_movemask_32(__m256 &l, __m256 &h)
	{
		__m256i li, hi;
		convert_16_to_32(m_exec.m_mask, li, hi);
		l = _mm256_castsi256_ps(li);
		h = _mm256_castsi256_ps(hi);
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat& dst, const vfloat& src)
	{
		__m256 mask_l, mask_h; 
		get_movemask_32(mask_l, mask_h);

		dst.m_value_l = _mm256_blendv_ps(dst.m_value_l, src.m_value_l, mask_l);
		dst.m_value_h = _mm256_blendv_ps(dst.m_value_h, src.m_value_h, mask_h);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat&& dst, const vfloat& src)
	{
		__m256 mask_l, mask_h;
		get_movemask_32(mask_l, mask_h);

		dst.m_value_l = _mm256_blendv_ps(dst.m_value_l, src.m_value_l, mask_l);
		dst.m_value_h = _mm256_blendv_ps(dst.m_value_h, src.m_value_h, mask_h);

		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vfloat& store_all(vfloat& dst, const vfloat& src)
	{
		dst.m_value_l = src.m_value_l;
		dst.m_value_h = src.m_value_h;
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store_all(vfloat&& dst, const vfloat& src)
	{
		dst.m_value_l = src.m_value_l;
		dst.m_value_h = src.m_value_h;
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
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_ps(dst.m_pValue, src.m_value_l);
			_mm256_storeu_ps(dst.m_pValue + 8, src.m_value_h);
		}
		else
		{
			__m256i mask_l, mask_h;
			get_movemask_32_i(mask_l, mask_h);

			_mm256_maskstore_ps(dst.m_pValue, mask_l, src.m_value_l);
			_mm256_maskstore_ps(dst.m_pValue + 8, mask_h, src.m_value_h);
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref&& dst, const vfloat& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_ps(dst.m_pValue, src.m_value_l);
			_mm256_storeu_ps(dst.m_pValue + 8, src.m_value_h);
		}
		else
		{
			__m256i mask_l, mask_h;
			get_movemask_32_i(mask_l, mask_h);

			_mm256_maskstore_ps(dst.m_pValue, mask_l, src.m_value_l);
			_mm256_maskstore_ps(dst.m_pValue + 8, mask_h, src.m_value_h);
		}
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref& dst, const vfloat& src)
	{
		_mm256_storeu_ps(dst.m_pValue, src.m_value_l);
		_mm256_storeu_ps(dst.m_pValue + 8, src.m_value_h);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref&& dst, const vfloat& src)
	{
		_mm256_storeu_ps(dst.m_pValue, src.m_value_l);
		_mm256_storeu_ps(dst.m_pValue + 8, src.m_value_h);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const float_lref& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			return vfloat{ _mm256_loadu_ps(src.m_pValue), _mm256_loadu_ps(src.m_pValue + 8) };
		}
		else
		{
			__m256i mask_l, mask_h;
			get_movemask_32_i(mask_l, mask_h);

			return vfloat{ _mm256_maskload_ps(src.m_pValue, mask_l), _mm256_maskload_ps(src.m_pValue + 8, mask_h) };
		}
	}
		
	// Varying ref to floats
	struct float_vref
	{
		__m256i m_vindex;
		float* m_pValue;
		
	private:
		float_vref& operator=(const float_vref&);
	};

	// Varying ref to varying float
	struct vfloat_vref
	{
		__m256i m_vindex;
		vfloat* m_pValue;
		
	private:
		vfloat_vref& operator=(const vfloat_vref&);
	};

	// Varying ref to varying int
	struct vint16_vref
	{
		__m256i m_vindex;
		vint16* m_pValue;
		
	private:
		vint16_vref& operator=(const vint16_vref&);
	};

	CPPSPMD_FORCE_INLINE const float_vref& store(const float_vref& dst, const vfloat& src);
	CPPSPMD_FORCE_INLINE const float_vref& store(const float_vref&& dst, const vfloat& src);
		
	CPPSPMD_FORCE_INLINE const float_vref& store_all(const float_vref& dst, const vfloat& src);
	CPPSPMD_FORCE_INLINE const float_vref& store_all(const float_vref&& dst, const vfloat& src);

	CPPSPMD_FORCE_INLINE vfloat load(const float_vref& src)
	{
		__m256i index_l, index_h;
		convert_16_to_32(src.m_vindex, index_l, index_h);
		
		__m256i mask_l, mask_h;
		convert_16_to_32(m_exec.m_mask, mask_l, mask_h);

		return vfloat{ _mm256_mask_i32gather_ps(_mm256_undefined_ps(), src.m_pValue, index_l, _mm256_castsi256_ps(mask_l), 4), 
							_mm256_mask_i32gather_ps(_mm256_undefined_ps(), src.m_pValue, index_h, _mm256_castsi256_ps(mask_h), 4) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
		__m256i index_l, index_h;
		convert_16_to_32(src.m_vindex, index_l, index_h);

		return vfloat{ _mm256_mask_i32gather_ps(_mm256_undefined_ps(), src.m_pValue, index_l, _mm256_load_ps((const float *)g_allones_256), 4),
							_mm256_mask_i32gather_ps(_mm256_undefined_ps(), src.m_pValue, index_h, _mm256_load_ps((const float *)g_allones_256), 4) };
	}

	// Linear ref to int16's
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};
		
	CPPSPMD_FORCE_INLINE const int16_lref& store(const int16_lref& dst, const vint16& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_si256((__m256i*)dst.m_pValue, src.m_value);
		}
		else
		{
			if (mask & (1 << 0)) dst.m_pValue[0] = (int16_t)_mm256_extract_epi16(src.m_value, 0);
			if (mask & (1 << 2)) dst.m_pValue[1] = (int16_t)_mm256_extract_epi16(src.m_value, 1);
			if (mask & (1 << 4)) dst.m_pValue[2] = (int16_t)_mm256_extract_epi16(src.m_value, 2);
			if (mask & (1 << 6)) dst.m_pValue[3] = (int16_t)_mm256_extract_epi16(src.m_value, 3);

			if (mask & (1 << 8)) dst.m_pValue[4] = (int16_t)_mm256_extract_epi16(src.m_value, 4);
			if (mask & (1 << 10)) dst.m_pValue[5] = (int16_t)_mm256_extract_epi16(src.m_value, 5);
			if (mask & (1 << 12)) dst.m_pValue[6] = (int16_t)_mm256_extract_epi16(src.m_value, 6);
			if (mask & (1 << 14)) dst.m_pValue[7] = (int16_t)_mm256_extract_epi16(src.m_value, 7);

			if (mask & (1 << 16)) dst.m_pValue[8] = (int16_t)_mm256_extract_epi16(src.m_value, 8);
			if (mask & (1 << 18)) dst.m_pValue[9] = (int16_t)_mm256_extract_epi16(src.m_value, 9);
			if (mask & (1 << 20)) dst.m_pValue[10] = (int16_t)_mm256_extract_epi16(src.m_value, 10);
			if (mask & (1 << 22)) dst.m_pValue[11] = (int16_t)_mm256_extract_epi16(src.m_value, 11);

			if (mask & (1 << 24)) dst.m_pValue[12] = (int16_t)_mm256_extract_epi16(src.m_value, 12);
			if (mask & (1 << 26)) dst.m_pValue[13] = (int16_t)_mm256_extract_epi16(src.m_value, 13);
			if (mask & (1 << 28)) dst.m_pValue[14] = (int16_t)_mm256_extract_epi16(src.m_value, 14);
			if (mask & (1 << 30)) dst.m_pValue[15] = (int16_t)_mm256_extract_epi16(src.m_value, 15);

		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_lref& store_all(const int16_lref& dst, const vint16& src)
	{
		_mm256_storeu_si256((__m256i*)dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint16 load(const int16_lref& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

		if (mask == ALL_ON_MOVEMASK)
			return vint16{ _mm256_loadu_si256((__m256i*)src.m_pValue) };
		else
		{
			__m256i v = _mm256_setzero_si256();
						
			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, src.m_pValue[0], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, src.m_pValue[1], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, src.m_pValue[2], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, src.m_pValue[3], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, src.m_pValue[4], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, src.m_pValue[5], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, src.m_pValue[6], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, src.m_pValue[7], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, src.m_pValue[8], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, src.m_pValue[9], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, src.m_pValue[10], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, src.m_pValue[11], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, src.m_pValue[12], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, src.m_pValue[13], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, src.m_pValue[14], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, src.m_pValue[15], 15);
			
			return vint16{ v };
		}
	}

	CPPSPMD_FORCE_INLINE vint16 load_all(const int16_lref& src)
	{
		return vint16{ _mm256_loadu_si256((__m256i*)src.m_pValue) };
	}

	// Linear ref to constant int16's
	struct cint16_lref
	{
		const int16_t* m_pValue;

	private:
		cint16_lref& operator=(const cint16_lref&);
	};

	CPPSPMD_FORCE_INLINE vint16 load(const cint16_lref& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

		if (mask == ALL_ON_MOVEMASK)
			return vint16{ _mm256_loadu_si256((__m256i*)src.m_pValue) };
		else
		{
			__m256i v = _mm256_setzero_si256();

			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, src.m_pValue[0], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, src.m_pValue[1], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, src.m_pValue[2], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, src.m_pValue[3], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, src.m_pValue[4], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, src.m_pValue[5], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, src.m_pValue[6], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, src.m_pValue[7], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, src.m_pValue[8], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, src.m_pValue[9], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, src.m_pValue[10], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, src.m_pValue[11], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, src.m_pValue[12], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, src.m_pValue[13], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, src.m_pValue[14], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, src.m_pValue[15], 15);

			return vint16{ v };
		}
	}

	CPPSPMD_FORCE_INLINE vint16 load_all(const cint16_lref& src)
	{
		return vint16{ _mm256_loadu_si256((__m256i*)src.m_pValue) };
	}
		
	// Varying ref to int16s
	struct int16_vref
	{
		__m256i m_vindex;
		int16_t* m_pValue;

	private:
		int16_vref& operator=(const int16_vref&);
	};

	// Varying ref to constant int16s
	struct cint16_vref
	{
		__m256i m_vindex;
		const int16_t* m_pValue;

	private:
		cint16_vref& operator=(const cint16_vref&);
	};

	// Varying int
	struct vint16
	{
		__m256i m_value;
		
		vint16() = default;

		CPPSPMD_FORCE_INLINE explicit vint16(const __m256i& value) : m_value(value)	{ }

		CPPSPMD_FORCE_INLINE vint16(int16_t value) : m_value(_mm256_set1_epi16(value)) { }
		
		CPPSPMD_FORCE_INLINE vint16(int value) : m_value(_mm256_set1_epi16((int16_t)value)) { }

		CPPSPMD_FORCE_INLINE explicit vint16(float value) : m_value(_mm256_set1_epi16((int16_t)value))	{ }

		CPPSPMD_FORCE_INLINE explicit vint16(const vfloat& other) :
			m_value(convert_32_to_16(_mm256_cvttps_epi32(other.m_value_l), _mm256_cvttps_epi32(other.m_value_h)))
		{
		}

		CPPSPMD_FORCE_INLINE explicit operator vbool() const 
		{
			return vbool{ xor_si256( _mm256_load_si256((const __m256i*)g_allones_256), _mm256_cmpeq_epi16(m_value, _mm256_setzero_si256())) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			__m256i l, h;
			convert_16_to_32(m_value, l, h);
			return vfloat{ _mm256_cvtepi32_ps(l), _mm256_cvtepi32_ps(h) };
		}

		CPPSPMD_FORCE_INLINE int16_vref operator[](int16_t* ptr) const
		{
			return int16_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE cint16_vref operator[](const int16_t* ptr) const
		{
			return cint16_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE float_vref operator[](float* ptr) const
		{
			return float_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE vfloat_vref operator[](vfloat* ptr) const
		{
			return vfloat_vref{ m_value, ptr };
		}

		CPPSPMD_FORCE_INLINE vint16_vref operator[](vint16* ptr) const
		{
			return vint16_vref{ m_value, ptr };
		}

	private:
		vint16& operator=(const vint16&);
	};

	// Load/store linear integer
	CPPSPMD_FORCE_INLINE void storeu_linear(int16_t *pDst, const vint16& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_si256((__m256i*)pDst, src.m_value);
		}
		else
		{
			if (mask & (1 << 0)) pDst[0] = (int16_t)_mm256_extract_epi16(src.m_value, 0);
			if (mask & (1 << 2)) pDst[1] = (int16_t)_mm256_extract_epi16(src.m_value, 1);
			if (mask & (1 << 4)) pDst[2] = (int16_t)_mm256_extract_epi16(src.m_value, 2);
			if (mask & (1 << 6)) pDst[3] = (int16_t)_mm256_extract_epi16(src.m_value, 3);

			if (mask & (1 << 8)) pDst[4] = (int16_t)_mm256_extract_epi16(src.m_value, 4);
			if (mask & (1 << 10)) pDst[5] = (int16_t)_mm256_extract_epi16(src.m_value, 5);
			if (mask & (1 << 12)) pDst[6] = (int16_t)_mm256_extract_epi16(src.m_value, 6);
			if (mask & (1 << 14)) pDst[7] = (int16_t)_mm256_extract_epi16(src.m_value, 7);

			if (mask & (1 << 16)) pDst[8] = (int16_t)_mm256_extract_epi16(src.m_value, 8);
			if (mask & (1 << 18)) pDst[9] = (int16_t)_mm256_extract_epi16(src.m_value, 9);
			if (mask & (1 << 20)) pDst[10] = (int16_t)_mm256_extract_epi16(src.m_value, 10);
			if (mask & (1 << 22)) pDst[11] = (int16_t)_mm256_extract_epi16(src.m_value, 11);

			if (mask & (1 << 24)) pDst[12] = (int16_t)_mm256_extract_epi16(src.m_value, 12);
			if (mask & (1 << 26)) pDst[13] = (int16_t)_mm256_extract_epi16(src.m_value, 13);
			if (mask & (1 << 28)) pDst[14] = (int16_t)_mm256_extract_epi16(src.m_value, 14);
			if (mask & (1 << 30)) pDst[15] = (int16_t)_mm256_extract_epi16(src.m_value, 15);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int16_t *pDst, const vint16& src)
	{
		_mm256_storeu_si256((__m256i*)pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int16_t *pDst, const vint16& src)
	{
		_mm256_store_si256((__m256i*)pDst, src.m_value);
	}
	
	CPPSPMD_FORCE_INLINE vint16 loadu_linear(const int16_t *pSrc)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		__m256i result;
		if (mask == ALL_ON_MOVEMASK)
			result = _mm256_loadu_si256((__m256i*)pSrc);
		else
		{
			__m256i v = _mm256_setzero_si256();

			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, pSrc[0], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, pSrc[1], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, pSrc[2], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, pSrc[3], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, pSrc[4], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, pSrc[5], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, pSrc[6], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, pSrc[7], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, pSrc[8], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, pSrc[9], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, pSrc[10], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, pSrc[11], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, pSrc[12], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, pSrc[13], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, pSrc[14], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, pSrc[15], 15);
			
			result = v;
		}
		return vint16{ result };
	}

	CPPSPMD_FORCE_INLINE vint16 loadu_linear_all(const int *pSrc)
	{
		return vint16{ _mm256_loadu_si256((__m256i*)pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint16 load_linear_all(const int *pSrc)
	{
		return vint16{ _mm256_load_si256((__m256i*)pSrc) };
	}

	// load/store linear float
	CPPSPMD_FORCE_INLINE void storeu_linear(float *pDst, const vfloat& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_ps((float*)pDst, src.m_value_l);
			_mm256_storeu_ps((float*)pDst + 8, src.m_value_h);
		}
		else
		{
			int *pDstI = (int *)pDst;
			if (mask & (1 << 0)) pDstI[0] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
			if (mask & (1 << 2)) pDstI[1] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
			if (mask & (1 << 4)) pDstI[2] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
			if (mask & (1 << 6)) pDstI[3] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

			if (mask & (1 << 8)) pDstI[4] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
			if (mask & (1 << 10)) pDstI[5] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
			if (mask & (1 << 12)) pDstI[6] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
			if (mask & (1 << 14)) pDstI[7] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

			if (mask & (1 << 16)) pDstI[8] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
			if (mask & (1 << 18)) pDstI[9] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
			if (mask & (1 << 20)) pDstI[10] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
			if (mask & (1 << 22)) pDstI[11] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

			if (mask & (1 << 24)) pDstI[12] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
			if (mask & (1 << 26)) pDstI[13] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
			if (mask & (1 << 28)) pDstI[14] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
			if (mask & (1 << 30)) pDstI[15] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(float *pDst, const vfloat& src)
	{
		_mm256_storeu_ps((float*)pDst, src.m_value_l);
		_mm256_storeu_ps((float*)pDst + 8, src.m_value_h);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(float *pDst, const vfloat& src)
	{
		_mm256_store_ps((float*)pDst, src.m_value_l);
		_mm256_store_ps((float*)pDst + 8, src.m_value_h);
	}
	
	CPPSPMD_FORCE_INLINE vfloat loadu_linear(const float *pSrc)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			return vfloat{ _mm256_loadu_ps((float*)pSrc), _mm256_loadu_ps((float*)pSrc + 8) };
		else
		{
			__m256i mask_l, mask_h;
			convert_16_to_32(m_exec.m_mask, mask_l, mask_h);

			return vfloat{ _mm256_maskload_ps(pSrc, mask_l), _mm256_maskload_ps(pSrc + 8, mask_h) };
		}
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear_all(const float *pSrc)
	{
		return vfloat{ _mm256_loadu_ps((float*)pSrc), _mm256_loadu_ps((float*)pSrc + 8) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_linear_all(const float *pSrc)
	{
		return vfloat{ _mm256_load_ps((float*)pSrc), _mm256_load_ps((float*)pSrc + 8) };
	}
	
	CPPSPMD_FORCE_INLINE vint16& store(vint16& dst, const vint16& src)
	{
		dst.m_value = _mm256_blendv_epi8(dst.m_value, src.m_value, m_exec.m_mask);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_vref& store_all(const int16_vref& dst, const vint16& src)
	{
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 0)] = (int16_t)_mm256_extract_epi16(src.m_value, 0);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 1)] = (int16_t)_mm256_extract_epi16(src.m_value, 1);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 2)] = (int16_t)_mm256_extract_epi16(src.m_value, 2);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 3)] = (int16_t)_mm256_extract_epi16(src.m_value, 3);

		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 4)] = (int16_t)_mm256_extract_epi16(src.m_value, 4);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 5)] = (int16_t)_mm256_extract_epi16(src.m_value, 5);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 6)] = (int16_t)_mm256_extract_epi16(src.m_value, 6);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 7)] = (int16_t)_mm256_extract_epi16(src.m_value, 7);

		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 8)] = (int16_t)_mm256_extract_epi16(src.m_value, 8);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 9)] = (int16_t)_mm256_extract_epi16(src.m_value, 9);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 10)] = (int16_t)_mm256_extract_epi16(src.m_value, 10);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 11)] = (int16_t)_mm256_extract_epi16(src.m_value, 11);

		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 12)] = (int16_t)_mm256_extract_epi16(src.m_value, 12);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 13)] = (int16_t)_mm256_extract_epi16(src.m_value, 13);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 14)] = (int16_t)_mm256_extract_epi16(src.m_value, 14);
		dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 15)] = (int16_t)_mm256_extract_epi16(src.m_value, 15);

		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_vref& store(const int16_vref& dst, const vint16& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			store_all(dst, src);
		}
		else
		{
			if (mask & (1 << 0))	dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 0)] = (int16_t)_mm256_extract_epi16(src.m_value, 0);
			if (mask & (1 << 2))	dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 1)] = (int16_t)_mm256_extract_epi16(src.m_value, 1);
			if (mask & (1 << 4))	dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 2)] = (int16_t)_mm256_extract_epi16(src.m_value, 2);
			if (mask & (1 << 6))	dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 3)] = (int16_t)_mm256_extract_epi16(src.m_value, 3);

			if (mask & (1 << 8))	dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 4)] = (int16_t)_mm256_extract_epi16(src.m_value, 4);
			if (mask & (1 << 10)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 5)] = (int16_t)_mm256_extract_epi16(src.m_value, 5);
			if (mask & (1 << 12)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 6)] = (int16_t)_mm256_extract_epi16(src.m_value, 6);
			if (mask & (1 << 14)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 7)] = (int16_t)_mm256_extract_epi16(src.m_value, 7);

			if (mask & (1 << 16)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 8)] = (int16_t)_mm256_extract_epi16(src.m_value, 8);
			if (mask & (1 << 18)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 9)] = (int16_t)_mm256_extract_epi16(src.m_value, 9);
			if (mask & (1 << 20)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 10)] = (int16_t)_mm256_extract_epi16(src.m_value, 10);
			if (mask & (1 << 22)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 11)] = (int16_t)_mm256_extract_epi16(src.m_value, 11);

			if (mask & (1 << 24)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 12)] = (int16_t)_mm256_extract_epi16(src.m_value, 12);
			if (mask & (1 << 26)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 13)] = (int16_t)_mm256_extract_epi16(src.m_value, 13);
			if (mask & (1 << 28)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 14)] = (int16_t)_mm256_extract_epi16(src.m_value, 14);
			if (mask & (1 << 30)) dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 15)] = (int16_t)_mm256_extract_epi16(src.m_value, 15);
		}
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vint16& store_all(vint16& dst, const vint16& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vint16 load_all(const int16_vref& src)
	{
		__m256i v = _mm256_undefined_si256();

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)], 0);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)], 1);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)], 2);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)], 3);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)], 4);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)], 5);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)], 6);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)], 7);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)], 8);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)], 9);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)], 10);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)], 11);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)], 12);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)], 13);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)], 14);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)], 15);
		
		return vint16{ v };
	}

	CPPSPMD_FORCE_INLINE vint16 load(const int16_vref& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			return load_all(src);
		else
		{
			__m256i v = _mm256_setzero_si256();

			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)], 15);

			return vint16{ v };
		}
	}
	
	CPPSPMD_FORCE_INLINE vint16 load_all(const cint16_vref& src)
	{
		__m256i v = _mm256_undefined_si256();

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)], 0);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)], 1);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)], 2);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)], 3);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)], 4);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)], 5);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)], 6);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)], 7);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)], 8);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)], 9);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)], 10);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)], 11);

		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)], 12);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)], 13);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)], 14);
		v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)], 15);

		return vint16{ v };
	}

	CPPSPMD_FORCE_INLINE vint16 load(const cint16_vref& src)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			return load_all(src);
		else
		{
			__m256i v = _mm256_setzero_si256();

			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)], 15);

			return vint16{ v };
		}
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(int16_t *pDst, uint32_t stride, const vint16 &v)
	{
		pDst[0] = (int16_t)_mm256_extract_epi16(v.m_value, 0);
		pDst[stride] = (int16_t)_mm256_extract_epi16(v.m_value, 1);
		pDst[stride * 2] = (int16_t)_mm256_extract_epi16(v.m_value, 2);
		pDst[stride * 3] = (int16_t)_mm256_extract_epi16(v.m_value, 3);
		pDst[stride * 4] = (int16_t)_mm256_extract_epi16(v.m_value, 4);
		pDst[stride * 5] = (int16_t)_mm256_extract_epi16(v.m_value, 5);
		pDst[stride * 6] = (int16_t)_mm256_extract_epi16(v.m_value, 6);
		pDst[stride * 7] = (int16_t)_mm256_extract_epi16(v.m_value, 7);

		pDst[stride * 8] = (int16_t)_mm256_extract_epi16(v.m_value, 8);
		pDst[stride * 9] = (int16_t)_mm256_extract_epi16(v.m_value, 9);
		pDst[stride * 10] = (int16_t)_mm256_extract_epi16(v.m_value, 10);
		pDst[stride * 11] = (int16_t)_mm256_extract_epi16(v.m_value, 11);
		pDst[stride * 12] = (int16_t)_mm256_extract_epi16(v.m_value, 12);
		pDst[stride * 13] = (int16_t)_mm256_extract_epi16(v.m_value, 13);
		pDst[stride * 14] = (int16_t)_mm256_extract_epi16(v.m_value, 14);
		pDst[stride * 15] = (int16_t)_mm256_extract_epi16(v.m_value, 15);
	}
	
	CPPSPMD_FORCE_INLINE void store_strided(int16_t *pDst, uint32_t stride, const vint16 &v)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
		{
			store_all_strided(pDst, stride, v);
		}
		else
		{
			if (mask & (1 << 0)) pDst[0] = (int16_t)_mm256_extract_epi16(v.m_value, 0);
			if (mask & (1 << 2)) pDst[stride] = (int16_t)_mm256_extract_epi16(v.m_value, 1);
			if (mask & (1 << 4)) pDst[stride * 2] = (int16_t)_mm256_extract_epi16(v.m_value, 2);
			if (mask & (1 << 6)) pDst[stride * 3] = (int16_t)_mm256_extract_epi16(v.m_value, 3);
			if (mask & (1 << 8)) pDst[stride * 4] = (int16_t)_mm256_extract_epi16(v.m_value, 4);
			if (mask & (1 << 10)) pDst[stride * 5] = (int16_t)_mm256_extract_epi16(v.m_value, 5);
			if (mask & (1 << 12)) pDst[stride * 6] = (int16_t)_mm256_extract_epi16(v.m_value, 6);
			if (mask & (1 << 14)) pDst[stride * 7] = (int16_t)_mm256_extract_epi16(v.m_value, 7);

			if (mask & (1 << 16)) pDst[stride * 8] = (int16_t)_mm256_extract_epi16(v.m_value, 8);
			if (mask & (1 << 18)) pDst[stride * 9] = (int16_t)_mm256_extract_epi16(v.m_value, 9);
			if (mask & (1 << 20)) pDst[stride * 10] = (int16_t)_mm256_extract_epi16(v.m_value, 10);
			if (mask & (1 << 22)) pDst[stride * 11] = (int16_t)_mm256_extract_epi16(v.m_value, 11);
			if (mask & (1 << 24)) pDst[stride * 12] = (int16_t)_mm256_extract_epi16(v.m_value, 12);
			if (mask & (1 << 26)) pDst[stride * 13] = (int16_t)_mm256_extract_epi16(v.m_value, 13);
			if (mask & (1 << 28)) pDst[stride * 14] = (int16_t)_mm256_extract_epi16(v.m_value, 14);
			if (mask & (1 << 30)) pDst[stride * 15] = (int16_t)_mm256_extract_epi16(v.m_value, 15);
		}
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		int *pDst = (int *)pDstF;

		pDst[0] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 0);
		pDst[stride] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 1);
		pDst[stride * 2] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 2);
		pDst[stride * 3] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 3);
		pDst[stride * 4] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 4);
		pDst[stride * 5] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 5);
		pDst[stride * 6] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 6);
		pDst[stride * 7] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 7);

		pDst[stride * 8] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 0);
		pDst[stride * 9] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 1);
		pDst[stride * 10] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 2);
		pDst[stride * 11] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 3);
		pDst[stride * 12] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 4);
		pDst[stride * 13] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 5);
		pDst[stride * 14] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 6);
		pDst[stride * 15] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 7);
	}

	CPPSPMD_FORCE_INLINE void store_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			store_all_strided(pDstF, stride, v);
		else
		{
			int *pDst = (int *)pDstF;

			if (mask & (1 << 0)) pDst[0] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 0);
			if (mask & (1 << 2)) pDst[stride] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 1);
			if (mask & (1 << 4)) pDst[stride * 2] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 2);
			if (mask & (1 << 6)) pDst[stride * 3] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 3);
			if (mask & (1 << 8)) pDst[stride * 4] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 4);
			if (mask & (1 << 10)) pDst[stride * 5] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 5);
			if (mask & (1 << 12)) pDst[stride * 6] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 6);
			if (mask & (1 << 14)) pDst[stride * 7] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_l), 7);

			if (mask & (1 << 16)) pDst[stride * 8] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 0);
			if (mask & (1 << 18)) pDst[stride * 9] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 1);
			if (mask & (1 << 20)) pDst[stride * 10] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 2);
			if (mask & (1 << 22)) pDst[stride * 11] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 3);
			if (mask & (1 << 24)) pDst[stride * 12] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 4);
			if (mask & (1 << 26)) pDst[stride * 13] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 5);
			if (mask & (1 << 28)) pDst[stride * 14] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 6);
			if (mask & (1 << 30)) pDst[stride * 15] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value_h), 7);
		}
	}

	CPPSPMD_FORCE_INLINE vint16 load_all_strided(const int16_t *pSrc, uint32_t stride)
	{
		__m256i v = _mm256_undefined_si256();

		v = _mm256_insert_epi16(v, pSrc[0], 0);
		v = _mm256_insert_epi16(v, pSrc[stride], 1);
		v = _mm256_insert_epi16(v, pSrc[stride * 2], 2);
		v = _mm256_insert_epi16(v, pSrc[stride * 3], 3);

		v = _mm256_insert_epi16(v, pSrc[stride * 4], 4);
		v = _mm256_insert_epi16(v, pSrc[stride * 5], 5);
		v = _mm256_insert_epi16(v, pSrc[stride * 6], 6);
		v = _mm256_insert_epi16(v, pSrc[stride * 7], 7);

		v = _mm256_insert_epi16(v, pSrc[stride * 8], 8);
		v = _mm256_insert_epi16(v, pSrc[stride * 9], 9);
		v = _mm256_insert_epi16(v, pSrc[stride * 10], 10);
		v = _mm256_insert_epi16(v, pSrc[stride * 11], 11);

		v = _mm256_insert_epi16(v, pSrc[stride * 12], 12);
		v = _mm256_insert_epi16(v, pSrc[stride * 13], 13);
		v = _mm256_insert_epi16(v, pSrc[stride * 14], 14);
		v = _mm256_insert_epi16(v, pSrc[stride * 15], 15);

		return vint16{ v };
	}
	
	CPPSPMD_FORCE_INLINE vint16 load_strided(const int16_t *pSrc, uint32_t stride)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			return load_all_strided(pSrc, stride);
		else
		{
			__m256i v = _mm256_setzero_si256();

			if (mask & (1 << 0)) v = _mm256_insert_epi16(v, pSrc[0], 0);
			if (mask & (1 << 2)) v = _mm256_insert_epi16(v, pSrc[stride], 1);
			if (mask & (1 << 4)) v = _mm256_insert_epi16(v, pSrc[stride * 2], 2);
			if (mask & (1 << 6)) v = _mm256_insert_epi16(v, pSrc[stride * 3], 3);

			if (mask & (1 << 8)) v = _mm256_insert_epi16(v, pSrc[stride * 4], 4);
			if (mask & (1 << 10)) v = _mm256_insert_epi16(v, pSrc[stride * 5], 5);
			if (mask & (1 << 12)) v = _mm256_insert_epi16(v, pSrc[stride * 6], 6);
			if (mask & (1 << 14)) v = _mm256_insert_epi16(v, pSrc[stride * 7], 7);

			if (mask & (1 << 16)) v = _mm256_insert_epi16(v, pSrc[stride * 8], 8);
			if (mask & (1 << 18)) v = _mm256_insert_epi16(v, pSrc[stride * 9], 9);
			if (mask & (1 << 20)) v = _mm256_insert_epi16(v, pSrc[stride * 10], 10);
			if (mask & (1 << 22)) v = _mm256_insert_epi16(v, pSrc[stride * 11], 11);

			if (mask & (1 << 24)) v = _mm256_insert_epi16(v, pSrc[stride * 12], 12);
			if (mask & (1 << 26)) v = _mm256_insert_epi16(v, pSrc[stride * 13], 13);
			if (mask & (1 << 28)) v = _mm256_insert_epi16(v, pSrc[stride * 14], 14);
			if (mask & (1 << 30)) v = _mm256_insert_epi16(v, pSrc[stride * 15], 15);

			return vint16{ v };
		}
	}

	CPPSPMD_FORCE_INLINE vfloat load_all_strided(const float *pSrc, uint32_t stride)
	{
		const int *pSrcI = (const int *)pSrc;

		__m256i l = _mm256_undefined_si256(), h = _mm256_undefined_si256();

		l = _mm256_insert_epi32(l, pSrcI[0], 0);
		l = _mm256_insert_epi32(l, pSrcI[stride], 1);
		l = _mm256_insert_epi32(l, pSrcI[stride * 2], 2);
		l = _mm256_insert_epi32(l, pSrcI[stride * 3], 3);

		l = _mm256_insert_epi32(l, pSrcI[stride * 4], 4);
		l = _mm256_insert_epi32(l, pSrcI[stride * 5], 5);
		l = _mm256_insert_epi32(l, pSrcI[stride * 6], 6);
		l = _mm256_insert_epi32(l, pSrcI[stride * 7], 7);

		h = _mm256_insert_epi32(h, pSrcI[stride * 8], 0);
		h = _mm256_insert_epi32(h, pSrcI[stride * 9], 1);
		h = _mm256_insert_epi32(h, pSrcI[stride * 10], 2);
		h = _mm256_insert_epi32(h, pSrcI[stride * 11], 3);

		h = _mm256_insert_epi32(h, pSrcI[stride * 12], 4);
		h = _mm256_insert_epi32(h, pSrcI[stride * 13], 5);
		h = _mm256_insert_epi32(h, pSrcI[stride * 14], 6);
		h = _mm256_insert_epi32(h, pSrcI[stride * 15], 7);

		return vfloat{ _mm256_castsi256_ps(l), _mm256_castsi256_ps(h) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_strided(const float *pSrc, uint32_t stride)
	{
		uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);
		if (mask == ALL_ON_MOVEMASK)
			return load_all_strided(pSrc, stride);
		else
		{
			const int *pSrcI = (const int *)pSrc;

			__m256i l = _mm256_setzero_si256(), h = _mm256_setzero_si256();

			if (mask & (1 << 0)) l = _mm256_insert_epi32(l, pSrcI[0], 0);
			if (mask & (1 << 2)) l = _mm256_insert_epi32(l, pSrcI[stride], 1);
			if (mask & (1 << 4)) l = _mm256_insert_epi32(l, pSrcI[stride * 2], 2);
			if (mask & (1 << 6)) l = _mm256_insert_epi32(l, pSrcI[stride * 3], 3);

			if (mask & (1 << 8)) l = _mm256_insert_epi32(l, pSrcI[stride * 4], 4);
			if (mask & (1 << 10)) l = _mm256_insert_epi32(l, pSrcI[stride * 5], 5);
			if (mask & (1 << 12)) l = _mm256_insert_epi32(l, pSrcI[stride * 6], 6);
			if (mask & (1 << 14)) l = _mm256_insert_epi32(l, pSrcI[stride * 7], 7);

			if (mask & (1 << 16)) h = _mm256_insert_epi32(h, pSrcI[stride * 8], 0);
			if (mask & (1 << 18)) h = _mm256_insert_epi32(h, pSrcI[stride * 9], 1);
			if (mask & (1 << 20)) h = _mm256_insert_epi32(h, pSrcI[stride * 10], 2);
			if (mask & (1 << 22)) h = _mm256_insert_epi32(h, pSrcI[stride * 11], 3);

			if (mask & (1 << 24)) h = _mm256_insert_epi32(h, pSrcI[stride * 12], 4);
			if (mask & (1 << 26)) h = _mm256_insert_epi32(h, pSrcI[stride * 13], 5);
			if (mask & (1 << 28)) h = _mm256_insert_epi32(h, pSrcI[stride * 14], 6);
			if (mask & (1 << 30)) h = _mm256_insert_epi32(h, pSrcI[stride * 15], 7);

			return vfloat{ _mm256_castsi256_ps(l), _mm256_castsi256_ps(h) };
		}
	}

	CPPSPMD_FORCE_INLINE const vfloat_vref& store(const vfloat_vref& dst, const vfloat& src)
	{
		int mask = _mm256_movemask_epi8(m_exec.m_mask);
		
		if (mask & (1 << 0)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 0)]))[0] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
		if (mask & (1 << 2)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 1)]))[1] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
		if (mask & (1 << 4)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 2)]))[2] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
		if (mask & (1 << 6)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 3)]))[3] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

		if (mask & (1 << 8)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 4)]))[4] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
		if (mask & (1 << 10)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 5)]))[5] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
		if (mask & (1 << 12)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 6)]))[6] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
		if (mask & (1 << 14)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 7)]))[7] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

		if (mask & (1 << 16)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 8)]))[8] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
		if (mask & (1 << 18)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 9)]))[9] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
		if (mask & (1 << 20)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 10)]))[10] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
		if (mask & (1 << 22)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 11)]))[11] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

		if (mask & (1 << 24)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 12)]))[12] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
		if (mask & (1 << 26)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 13)]))[13] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
		if (mask & (1 << 28)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 14)]))[14] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
		if (mask & (1 << 30)) ((int *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 15)]))[15] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		int mask = _mm256_movemask_epi8(m_exec.m_mask);

		__m256i l = _mm256_setzero_si256(), h = _mm256_setzero_si256();

		if (mask & (1 << 0)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)]))[0], 0);
		if (mask & (1 << 2)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)]))[1], 1);
		if (mask & (1 << 4)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)]))[2], 2);
		if (mask & (1 << 6)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)]))[3], 3);

		if (mask & (1 << 8)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)]))[4], 4);
		if (mask & (1 << 10)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)]))[5], 5);
		if (mask & (1 << 12)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)]))[6], 6);
		if (mask & (1 << 14)) l = _mm256_insert_epi32(l, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)]))[7], 7);

		if (mask & (1 << 16)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)]))[8], 0);
		if (mask & (1 << 18)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)]))[9], 1);
		if (mask & (1 << 20)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)]))[10], 2);
		if (mask & (1 << 22)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)]))[11], 3);

		if (mask & (1 << 24)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)]))[12], 4);
		if (mask & (1 << 26)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)]))[13], 5);
		if (mask & (1 << 28)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)]))[14], 6);
		if (mask & (1 << 30)) h = _mm256_insert_epi32(h, ((int *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)]))[15], 7);

		return vfloat{ _mm256_castsi256_ps(l), _mm256_castsi256_ps(h) };
	}

	CPPSPMD_FORCE_INLINE const vint16_vref& store(const vint16_vref& dst, const vint16& src)
	{
		int mask = _mm256_movemask_epi8(m_exec.m_mask);
		
		if (mask & (1 << 0)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 0)]))[0] = (int16_t)_mm256_extract_epi16(src.m_value, 0);
		if (mask & (1 << 2)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 1)]))[1] = (int16_t)_mm256_extract_epi16(src.m_value, 1);
		if (mask & (1 << 4)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 2)]))[2] = (int16_t)_mm256_extract_epi16(src.m_value, 2);
		if (mask & (1 << 6)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 3)]))[3] = (int16_t)_mm256_extract_epi16(src.m_value, 3);

		if (mask & (1 << 8)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 4)]))[4] = (int16_t)_mm256_extract_epi16(src.m_value, 4);
		if (mask & (1 << 10)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 5)]))[5] = (int16_t)_mm256_extract_epi16(src.m_value, 5);
		if (mask & (1 << 12)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 6)]))[6] = (int16_t)_mm256_extract_epi16(src.m_value, 6);
		if (mask & (1 << 14)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 7)]))[7] = (int16_t)_mm256_extract_epi16(src.m_value, 7);

		if (mask & (1 << 16)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 8)]))[8] = (int16_t)_mm256_extract_epi16(src.m_value, 8);
		if (mask & (1 << 18)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 9)]))[9] = (int16_t)_mm256_extract_epi16(src.m_value, 9);
		if (mask & (1 << 20)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 10)]))[10] = (int16_t)_mm256_extract_epi16(src.m_value, 10);
		if (mask & (1 << 22)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 11)]))[11] = (int16_t)_mm256_extract_epi16(src.m_value, 11);

		if (mask & (1 << 24)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 12)]))[12] = (int16_t)_mm256_extract_epi16(src.m_value, 12);
		if (mask & (1 << 26)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 13)]))[13] = (int16_t)_mm256_extract_epi16(src.m_value, 13);
		if (mask & (1 << 28)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 14)]))[14] = (int16_t)_mm256_extract_epi16(src.m_value, 14);
		if (mask & (1 << 30)) ((int16_t *)(&dst.m_pValue[_mm256_extract_epi16(dst.m_vindex, 15)]))[15] = (int16_t)_mm256_extract_epi16(src.m_value, 15);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint16 load(const vint16_vref& src)
	{
		int mask = _mm256_movemask_epi8(m_exec.m_mask);

		__m256i k = _mm256_setzero_si256();

		if (mask & (1 << 0)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 0)]))[0], 0);
		if (mask & (1 << 2)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 1)]))[1], 1);
		if (mask & (1 << 4)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 2)]))[2], 2);
		if (mask & (1 << 6)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 3)]))[3], 3);

		if (mask & (1 << 8)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 4)]))[4], 4);
		if (mask & (1 << 10)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 5)]))[5], 5);
		if (mask & (1 << 12)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 6)]))[6], 6);
		if (mask & (1 << 14)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 7)]))[7], 7);
		
		if (mask & (1 << 16)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 8)]))[8], 8);
		if (mask & (1 << 18)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 9)]))[9], 9);
		if (mask & (1 << 20)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 10)]))[10], 10);
		if (mask & (1 << 22)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 11)]))[11], 11);

		if (mask & (1 << 24)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 12)]))[12], 12);
		if (mask & (1 << 26)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 13)]))[13], 13);
		if (mask & (1 << 28)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 14)]))[14], 14);
		if (mask & (1 << 30)) k = _mm256_insert_epi16(k, ((int16_t *)(&src.m_pValue[_mm256_extract_epi16(src.m_vindex, 15)]))[15], 15);

		return vint16{ k };
	}
	
	// Linear integer
	struct lint16
	{
		__m256i m_value;

		CPPSPMD_FORCE_INLINE explicit lint16(__m256i value) : m_value(value)	{ }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const	
		{	
			__m256i l, h;
			convert_16_to_32(m_value, l, h);

			return vfloat{ _mm256_cvtepi32_ps(l), _mm256_cvtepi32_ps(h) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vint16() const
		{
			return vint16{ m_value };
		}

		int get_first_value() const 
		{
			// TODO: Should this be signed or unsigned? For now, signed.
			int16_t v = (int16_t)(_mm_cvtsi128_si32(_mm256_castsi256_si128(m_value)));
			return v;
		}

		CPPSPMD_FORCE_INLINE float_lref operator[](float* ptr) const
		{
			return float_lref{ ptr + get_first_value() };
		}

		CPPSPMD_FORCE_INLINE int16_lref operator[](int16_t* ptr) const
		{
			return int16_lref{ ptr + get_first_value() };
		}

		CPPSPMD_FORCE_INLINE cint16_lref operator[](const int16_t* ptr) const
		{
			return cint16_lref{ ptr + get_first_value() };
		}

	private:
		lint16& operator=(const lint16&);
	};

	CPPSPMD_FORCE_INLINE lint16& store_all(lint16& dst, const lint16& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
	
	const lint16 program_index = lint16{ _mm256_set_epi16( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 ) };
	
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
	CPPSPMD_FORCE_INLINE void spmd_return();;
	
	template<typename UnmaskedBody>
	CPPSPMD_FORCE_INLINE void spmd_unmasked(const UnmaskedBody& unmaskedBody);

	template<typename SPMDKernel, typename... Args>
	CPPSPMD_FORCE_INLINE decltype(auto) spmd_call(Args&&... args);

	CPPSPMD_FORCE_INLINE void swap(vint16 &a, vint16 &b) { vint16 temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vfloat &a, vfloat &b) { vfloat temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vbool &a, vbool &b) { vbool temp = a; store(a, b); store(b, temp); }
		
}; // struct spmd_kernel

using exec_mask = spmd_kernel::exec_mask;
using vint16 = spmd_kernel::vint16;
using int16_lref = spmd_kernel::int16_lref;
using cint16_vref = spmd_kernel::cint16_vref;
using int16_vref = spmd_kernel::int16_vref;
using lint16 = spmd_kernel::lint16;
using vbool = spmd_kernel::vbool;
using vfloat = spmd_kernel::vfloat;
using float_lref = spmd_kernel::float_lref;
using float_vref = spmd_kernel::float_vref;
using vfloat_vref = spmd_kernel::vfloat_vref;
using vint16_vref = spmd_kernel::vint16_vref;

CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vfloat() const 
{ 
	__m256i l, h;
	convert_16_to_32(m_value, l, h);

	return vfloat { _mm256_and_ps( _mm256_castsi256_ps(l), *(const __m256 *)g_onef_256 ), _mm256_and_ps(_mm256_castsi256_ps(h), *(const __m256 *)g_onef_256) };
}

// Returns UINT16_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint16() const 
{ 
	return vint16 { m_value };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
	return vbool{ _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), v.m_value) };
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ xor_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) {	return exec_mask{ and_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ or_si256(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return (uint32_t)_mm256_movemask_epi8(e.m_mask) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return _mm256_movemask_epi8(e.m_mask) != 0; }

// Bad pattern - doesn't factor in the current exec mask. Prefer spmd_any() instead.
CPPSPMD_FORCE_INLINE bool all(const vbool& e) { return (uint32_t)_mm256_movemask_epi8(e.m_value) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const vbool& e) { return _mm256_movemask_epi8(e.m_value) != 0; }

CPPSPMD_FORCE_INLINE exec_mask andnot(const exec_mask& a, const exec_mask& b) { return exec_mask{ andnot_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE vbool operator||(const vbool& a, const vbool& b) { return vbool{ or_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator&&(const vbool& a, const vbool& b) { return vbool{ and_si256(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_add_ps(a.m_value_l, b.m_value_l), _mm256_add_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_sub_ps(a.m_value_l, b.m_value_l), _mm256_sub_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const vfloat& b) { return vfloat(a) + b; }
CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, float b) { return a + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vint16& b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vint16& a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, int b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(int a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, float b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(float a, const vfloat& b) { return vfloat(a) - b; }

CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_mul_ps(a.m_value_l, b.m_value_l), _mm256_mul_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, float b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float a, const vfloat& b) { return vfloat(a) * b; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, int b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(int a, const vfloat& b) { return vfloat(a) * b; }

CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_div_ps(a.m_value_l, b.m_value_l), _mm256_div_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, int b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(int a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, float b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(float a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& v) { return vfloat{ _mm256_sub_ps(_mm256_xor_ps(v.m_value_l, v.m_value_l), v.m_value_l), _mm256_sub_ps(_mm256_xor_ps(v.m_value_h, v.m_value_h), v.m_value_h) }; }

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, const vfloat& b) 
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_EQ_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_EQ_OQ));
	return vbool{ convert_32_to_16(l, h) }; 
}

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, float b) { return a == vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, const vfloat& b) 
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_EQ_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_EQ_OQ));
	return !vbool{ convert_32_to_16(l, h) };
}

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, float b) { return a != vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, const vfloat& b) 
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_LT_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_LT_OQ));
	return vbool{ convert_32_to_16(l, h) };
}

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, float b) { return a < vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, const vfloat& b)  
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_GT_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_GT_OQ));
	return vbool{ convert_32_to_16(l, h) };
}

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, float b) { return a > vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, const vfloat& b) 
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_LE_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_LE_OQ));
	return vbool{ convert_32_to_16(l, h) };
}

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, float b) { return a <= vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, const vfloat& b) 
{ 
	__m256i l = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_l, b.m_value_l, _CMP_GE_OQ));
	__m256i h = _mm256_castps_si256(_mm256_cmp_ps(a.m_value_h, b.m_value_h, _CMP_GE_OQ));
	return vbool{ convert_32_to_16(l, h) };
}

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, float b) { return a >= vfloat(b); }

CPPSPMD_FORCE_INLINE vfloat spmd_ternaryf(const vbool& cond, const vfloat& a, const vfloat& b) 
{ 
	__m256i l, h;
	convert_16_to_32(cond.m_value, l, h);
	return vfloat{ _mm256_blendv_ps(b.m_value_l, a.m_value_l, _mm256_castsi256_ps(l)), _mm256_blendv_ps(b.m_value_h, a.m_value_h, _mm256_castsi256_ps(h)) };
}

CPPSPMD_FORCE_INLINE vint16 spmd_ternaryi(const vbool& cond, const vint16& a, const vint16& b) 
{ 
	return vint16{ _mm256_blendv_epi8(b.m_value, a.m_value, cond.m_value) }; 
}

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ _mm256_sqrt_ps(v.m_value_l), _mm256_sqrt_ps(v.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v.m_value_l), _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_max_ps(a.m_value_l, b.m_value_l), _mm256_max_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_min_ps(a.m_value_l, b.m_value_l), _mm256_min_ps(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ _mm256_ceil_ps(a.m_value_l), _mm256_ceil_ps(a.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ _mm256_floor_ps(v.m_value_l), _mm256_floor_ps(v.m_value_h) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value_l, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ), _mm256_round_ps(a.m_value_h, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value_l, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC ), _mm256_round_ps(a.m_value_h, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }; }
CPPSPMD_FORCE_INLINE vfloat frac(const vfloat& a) { return a - floor(a); }
CPPSPMD_FORCE_INLINE vfloat fmod(vfloat a, vfloat b) { vfloat c = frac(abs(a / b)) * abs(b); return spmd_ternaryf(a < 0, -c, c); }

CPPSPMD_FORCE_INLINE vfloat sign(const vfloat &a)
{
	__m256 z = _mm256_setzero_ps(), n = _mm256_set1_ps(-1.0f), p = _mm256_set1_ps(1.0f);
	__m256 mask_l = _mm256_cmp_ps(a.m_value_l, z, _CMP_LT_OQ), mask_h = _mm256_cmp_ps(a.m_value_h, z, _CMP_LT_OQ);
	return vfloat{ _mm256_blendv_ps(p, n, mask_l), _mm256_blendv_ps(p, n, mask_h) };
}

static CPPSPMD_FORCE_INLINE vfloat recip_approx1(const vfloat &q)
{
	//VASSERT(q > 0.0f);

	const __m256i mag = _mm256_set1_epi32(0x7EF311C3);
	const __m256 two = _mm256_set1_ps(2.0f);
	
	__m256i x_l = _mm256_castps_si256(q.m_value_l);
	__m256i x_h = _mm256_castps_si256(q.m_value_h);

	x_l = _mm256_sub_epi32(mag, x_l);
	x_h = _mm256_sub_epi32(mag, x_h);
	
	__m256 rcp_l = _mm256_castsi256_ps(x_l);
	__m256 rcp_h = _mm256_castsi256_ps(x_h);

	rcp_l = _mm256_mul_ps(rcp_l, _mm256_fnmadd_ps(rcp_l, q.m_value_l, two));
	rcp_h = _mm256_mul_ps(rcp_h, _mm256_fnmadd_ps(rcp_h, q.m_value_h, two));

	return vfloat{ rcp_l, rcp_h };
}

static CPPSPMD_FORCE_INLINE vfloat recip_approx1_pn(const vfloat &q)
{
	vfloat s = sign(q);
	vfloat a = abs(q);

	const __m256 thresh = _mm256_set1_ps(.000125f);
	const __m256i mag = _mm256_set1_epi32(0x7EF311C3);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 defv = _mm256_castsi256_ps(mag);

	__m256 notzero_cond_l = _mm256_cmp_ps(a.m_value_l, thresh, _CMP_GE_OQ);
	__m256 notzero_cond_h = _mm256_cmp_ps(a.m_value_h, thresh, _CMP_GE_OQ);
		
	__m256 l = _mm256_blendv_ps(defv, a.m_value_l, notzero_cond_l);
	__m256 h = _mm256_blendv_ps(defv, a.m_value_h, notzero_cond_h);
		
	__m256i x_l = _mm256_castps_si256(l);
	__m256i x_h = _mm256_castps_si256(h);
	x_l = _mm256_sub_epi32(mag, x_l);
	x_h = _mm256_sub_epi32(mag, x_h);
	__m256 rcp_l = _mm256_castsi256_ps(x_l);
	__m256 rcp_h = _mm256_castsi256_ps(x_h);
	rcp_l = _mm256_mul_ps(_mm256_mul_ps(rcp_l, _mm256_fnmadd_ps(rcp_l, q.m_value_l, two)), s.m_value_l);
	rcp_h = _mm256_mul_ps(_mm256_mul_ps(rcp_h, _mm256_fnmadd_ps(rcp_h, q.m_value_h, two)), s.m_value_h);

	return vfloat{ rcp_l, rcp_h };
}

static CPPSPMD_FORCE_INLINE vfloat recip_approx1_p(const vfloat &q)
{
	const __m256 thresh = _mm256_set1_ps(.000125f);
	//const __m256i mag = _mm256_set1_epi32(0x7EF312AC); // 2 NR iters, 3 is  0x7EEEEBB3
	const __m256i mag = _mm256_set1_epi32(0x7EF311C3);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 defv = _mm256_castsi256_ps(mag);

	__m256 notzero_cond_l = _mm256_cmp_ps(q.m_value_l, thresh, _CMP_GE_OQ);
	__m256 notzero_cond_h = _mm256_cmp_ps(q.m_value_h, thresh, _CMP_GE_OQ);

	__m256 l = _mm256_blendv_ps(defv, q.m_value_l, notzero_cond_l);
	__m256 h = _mm256_blendv_ps(defv, q.m_value_h, notzero_cond_h);

	__m256i x_l = _mm256_castps_si256(l);
	__m256i x_h = _mm256_castps_si256(h);
	x_l = _mm256_sub_epi32(mag, x_l);
	x_h = _mm256_sub_epi32(mag, x_h);
	__m256 rcp_l = _mm256_castsi256_ps(x_l);
	__m256 rcp_h = _mm256_castsi256_ps(x_h);
	rcp_l = _mm256_mul_ps(rcp_l, _mm256_fnmadd_ps(rcp_l, q.m_value_l, two));
	rcp_h = _mm256_mul_ps(rcp_h, _mm256_fnmadd_ps(rcp_h, q.m_value_h, two));

	return vfloat{ rcp_l, rcp_h };
}

static CPPSPMD_FORCE_INLINE vfloat div_approx1_pn_check(float m, const vfloat &a)
{
	return m * recip_approx1_pn(a);
}

CPPSPMD_FORCE_INLINE vfloat div_check(float m, const vfloat &a)
{
	const __m256 z = _mm256_setzero_ps();

	__m256 notzero_cond_l = _mm256_cmp_ps(a.m_value_l, z, _CMP_NEQ_OQ);
	__m256 notzero_cond_h = _mm256_cmp_ps(a.m_value_h, z, _CMP_NEQ_OQ);
		
	__m256 d = _mm256_set1_ps(m);

	__m256 l = _mm256_blendv_ps(d, a.m_value_l, notzero_cond_l);
	__m256 h = _mm256_blendv_ps(d, a.m_value_h, notzero_cond_h);
	
	l = _mm256_div_ps(d, l);
	h = _mm256_div_ps(d, h);

	return vfloat{ _mm256_blendv_ps(z, l, notzero_cond_l), _mm256_blendv_ps(z, h, notzero_cond_h) };
}

CPPSPMD_FORCE_INLINE vint16 avg(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_avg_epu16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 adds(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_adds_epi16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 addsu(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_adds_epu16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 subs(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_subs_epi16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 subsu(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_subs_epu16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 max(const vint16& a, const vint16& b) 
{ 
	return vint16{ _mm256_max_epi16(a.m_value, b.m_value) }; 
}

CPPSPMD_FORCE_INLINE vint16 min(const vint16& a, const vint16& b) 
{	
	return vint16{ _mm256_min_epi16(a.m_value, b.m_value) }; 
}

CPPSPMD_FORCE_INLINE vint16 maxu(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_max_epu16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 minu(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_min_epu16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 abs(const vint16 &a) 
{ 
	__m256i s = _mm256_srai_epi16(a.m_value, 15);
	return vint16{ _mm256_sub_epi16(_mm256_xor_si256(a.m_value, s), s) };
}

CPPSPMD_FORCE_INLINE vint16 absdiffu(const vint16 &a, const vint16 &b) { return vint16{ _mm256_absdiff_epu16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 div255u(const vint16 &a) { return vint16{ _mm256_div255_epu16(a.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ _mm256_min_ps(b.m_value_l, _mm256_max_ps(v.m_value_l, a.m_value_l) ), _mm256_min_ps(b.m_value_h, _mm256_max_ps(v.m_value_h, a.m_value_h)) };
}

CPPSPMD_FORCE_INLINE vint16 clamp(const vint16& v, const vint16& a, const vint16& b)
{
	return vint16{ _mm256_max_epi16(a.m_value, _mm256_min_epi16(v.m_value, b.m_value) ) };
}

CPPSPMD_FORCE_INLINE vint16 clampu(const vint16& v, const vint16& a, const vint16& b)
{
	return vint16{ _mm256_max_epu16(a.m_value, _mm256_min_epu16(v.m_value, b.m_value)) };
}

CPPSPMD_FORCE_INLINE vint16 mulhrs(const vint16& a, const vint16& b)
{
	return vint16{ _mm256_mulhrs_epi16(a.m_value, b.m_value) };
}

// a * b + c
CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_fmadd_ps(a.m_value_l, b.m_value_l, c.m_value_l), _mm256_fmadd_ps(a.m_value_h, b.m_value_h, c.m_value_h) };
}

// a * b - c
CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_fmsub_ps(a.m_value_l, b.m_value_l, c.m_value_l), _mm256_fmsub_ps(a.m_value_h, b.m_value_h, c.m_value_h) };
}

// -a * b + c
CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_fnmadd_ps(a.m_value_l, b.m_value_l, c.m_value_l), _mm256_fnmadd_ps(a.m_value_h, b.m_value_h, c.m_value_h) };
}

// -a * b - c
CPPSPMD_FORCE_INLINE vfloat vfnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_fnmsub_ps(a.m_value_l, b.m_value_l, c.m_value_l), _mm256_fnmsub_ps(a.m_value_h, b.m_value_h, c.m_value_h) };
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint16 operator+(int a, const lint16& b) { return lint16{ _mm256_add_epi16(_mm256_set1_epi16((int16_t)a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint16 operator+(const lint16& a, int b) { return lint16{ _mm256_add_epi16(a.m_value, _mm256_set1_epi16((int16_t)b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint16& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint16& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint16& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint16& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint16 operator&(const vint16& a, const vint16& b) { return vint16{ and_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator&(const vint16& a, int b) { return a & vint16(b); }
CPPSPMD_FORCE_INLINE vint16 andnot(const vint16& a, const vint16& b) { return vint16{ andnot_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator|(const vint16& a, const vint16& b) { return vint16{ or_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator|(const vint16& a, int b) { return a | vint16(b); }
CPPSPMD_FORCE_INLINE vint16 operator^(const vint16& a, const vint16& b) { return vint16{ xor_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator^(const vint16& a, int b) { return a ^ vint16(b); }

CPPSPMD_FORCE_INLINE vbool operator==(const vint16& a, const vint16& b) { return vbool{ _mm256_cmpeq_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vint16& a, const vint16& b) { return !vbool{ _mm256_cmpeq_epi16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vbool operator<(const vint16& a, const vint16& b) { return vbool{ _mm256_cmpgt_epi16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vint16& a, const vint16& b) { return !vbool{ _mm256_cmpgt_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vint16& a, const vint16& b) { return !vbool{ _mm256_cmpgt_epi16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vint16& a, const vint16& b) { return vbool{ _mm256_cmpgt_epi16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vbool cmp_lt_u(const vint16& a, const vint16& b) { return vbool{ _mm256_cmpgt_epu16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool cmp_le_u(const vint16& a, const vint16& b) { return !vbool{ _mm256_cmpgt_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool cmp_ge_u(const vint16& a, const vint16& b) { return !vbool{ _mm256_cmpgt_epu16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool cmp_gt_u(const vint16& a, const vint16& b) { return vbool{ _mm256_cmpgt_epu16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 operator+(const vint16& a, const vint16& b) { return vint16{ _mm256_add_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator-(const vint16& a, const vint16& b) { return vint16{ _mm256_sub_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator+(const vint16& a, int b) { return a + vint16(b); }
CPPSPMD_FORCE_INLINE vint16 operator-(const vint16& a, int b) { return a - vint16(b); }
CPPSPMD_FORCE_INLINE vint16 operator+(int a, const vint16& b) { return vint16(a) + b; }
CPPSPMD_FORCE_INLINE vint16 operator-(int a, const vint16& b) { return vint16(a) - b; }
CPPSPMD_FORCE_INLINE vint16 operator*(const vint16& a, const vint16& b) { return vint16{ _mm256_mullo_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 operator*(const vint16& a, int b) { return a * vint16(b); }
CPPSPMD_FORCE_INLINE vint16 operator*(int a, const vint16& b) { return vint16(a) * b; }

CPPSPMD_FORCE_INLINE vint16 mulhi(const vint16& a, const vint16& b) { return vint16{ _mm256_mulhi_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 mulhi(const vint16& a, int b) { return vint16{ _mm256_mulhi_epi16(a.m_value, _mm256_set1_epi16((int16_t)b)) }; }
CPPSPMD_FORCE_INLINE vint16 mulhi(int a, const vint16& b) { return vint16{ _mm256_mulhi_epi16(_mm256_set1_epi16((int16_t)a), b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 umulhi(const vint16& a, const vint16& b) { return vint16{ _mm256_mulhi_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 umulhi(const vint16& a, uint16_t b) { return vint16{ _mm256_mulhi_epu16(a.m_value, _mm256_set1_epi16(b)) }; }
CPPSPMD_FORCE_INLINE vint16 umulhi(uint16_t a, const vint16& b) { return vint16{ _mm256_mulhi_epu16(_mm256_set1_epi16(a), b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 sign(const vint16 &a) { return vint16{ _mm256_or_si256(_mm256_srai_epi16(a.m_value, 15), _mm256_set1_epi16(1)) }; }

CPPSPMD_FORCE_INLINE vint16 operator-(const vint16& v) { return vint16{ _mm256_sub_epi16(_mm256_setzero_si256(), v.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 operator~(const vint16& a) { return vint16{ -a - 1 }; }

// Vertically multiply each unsigned 8 - bit integer from a with the corresponding signed 8 - bit integer from b, producing intermediate 
// signed 16 - bit integers.Horizontally add adjacent pairs of intermediate signed 16 - bit integers, and pack the saturated results in dst.
// FOR j : = 0 to 15
//	  i : = j * 16
//	  dst[i + 15:i] : = Saturate16(a[i + 15:i + 8] * b[i + 15:i + 8] + a[i + 7:i] * b[i + 7:i])
// ENDFOR
CPPSPMD_FORCE_INLINE vint16 maddubs(const vint16& a, const vint16& b) 
{ 
	return vint16{ _mm256_maddubs_epi16(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint16 adds_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_adds_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 subs_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_subs_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 avg_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_avg_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 max_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_max_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 min_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_min_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 sad_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_sad_epu8(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 add_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_add_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 adds_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_adds_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 sub_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_sub_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 subs_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_subs_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmpeq_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpeq_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmpgt_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpgt_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmplt_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpgt_epi8(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 unpacklo_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_unpacklo_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 unpackhi_epi8(const vint16& a, const vint16& b) { return vint16{ _mm256_unpackhi_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE int movemask_epi8(const vint16& a) { return _mm256_movemask_epi8(a.m_value); }

CPPSPMD_FORCE_INLINE vint16 cmple_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpeq_epi8(_mm256_min_epu8(a.m_value, b.m_value), a.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmpge_epu8(const vint16& a, const vint16& b) { return vint16{ cmple_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint16 cmpgt_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_andnot_si256(_mm256_cmpeq_epi8(a.m_value, b.m_value), _mm256_cmpeq_epi8(_mm256_max_epu8(a.m_value, b.m_value), a.m_value)) }; }
CPPSPMD_FORCE_INLINE vint16 cmplt_epu8(const vint16& a, const vint16& b) { return vint16{ cmpgt_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint16 absdiff_epu8(const vint16& a, const vint16& b) { return vint16{ _mm256_or_si256(_mm256_subs_epu8(a.m_value, b.m_value), _mm256_subs_epu8(b.m_value, a.m_value)) }; }

CPPSPMD_FORCE_INLINE vint16 blendv_epi8(const vint16& a, const vint16& b, const vint16 &mask) { return vint16{ _mm256_blendv_epi8(a.m_value, b.m_value, mask.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 blendv_epi32(const vint16& a, const vint16& b, const vint16 &mask) { return vint16{ blendv_epi32(a.m_value, b.m_value, mask.m_value) }; }

CPPSPMD_FORCE_INLINE int movemask_epi32(const vint16& a) { return _mm256_movemask_ps(_mm256_castsi256_ps(a.m_value)); }

CPPSPMD_FORCE_INLINE vint16 undefined_vint() { return vint16{ _mm256_undefined_si256() }; }
CPPSPMD_FORCE_INLINE vfloat undefined_vfloat() { return vfloat{ _mm256_undefined_ps(), _mm256_undefined_ps() }; }

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int32's in each 128-bit lane.
#define VINT_LANE_SHUFFLE_EPI32(a, control) vint16(_mm256_shuffle_epi32((a).m_value, control))

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int16's in either the high or low 64-bit lane.
#define VINT_LANE_SHUFFLELO_EPI16(a, control) vint16(_mm256_shufflelo_epi16((a).m_value, control))
#define VINT_LANE_SHUFFLEHI_EPI16(a, control) vint16(_mm256_shufflehi_epi16((a).m_value, control))

#define VINT_LANE_SHUFFLE_MASK(a, b, c, d) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))
#define VINT_LANE_SHUFFLE_MASK_R(d, c, b, a) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))

#define VINT_LANE_SHIFT_LEFT_BYTES(a, l) vint16(_mm256_slli_si256((a).m_value, l))
#define VINT_LANE_SHIFT_RIGHT_BYTES(a, l) vint16(_mm256_srli_si256((a).m_value, l))

// Unpack and interleave 8-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpacklo_epi8(const vint16& a, const vint16& b) { return vint16(_mm256_unpacklo_epi8(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpackhi_epi8(const vint16& a, const vint16& b) { return vint16(_mm256_unpackhi_epi8(a.m_value, b.m_value)); }

// Unpack and interleave 16-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpacklo_epi16(const vint16& a, const vint16& b) { return vint16(_mm256_unpacklo_epi16(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpackhi_epi16(const vint16& a, const vint16& b) { return vint16(_mm256_unpackhi_epi16(a.m_value, b.m_value)); }

// Unpack and interleave 32-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpacklo_epi32(const vint16& a, const vint16& b) { return vint16(_mm256_unpacklo_epi32(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpackhi_epi32(const vint16& a, const vint16& b) { return vint16(_mm256_unpackhi_epi32(a.m_value, b.m_value)); }

// Unpack and interleave 64-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpacklo_epi64(const vint16& a, const vint16& b) { return vint16(_mm256_unpacklo_epi64(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint16 vint_lane_unpackhi_epi64(const vint16& a, const vint16& b) { return vint16(_mm256_unpackhi_epi64(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint16 vint_set1_epi8(int8_t a) { return vint16(_mm256_set1_epi8(a)); }
CPPSPMD_FORCE_INLINE vint16 vint_set1_epi16(int16_t a) { return vint16(_mm256_set1_epi16(a)); }
CPPSPMD_FORCE_INLINE vint16 vint_set1_epi32(int32_t a) { return vint16(_mm256_set1_epi32(a)); }
CPPSPMD_FORCE_INLINE vint16 vint_set1_epi64(int64_t a) { return vint16(_mm256_set1_epi64x(a)); }

CPPSPMD_FORCE_INLINE vint16 add_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_add_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 adds_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_adds_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 adds_epu16(const vint16& a, const vint16& b) { return vint16{ _mm256_adds_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 avg_epu16(const vint16& a, const vint16& b) { return vint16{ _mm256_avg_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 sub_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_sub_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 subs_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_subs_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 subs_epu16(const vint16& a, const vint16& b) { return vint16{ _mm256_subs_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 mullo_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_mullo_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 mulhi_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_mulhi_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 mulhi_epu16(const vint16& a, const vint16& b) { return vint16{ _mm256_mulhi_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 min_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_min_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 max_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_max_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 madd_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_madd_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmpeq_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpeq_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmpgt_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpgt_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 cmplt_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_cmpgt_epi16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 packs_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_packs_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint16 packus_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_packus_epi16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint16 uniform_shift_left_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_sll_epi16(a.m_value, _mm256_castsi256_si128(b.m_value)) }; }
CPPSPMD_FORCE_INLINE vint16 uniform_arith_shift_right_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_sra_epi16(a.m_value, _mm256_castsi256_si128(b.m_value)) }; }
CPPSPMD_FORCE_INLINE vint16 uniform_shift_right_epi16(const vint16& a, const vint16& b) { return vint16{ _mm256_srl_epi16(a.m_value, _mm256_castsi256_si128(b.m_value)) }; }

#define VINT_SHIFT_LEFT_EPI16(a, b) vint16(_mm256_slli_epi16((a).m_value, b))
#define VINT_SHIFT_RIGHT_EPI16(a, b) vint16(_mm256_srai_epi16((a).m_value, b))
#define VUINT_SHIFT_RIGHT_EPI16(a, b) vint16(_mm256_srli_epi16((a).m_value, b))

CPPSPMD_FORCE_INLINE vint16 mul_epu32(const vint16 &a, const vint16& b) { return vint16(_mm256_mul_epu32(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE int16_t safe_div(int16_t a, int16_t b) { return (int16_t)(b ? (a / b) : 0); }
CPPSPMD_FORCE_INLINE int16_t safe_mod(int16_t a, int16_t b) { return (int16_t)(b ? (a % b) : 0); }

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint16 operator/ (const vint16& a, const vint16& b)
{
	CPPSPMD_ALIGN(32) int16_t va[16];
	CPPSPMD_ALIGN(32) int16_t vb[16];
	_mm256_store_si256((__m256i *)va, a.m_value);
	_mm256_store_si256((__m256i *)vb, b.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = safe_div(va[i], vb[i]);

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint16 operator/ (const vint16& a, int b)
{
	CPPSPMD_ALIGN(32) int16_t va[16];
	_mm256_store_si256((__m256i *)va, a.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = safe_div(va[i], (int16_t)b);

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint16 operator% (const vint16& a, const vint16& b)
{
	CPPSPMD_ALIGN(32) int16_t va[16];
	CPPSPMD_ALIGN(32) int16_t vb[16];
	_mm256_store_si256((__m256i *)va, a.m_value);
	_mm256_store_si256((__m256i *)vb, b.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = safe_mod(va[i], vb[i]);

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint16 operator% (const vint16& a, int b)
{
	if (!b)
		return vint16{ _mm256_setzero_si256() };
				
	CPPSPMD_ALIGN(32) int16_t va[16];
	_mm256_store_si256((__m256i *)va, a.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = va[i] % b;

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow
CPPSPMD_FORCE_INLINE vint16 operator<< (const vint16& a, const vint16& b)
{
	CPPSPMD_ALIGN(32) int16_t va[16];
	CPPSPMD_ALIGN(32) int16_t vb[16];
	_mm256_store_si256((__m256i *)va, a.m_value);
	_mm256_store_si256((__m256i *)vb, b.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = va[i] << vb[i];

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// uniform shift left
CPPSPMD_FORCE_INLINE vint16 operator<< (const vint16& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint16{ _mm256_sll_epi16(a.m_value, bv) };
}

// uniform arithmetic shift right
CPPSPMD_FORCE_INLINE vint16 operator>> (const vint16& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint16{ _mm256_sra_epi16(a.m_value, bv) };
}

// uniform shift right
CPPSPMD_FORCE_INLINE vint16 vuint_shift_right(const vint16& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint16{ _mm256_srl_epi16(a.m_value, bv) };
}

// This is very slow
CPPSPMD_FORCE_INLINE vint16 operator>> (const vint16& a, const vint16& b)
{
	CPPSPMD_ALIGN(32) int16_t va[16];
	CPPSPMD_ALIGN(32) int16_t vb[16];
	_mm256_store_si256((__m256i *)va, a.m_value);
	_mm256_store_si256((__m256i *)vb, b.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = va[i] >> vb[i];

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow
CPPSPMD_FORCE_INLINE vint16 vuint_shift_right(const vint16& a, const vint16& b)
{
	CPPSPMD_ALIGN(32) uint16_t va[16];
	CPPSPMD_ALIGN(32) uint16_t vb[16];
	_mm256_store_si256((__m256i *)va, a.m_value);
	_mm256_store_si256((__m256i *)vb, b.m_value);

	CPPSPMD_ALIGN(32) int16_t result[16];
	for (int i = 0; i < 16; i++)
		result[i] = va[i] >> vb[i];

	return vint16{ _mm256_load_si256((__m256i*)result) };
}

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) vint16( _mm256_slli_epi16( (a).m_value, (b) ) ) 
#define VINT_SHIFT_RIGHT(a, b) vint16( _mm256_srai_epi16( (a).m_value, (b) ) )
#define VUINT_SHIFT_RIGHT(a, b) vint16( _mm256_srli_epi16( (a).m_value, (b) ) )
#define VINT_ROT(x, k) (VINT_SHIFT_LEFT((x), (k)) | VUINT_SHIFT_RIGHT((x), 32 - (k)))

CPPSPMD_FORCE_INLINE vbool operator==(const lint16& a, const lint16& b) { return vbool{ _mm256_cmpeq_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const lint16& a, int b) { return vint16(a) == vint16(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint16& b) { return vint16(a) == vint16(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint16& a, const lint16& b) { return vbool{ _mm256_cmpgt_epi16(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const lint16& a, const lint16& b) { return vbool{ _mm256_cmpgt_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const lint16& a, const lint16& b) { return !vbool{ _mm256_cmpgt_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const lint16& a, const lint16& b) { return !vbool{ _mm256_cmpgt_epi16(b.m_value, a.m_value) }; }

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) 
{ 
	assert(instance < 16); 
	CPPSPMD_ALIGN(32) float values[8]; 
	_mm256_store_ps(values, (instance < 8) ? v.m_value_l : v.m_value_h); 
	return values[instance & 7]; 
}

CPPSPMD_FORCE_INLINE int extract(const vint16& v, int instance) 
{ 
	assert(instance < 16); 
	CPPSPMD_ALIGN(32) int16_t values[16]; 
	_mm256_store_si256((__m256i*)values, v.m_value); 
	return values[instance]; 
}

CPPSPMD_FORCE_INLINE int extract(const lint16& v, int instance) 
{ 
	assert(instance < 16);
	CPPSPMD_ALIGN(32) int16_t values[16];
	_mm256_store_si256((__m256i*)values, v.m_value);
	return values[instance];
}

CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) 
{ 
	assert(instance < 16);
	CPPSPMD_ALIGN(32) int16_t values[16];
	_mm256_store_si256((__m256i*)values, v.m_value);
	return values[instance] != 0;
}

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

CPPSPMD_FORCE_INLINE float cast_int_to_float(int i) { return *(const float*)&i; }

#define VINT_EXTRACT(v, instance) _mm256_extract_epi16((v).m_value, instance)
#define VBOOL_EXTRACT(v, instance) _mm256_extract_epi16((v).m_value, instance)
#define VFLOAT_EXTRACT(v, instance) cast_int_to_float(_mm256_extract_epi32(_mm256_castps_si256(((instance) < 8) ? (v).m_value_l : (v).m_value_h), (instance) & 7))

CPPSPMD_FORCE_INLINE vfloat &insert(vfloat& v, int instance, float f)
{
	assert(instance < 16);
	CPPSPMD_ALIGN(32) float values[8];
	_mm256_store_ps(values, (instance < 8) ? v.m_value_l : v.m_value_h);
	values[instance & 7] = f;
	if (instance < 8)
		v.m_value_l = _mm256_load_ps(values);
	else
		v.m_value_h = _mm256_load_ps(values);
	return v;
}

CPPSPMD_FORCE_INLINE vint16 &insert(vint16& v, int instance, int16_t i)
{
	assert(instance < 16);
	CPPSPMD_ALIGN(32) int16_t values[16];
	_mm256_store_si256((__m256i *)values, v.m_value);
	values[instance] = i;
	v.m_value = _mm256_load_si256((__m256i *)values);
	return v;
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

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	int *pDstI = (int *)dst.m_pValue;

	pDstI[_mm256_extract_epi16(dst.m_vindex, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 4)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 5)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 6)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 7)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 8)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 9)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 10)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 11)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 12)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 13)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 14)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 15)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref& dst, const vfloat& src)
{
	uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

	if (mask == ALL_ON_MOVEMASK)
		return store_all(dst, src);
	
	int *pDstI = (int *)dst.m_pValue;

	if (mask & (1 << 0)) pDstI[_mm256_extract_epi16(dst.m_vindex, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
	if (mask & (1 << 2)) pDstI[_mm256_extract_epi16(dst.m_vindex, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
	if (mask & (1 << 4)) pDstI[_mm256_extract_epi16(dst.m_vindex, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
	if (mask & (1 << 6)) pDstI[_mm256_extract_epi16(dst.m_vindex, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

	if (mask & (1 << 8)) pDstI[_mm256_extract_epi16(dst.m_vindex, 4)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
	if (mask & (1 << 10)) pDstI[_mm256_extract_epi16(dst.m_vindex, 5)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
	if (mask & (1 << 12)) pDstI[_mm256_extract_epi16(dst.m_vindex, 6)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
	if (mask & (1 << 14)) pDstI[_mm256_extract_epi16(dst.m_vindex, 7)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

	if (mask & (1 << 16)) pDstI[_mm256_extract_epi16(dst.m_vindex, 8)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
	if (mask & (1 << 18)) pDstI[_mm256_extract_epi16(dst.m_vindex, 9)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
	if (mask & (1 << 20)) pDstI[_mm256_extract_epi16(dst.m_vindex, 10)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
	if (mask & (1 << 22)) pDstI[_mm256_extract_epi16(dst.m_vindex, 11)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

	if (mask & (1 << 24)) pDstI[_mm256_extract_epi16(dst.m_vindex, 12)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
	if (mask & (1 << 26)) pDstI[_mm256_extract_epi16(dst.m_vindex, 13)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
	if (mask & (1 << 28)) pDstI[_mm256_extract_epi16(dst.m_vindex, 14)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
	if (mask & (1 << 30)) pDstI[_mm256_extract_epi16(dst.m_vindex, 15)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);

	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	int *pDstI = (int *)dst.m_pValue;

	pDstI[_mm256_extract_epi16(dst.m_vindex, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 4)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 5)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 6)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 7)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 8)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 9)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 10)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 11)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

	pDstI[_mm256_extract_epi16(dst.m_vindex, 12)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 13)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 14)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
	pDstI[_mm256_extract_epi16(dst.m_vindex, 15)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	uint32_t mask = _mm256_movemask_epi8(m_exec.m_mask);

	if (mask == ALL_ON_MOVEMASK)
		return store_all(dst, src);

	int *pDstI = (int *)dst.m_pValue;

	if (mask & (1 << 0)) pDstI[_mm256_extract_epi16(dst.m_vindex, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 0);
	if (mask & (1 << 2)) pDstI[_mm256_extract_epi16(dst.m_vindex, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 1);
	if (mask & (1 << 4)) pDstI[_mm256_extract_epi16(dst.m_vindex, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 2);
	if (mask & (1 << 6)) pDstI[_mm256_extract_epi16(dst.m_vindex, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 3);

	if (mask & (1 << 8)) pDstI[_mm256_extract_epi16(dst.m_vindex, 4)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 4);
	if (mask & (1 << 10)) pDstI[_mm256_extract_epi16(dst.m_vindex, 5)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 5);
	if (mask & (1 << 12)) pDstI[_mm256_extract_epi16(dst.m_vindex, 6)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 6);
	if (mask & (1 << 14)) pDstI[_mm256_extract_epi16(dst.m_vindex, 7)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_l), 7);

	if (mask & (1 << 16)) pDstI[_mm256_extract_epi16(dst.m_vindex, 8)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 0);
	if (mask & (1 << 18)) pDstI[_mm256_extract_epi16(dst.m_vindex, 9)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 1);
	if (mask & (1 << 20)) pDstI[_mm256_extract_epi16(dst.m_vindex, 10)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 2);
	if (mask & (1 << 22)) pDstI[_mm256_extract_epi16(dst.m_vindex, 11)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 3);

	if (mask & (1 << 24)) pDstI[_mm256_extract_epi16(dst.m_vindex, 12)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 4);
	if (mask & (1 << 26)) pDstI[_mm256_extract_epi16(dst.m_vindex, 13)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 5);
	if (mask & (1 << 28)) pDstI[_mm256_extract_epi16(dst.m_vindex, 14)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 6);
	if (mask & (1 << 30)) pDstI[_mm256_extract_epi16(dst.m_vindex, 15)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value_h), 7);

	return dst;
}

#include "cppspmd_flow.h"

} // namespace cppspmd_int16_avx2_fma

