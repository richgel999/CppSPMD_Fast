// cppspmd_avx1.h
// AVX1 support.
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
#define CPPSPMD_AVX2 0
#define CPPSPMD_INT16 0

#define CPPSPMD cppspmd_avx1_alt
#define CPPSPMD_ARCH _avx1_alt
#define CPPSPMD_AVX1 1

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
#define CPPSPMD_ALIGNMENT (32)

namespace CPPSPMD
{

const int PROGRAM_COUNT_SHIFT = 3;
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

CPPSPMD_FORCE_INLINE __m256i compare_eq_epi32(__m256i a, __m256i b)
{
	return combine_i(_mm_cmpeq_epi32(get_lo_i(a), get_lo_i(b)), _mm_cmpeq_epi32(get_hi_i(a), get_hi_i(b)));
}

CPPSPMD_FORCE_INLINE __m256i and_si256(__m256i a, __m256i b)
{
	return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

CPPSPMD_FORCE_INLINE __m256i or_si256(__m256i a, __m256i b)
{
	return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

CPPSPMD_FORCE_INLINE __m256i xor_si256(__m256i a, __m256i b)
{
	return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

CPPSPMD_FORCE_INLINE __m256i andnot_si256(__m256i a, __m256i b)
{
	return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

CPPSPMD_FORCE_INLINE __m128i _mm_blendv_epi32(__m128i a, __m128i b, __m128i c) { return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _mm_castsi128_ps(c))); }

CPPSPMD_FORCE_INLINE __m128i mulhi_epu32(__m128i a, __m128i b)
{
	__m128i tmp1 = _mm_mul_epu32(a, b);
	__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
	return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 3, 1)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 3, 1)));
}

CPPSPMD_FORCE_INLINE __m256i mulhi_epu32(__m256i a, __m256i b)
{
	return combine_i(mulhi_epu32(get_lo_i(a), get_lo_i(b)), mulhi_epu32(get_hi_i(a), get_hi_i(b)));
}

const uint32_t ALL_ON_MOVEMASK = 0xFF;

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
		__m256i m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const __m256i& mask) : m_mask(mask) { }

		CPPSPMD_FORCE_INLINE void enable_lane(uint32_t lane) { m_mask = _mm256_load_si256((const __m256i *)&g_lane_masks_256[lane][0]); }
				
		static CPPSPMD_FORCE_INLINE exec_mask all_on()	{ return exec_mask{ _mm256_load_si256((const __m256i*)g_allones_256) };	}
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ _mm256_setzero_si256() }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(m_mask)); }
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
		CPPSPMD_FORCE_INLINE explicit operator vint() const;
								
	private:
		vbool& operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst.m_value), _mm256_castsi256_ps(src.m_value), _mm256_castsi256_ps(m_exec.m_mask)));
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
		__m256 m_value;

		vfloat() = default;

		CPPSPMD_FORCE_INLINE explicit vfloat(const __m256& v) : m_value(v) { }

		CPPSPMD_FORCE_INLINE vfloat(float value) : m_value(_mm256_set1_ps(value)) { }

		CPPSPMD_FORCE_INLINE explicit vfloat(int value) : m_value(_mm256_set1_ps((float)value)) { }

	private:
		vfloat& operator=(const vfloat&);
	};

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat& dst, const vfloat& src)
	{
		dst.m_value = _mm256_blendv_ps(dst.m_value, src.m_value, _mm256_castsi256_ps(m_exec.m_mask));
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat&& dst, const vfloat& src)
	{
		dst.m_value = _mm256_blendv_ps(dst.m_value, src.m_value, _mm256_castsi256_ps(m_exec.m_mask));
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
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm256_storeu_ps(dst.m_pValue, src.m_value);
		else
			_mm256_maskstore_ps(dst.m_pValue, m_exec.m_mask, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref&& dst, const vfloat& src)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm256_storeu_ps(dst.m_pValue, src.m_value);
		else
			_mm256_maskstore_ps(dst.m_pValue, m_exec.m_mask, src.m_value);
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref& dst, const vfloat& src)
	{
		_mm256_storeu_ps(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref&& dst, const vfloat& src)
	{
		_mm256_storeu_ps(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const float_lref& src)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			return vfloat{ _mm256_loadu_ps(src.m_pValue) };
		else
			return vfloat{ _mm256_maskload_ps(src.m_pValue, m_exec.m_mask) };
	}
		
	// Varying ref to floats
	struct float_vref
	{
		__m128i m_vindex_l;
		__m128i m_vindex_h;
		float* m_pValue;
		
	private:
		float_vref& operator=(const float_vref&);
	};

	// Varying ref to varying float
	struct vfloat_vref
	{
		__m128i m_vindex_l;
		__m128i m_vindex_h;
		vfloat* m_pValue;
		
	private:
		vfloat_vref& operator=(const vfloat_vref&);
	};

	// Varying ref to varying int
	struct vint_vref
	{
		__m128i m_vindex_l;
		__m128i m_vindex_h;
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
		CPPSPMD_ALIGN(32) float loaded[8];

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		if (mask & 1) loaded[0] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)];
		if (mask & 2) loaded[1] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)];
		if (mask & 4) loaded[2] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)];
		if (mask & 8) loaded[3] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)];

		if (mask & 16) loaded[4] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)];
		if (mask & 32) loaded[5] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)];
		if (mask & 64) loaded[6] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)];
		if (mask & 128) loaded[7] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)];
		
		return vfloat{ _mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)loaded)) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
		CPPSPMD_ALIGN(32) float loaded[8];

		const float *pSrc = src.m_pValue;

		loaded[0] = pSrc[_mm_extract_epi32(src.m_vindex_l, 0)];
		loaded[1] = pSrc[_mm_extract_epi32(src.m_vindex_l, 1)];
		loaded[2] = pSrc[_mm_extract_epi32(src.m_vindex_l, 2)];
		loaded[3] = pSrc[_mm_extract_epi32(src.m_vindex_l, 3)];

		loaded[4] = pSrc[_mm_extract_epi32(src.m_vindex_h, 0)];
		loaded[5] = pSrc[_mm_extract_epi32(src.m_vindex_h, 1)];
		loaded[6] = pSrc[_mm_extract_epi32(src.m_vindex_h, 2)];
		loaded[7] = pSrc[_mm_extract_epi32(src.m_vindex_h, 3)];

		return vfloat{ _mm256_load_ps((const float*)loaded) };
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
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm_storeu_si128((__m128i *)dst.m_pValue, src.m_value_l);
			_mm_storeu_si128(((__m128i *)dst.m_pValue) + 1, src.m_value_h);
		}
		else
		{
			if (mask & 1) dst.m_pValue[0] = _mm_extract_epi32(src.m_value_l, 0);
			if (mask & 2) dst.m_pValue[1] = _mm_extract_epi32(src.m_value_l, 1);
			if (mask & 4) dst.m_pValue[2] = _mm_extract_epi32(src.m_value_l, 2);
			if (mask & 8) dst.m_pValue[3] = _mm_extract_epi32(src.m_value_l, 3);

			if (mask & 16) dst.m_pValue[4] = _mm_extract_epi32(src.m_value_h, 0);
			if (mask & 32) dst.m_pValue[5] = _mm_extract_epi32(src.m_value_h, 1);
			if (mask & 64) dst.m_pValue[6] = _mm_extract_epi32(src.m_value_h, 2);
			if (mask & 128) dst.m_pValue[7] = _mm_extract_epi32(src.m_value_h, 3);
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_lref& src)
	{
		__m256i v = _mm256_loadu_si256((const __m256i*)src.m_pValue);

		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}

	// Linear ref to int16's
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};

	// TODO: This surely could be improved.
	CPPSPMD_FORCE_INLINE const int16_lref& store(const int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(32) int stored[8];
		_mm_store_si128(((__m128i*)stored), src.m_value_l);
		_mm_store_si128(((__m128i*)stored) + 1, src.m_value_h);

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		for (int i = 0; i < 8; i++)
		{
			if (mask & (1 << i))
				dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_lref& store_all(const int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(32) int stored[8];
		_mm_store_si128(((__m128i*)stored), src.m_value_l);
		_mm_store_si128(((__m128i*)stored) + 1, src.m_value_h);

		for (int i = 0; i < 8; i++)
			dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		return dst;
	}
		
	CPPSPMD_FORCE_INLINE vint load(const int16_lref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		for (int i = 0; i < 8; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		__m256i t = _mm256_load_si256( (const __m256i *)values );

		__m256i v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps( t ), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const int16_lref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		for (int i = 0; i < 8; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		__m128i l = _mm_load_si128( (const __m128i *)values );
		__m128i h = _mm_load_si128( ((const __m128i *)values) + 1);

		return vint{ l, h };
	}

	// Linear ref to a constant int's
	struct cint_lref
	{
		const int* m_pValue;

	private:
		cint_lref& operator=(const cint_lref&);
	};

	CPPSPMD_FORCE_INLINE vint load(const cint_lref& src)
	{
		__m256i v = _mm256_loadu_si256((const __m256i*)src.m_pValue);

		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_lref& src)
	{
		return vint{ _mm_loadu_si128((const __m128i*)src.m_pValue), _mm_loadu_si128( ((const __m128i*)src.m_pValue) + 1) };
	}
	
	// Varying ref to ints
	struct int_vref
	{
		__m128i m_vindex_l;
		__m128i m_vindex_h;
		int* m_pValue;

	private:
		int_vref& operator=(const int_vref&);
	};

	// Varying ref to constant ints
	struct cint_vref
	{
		__m128i m_vindex_l;
		__m128i m_vindex_h;
		const int* m_pValue;

	private:
		cint_vref& operator=(const cint_vref&);
	};

	// Varying int
	struct vint
	{
		// Two __m128i's, not a __m256i, because AVX1 doesn't have hardly any integer ops (AVX2 does). The compiler will put these in registers.
		__m128i m_value_l;
		__m128i m_value_h;

		vint() = default;

		//CPPSPMD_FORCE_INLINE explicit vint(const __m256i& value) : m_value_l(get_lo_i(value)), m_value_h(get_hi_i(value))	{ }
		
		CPPSPMD_FORCE_INLINE explicit vint(const __m128i& l, const __m128i &h) : m_value_l(l), m_value_h(h)	{ }
				
		CPPSPMD_FORCE_INLINE vint(int value)  { __m128i k = _mm_set1_epi32(value); m_value_l = k; m_value_h = k; }
		
		CPPSPMD_FORCE_INLINE explicit vint(float value) { __m128i k = _mm_set1_epi32((int)value); m_value_l = k; m_value_h = k; }

		CPPSPMD_FORCE_INLINE explicit vint(const vfloat& other) { __m256i k = _mm256_cvttps_epi32(other.m_value); m_value_l = get_lo_i(k); m_value_h = get_hi_i(k); }
				
		CPPSPMD_FORCE_INLINE explicit operator vbool() const { return vbool{ xor_si256( _mm256_load_si256((const __m256i*)g_allones_256), compare_eq_epi32(get_value_256i(), _mm256_setzero_si256())) }; }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const	{ return vfloat{ _mm256_cvtepi32_ps(get_value_256i()) }; }

		CPPSPMD_FORCE_INLINE int_vref operator[](int* ptr) const
		{
			return int_vref{ m_value_l, m_value_h, ptr };
		}

		CPPSPMD_FORCE_INLINE cint_vref operator[](const int* ptr) const
		{
			return cint_vref{ m_value_l, m_value_h, ptr };
		}

		CPPSPMD_FORCE_INLINE float_vref operator[](float* ptr) const
		{
			return float_vref{ m_value_l, m_value_h, ptr };
		}

		CPPSPMD_FORCE_INLINE vfloat_vref operator[](vfloat* ptr) const
		{
			return vfloat_vref{ m_value_l, m_value_h, ptr };
		}

		CPPSPMD_FORCE_INLINE vint_vref operator[](vint* ptr) const
		{
			return vint_vref{ m_value_l, m_value_h, ptr };
		}

		CPPSPMD_FORCE_INLINE __m256i get_value_256i() const { return combine_i(m_value_l, m_value_h); }

	private:
		vint& operator=(const vint&);
	};

	// load/store linear int
	CPPSPMD_FORCE_INLINE void storeu_linear(int *pDst, const vint& src)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm_storeu_si128((__m128i*)pDst, src.m_value_l);
			_mm_storeu_si128(((__m128i*)pDst) + 1, src.m_value_h);
		}
		else
		{
			if (mask & 1) pDst[0] = _mm_extract_epi32(src.m_value_l, 0);
			if (mask & 2) pDst[1] = _mm_extract_epi32(src.m_value_l, 1);
			if (mask & 4) pDst[2] = _mm_extract_epi32(src.m_value_l, 2);
			if (mask & 8) pDst[3] = _mm_extract_epi32(src.m_value_l, 3);

			if (mask & 16) pDst[4] = _mm_extract_epi32(src.m_value_h, 0);
			if (mask & 32) pDst[5] = _mm_extract_epi32(src.m_value_h, 1);
			if (mask & 64) pDst[6] = _mm_extract_epi32(src.m_value_h, 2);
			if (mask & 128) pDst[7] = _mm_extract_epi32(src.m_value_h, 3);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int *pDst, const vint& src)
	{
		_mm_storeu_si128((__m128i*)pDst, src.m_value_l);
		_mm_storeu_si128(((__m128i*)pDst) + 1, src.m_value_h);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int *pDst, const vint& src)
	{
		_mm_store_si128(((__m128i*)pDst), src.m_value_l);
		_mm_store_si128(((__m128i*)pDst) + 1, src.m_value_h);
	}
	
	CPPSPMD_FORCE_INLINE vint loadu_linear(const int *pSrc)
	{
		__m256i v = _mm256_loadu_si256((const __m256i*)pSrc);

		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear_all(const int *pSrc)
	{
		return vint{ _mm_loadu_si128((__m128i*)pSrc), _mm_loadu_si128(((__m128i*)pSrc) + 1) };
	}

	CPPSPMD_FORCE_INLINE vint load_linear_all(const int *pSrc)
	{
		return vint{ _mm_load_si128((__m128i*)pSrc), _mm_load_si128(((__m128i*)pSrc) + 1) };
	}

	// load/store linear float
	CPPSPMD_FORCE_INLINE void storeu_linear(float *pDst, const vfloat& src)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_ps((float*)pDst, src.m_value);
		}
		else
		{
			int *pDstI = (int *)pDst;
			if (mask & 1) pDstI[0] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
			if (mask & 2) pDstI[1] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
			if (mask & 4) pDstI[2] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
			if (mask & 8) pDstI[3] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

			if (mask & 16) pDstI[4] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
			if (mask & 32) pDstI[5] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
			if (mask & 64) pDstI[6] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
			if (mask & 128) pDstI[7] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(float *pDst, const vfloat& src)
	{
		_mm256_storeu_ps((float*)pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(float *pDst, const vfloat& src)
	{
		_mm256_store_ps((float*)pDst, src.m_value);
	}
	
	CPPSPMD_FORCE_INLINE vfloat loadu_linear(const float *pSrc)
	{
		__m256 v = _mm256_loadu_ps((const float*)pSrc);

		v = _mm256_and_ps(v, _mm256_castsi256_ps(m_exec.m_mask));

		return vfloat{ v };
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear_all(const float *pSrc)
	{
		return vfloat{ _mm256_loadu_ps((float*)pSrc) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_linear_all(const float *pSrc)
	{
		return vfloat{ _mm256_load_ps((float*)pSrc) };
	}
		
	CPPSPMD_FORCE_INLINE vint& store(vint& dst, const vint& src)
	{
		dst.m_value_l = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(dst.m_value_l), _mm_castsi128_ps(src.m_value_l), _mm_castsi128_ps(get_lo_i(m_exec.m_mask))));
		dst.m_value_h = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(dst.m_value_h), _mm_castsi128_ps(src.m_value_h), _mm_castsi128_ps(get_hi_i(m_exec.m_mask))));
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store(const int_vref& dst, const vint& src)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm_extract_epi32(src.m_value_l, 0);
		if (mask & 2) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm_extract_epi32(src.m_value_l, 1);
		if (mask & 4) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm_extract_epi32(src.m_value_l, 2);
		if (mask & 8) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm_extract_epi32(src.m_value_l, 3);
		
		if (mask & 16) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm_extract_epi32(src.m_value_h, 0);
		if (mask & 32) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm_extract_epi32(src.m_value_h, 1);
		if (mask & 64) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm_extract_epi32(src.m_value_h, 2);
		if (mask & 128) dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm_extract_epi32(src.m_value_h, 3);

		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vint& store_all(vint& dst, const vint& src)
	{
		dst.m_value_l = src.m_value_l;
		dst.m_value_h = src.m_value_h;
		return dst;
	}
				
	CPPSPMD_FORCE_INLINE const int_vref& store_all(const int_vref& dst, const vint& src)
	{
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm_extract_epi32(src.m_value_l, 0);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm_extract_epi32(src.m_value_l, 1);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm_extract_epi32(src.m_value_l, 2);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm_extract_epi32(src.m_value_l, 3);
		
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm_extract_epi32(src.m_value_h, 0);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm_extract_epi32(src.m_value_h, 1);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm_extract_epi32(src.m_value_h, 2);
		dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm_extract_epi32(src.m_value_h, 3);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_vref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) values[0] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)];
		if (mask & 2) values[1] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)];
		if (mask & 4) values[2] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)];
		if (mask & 8) values[3] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)];

		if (mask & 16) values[4] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)];
		if (mask & 32) values[5] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)];
		if (mask & 64) values[6] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)];
		if (mask & 128) values[7] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)];

		__m256i v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const int_vref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		values[0] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)];
		values[1] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)];
		values[2] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)];
		values[3] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)];

		values[4] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)];
		values[5] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)];
		values[6] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)];
		values[7] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)];

		__m256i v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}
		
	CPPSPMD_FORCE_INLINE vint load(const cint_vref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) values[0] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)];
		if (mask & 2) values[1] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)];
		if (mask & 4) values[2] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)];
		if (mask & 8) values[3] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)];

		if (mask & 16) values[4] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)];
		if (mask & 32) values[5] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)];
		if (mask & 64) values[6] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)];
		if (mask & 128) values[7] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)];

		__m256i v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const cint_vref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		values[0] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)];
		values[1] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)];
		values[2] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)];
		values[3] = src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)];

		values[4] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)];
		values[5] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)];
		values[6] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)];
		values[7] = src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)];

		__m256i v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values)));

		return vint{ get_lo_i(v), get_hi_i(v) };
	}

	CPPSPMD_FORCE_INLINE vint load_bytes_all(const cint_vref& src)
	{
		const uint8_t* pSrc = (const uint8_t*)src.m_pValue;
		__m128i v0_l = _mm_insert_epi32(_mm_undefined_si128(), ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_l, 0)))[0], 0);
		v0_l = _mm_insert_epi32(v0_l, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_l, 1)))[0], 1);
		v0_l = _mm_insert_epi32(v0_l, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_l, 2)))[0], 2);
		v0_l = _mm_insert_epi32(v0_l, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_l, 3)))[0], 3);

		__m128i v0_h = _mm_insert_epi32(_mm_undefined_si128(), ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_h, 0)))[0], 0);
		v0_h = _mm_insert_epi32(v0_h, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_h, 1)))[0], 1);
		v0_h = _mm_insert_epi32(v0_h, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_h, 2)))[0], 2);
		v0_h = _mm_insert_epi32(v0_h, ((int*)(pSrc + _mm_extract_epi32(src.m_vindex_h, 3)))[0], 3);

		return vint{ v0_l, v0_h };
	}

	CPPSPMD_FORCE_INLINE vint load_words_all(const cint_vref& src)
	{
		const uint8_t* pSrc = (const uint8_t*)src.m_pValue;
		__m128i v0_l = _mm_insert_epi32(_mm_undefined_si128(), ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_l, 0)))[0], 0);
		v0_l = _mm_insert_epi32(v0_l, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_l, 1)))[0], 1);
		v0_l = _mm_insert_epi32(v0_l, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_l, 2)))[0], 2);
		v0_l = _mm_insert_epi32(v0_l, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_l, 3)))[0], 3);

		__m128i v0_h = _mm_insert_epi32(_mm_undefined_si128(), ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_h, 0)))[0], 0);
		v0_h = _mm_insert_epi32(v0_h, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_h, 1)))[0], 1);
		v0_h = _mm_insert_epi32(v0_h, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_h, 2)))[0], 2);
		v0_h = _mm_insert_epi32(v0_h, ((int16_t*)(pSrc + 2 * _mm_extract_epi32(src.m_vindex_h, 3)))[0], 3);

		return vint{ v0_l, v0_h };
	}

	CPPSPMD_FORCE_INLINE void store_strided(int *pDst, uint32_t stride, const vint &v)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) pDst[0] = _mm_extract_epi32(v.m_value_l, 0);
		if (mask & 2) pDst[stride] = _mm_extract_epi32(v.m_value_l, 1);
		if (mask & 4) pDst[stride*2] = _mm_extract_epi32(v.m_value_l, 2);
		if (mask & 8) pDst[stride*3] = _mm_extract_epi32(v.m_value_l, 3);

		if (mask & 16) pDst[stride*4] = _mm_extract_epi32(v.m_value_h, 0);
		if (mask & 32) pDst[stride*5] = _mm_extract_epi32(v.m_value_h, 1);
		if (mask & 64) pDst[stride*6] = _mm_extract_epi32(v.m_value_h, 2);
		if (mask & 128) pDst[stride*7] = _mm_extract_epi32(v.m_value_h, 3);
	}

	CPPSPMD_FORCE_INLINE void store_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		int *pDst = (int *)pDstF;
		
		if (mask & 1) pDst[0] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 0);
		if (mask & 2) pDst[stride] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 1);
		if (mask & 4) pDst[stride*2] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 2);
		if (mask & 8) pDst[stride*3] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 3);
		if (mask & 16) pDst[stride*4] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 4);
		if (mask & 32) pDst[stride*5] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 5);
		if (mask & 64) pDst[stride*6] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 6);
		if (mask & 128) pDst[stride*7] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 7);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(int *pDst, uint32_t stride, const vint &v)
	{
		pDst[0] = _mm_extract_epi32(v.m_value_l, 0);
		pDst[stride] = _mm_extract_epi32(v.m_value_l, 1);
		pDst[stride*2] = _mm_extract_epi32(v.m_value_l, 2);
		pDst[stride*3] = _mm_extract_epi32(v.m_value_l, 3);

		pDst[stride*4] = _mm_extract_epi32(v.m_value_h, 0);
		pDst[stride*5] = _mm_extract_epi32(v.m_value_h, 1);
		pDst[stride*6] = _mm_extract_epi32(v.m_value_h, 2);
		pDst[stride*7] = _mm_extract_epi32(v.m_value_h, 3);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		int *pDst = (int *)pDstF;
		
		pDst[0] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 0);
		pDst[stride] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 1);
		pDst[stride*2] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 2);
		pDst[stride*3] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 3);

		pDst[stride*4] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 4);
		pDst[stride*5] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 5);
		pDst[stride*6] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 6);
		pDst[stride*7] = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), 7);
	}

	CPPSPMD_FORCE_INLINE vint load_strided(const int *pSrc, uint32_t stride)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		__m128i l = _mm_castps_si128(_mm_setzero_ps());
		__m128i h = _mm_castps_si128(_mm_setzero_ps());

		if (mask & 1) l = _mm_castps_si128(_mm_load_ss( (const float *)pSrc ));
		if (mask & 2) l = _mm_insert_epi32(l, pSrc[stride], 1);
		if (mask & 4) l = _mm_insert_epi32(l, pSrc[stride*2], 2);
		if (mask & 8) l = _mm_insert_epi32(l, pSrc[stride*3], 3);
		
		if (mask & 16) h = _mm_castps_si128(_mm_load_ss( (const float *)pSrc + stride*4 ));
		if (mask & 32) h = _mm_insert_epi32(h, pSrc[stride*5], 1);
		if (mask & 64) h = _mm_insert_epi32(h, pSrc[stride*6], 2);
		if (mask & 128) h = _mm_insert_epi32(h, pSrc[stride*7], 3);

		return vint{ l, h };
	}

	CPPSPMD_FORCE_INLINE vfloat load_strided(const float *pSrc, uint32_t stride)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		__m256i v = _mm256_castps_si256(_mm256_setzero_ps());

		const int *pSrcI = (const int *)pSrc;

		if (mask & 1) v = _mm256_castsi128_si256(_mm_castps_si128( _mm_load_ss( pSrc ) ));
		if (mask & 2) v = _mm256_insert_epi32(v, pSrcI[stride], 1);
		if (mask & 4) v = _mm256_insert_epi32(v, pSrcI[stride*2], 2);
		if (mask & 8) v = _mm256_insert_epi32(v, pSrcI[stride*3], 3);

		if (mask & 16) v = _mm256_insert_epi32(v, pSrcI[stride*4], 4);
		if (mask & 32) v = _mm256_insert_epi32(v, pSrcI[stride*5], 5);
		if (mask & 64) v = _mm256_insert_epi32(v, pSrcI[stride*6], 6);
		if (mask & 128) v = _mm256_insert_epi32(v, pSrcI[stride*7], 7);

		return vfloat{ _mm256_castsi256_ps(v) };
	}

	CPPSPMD_FORCE_INLINE vint load_all_strided(const int *pSrc, uint32_t stride)
	{
		__m128i l, h;

		l = _mm_castps_si128(_mm_load_ss( (const float *)pSrc ));
		l = _mm_insert_epi32(l, pSrc[stride], 1);
		l = _mm_insert_epi32(l, pSrc[stride*2], 2);
		l = _mm_insert_epi32(l, pSrc[stride*3], 3);
		
		h = _mm_castps_si128(_mm_load_ss( (const float *)pSrc + stride*4 ));
		h = _mm_insert_epi32(h, pSrc[stride*5], 1);
		h = _mm_insert_epi32(h, pSrc[stride*6], 2);
		h = _mm_insert_epi32(h, pSrc[stride*7], 3);

		return vint{ l, h };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all_strided(const float *pSrc, uint32_t stride)
	{
		__m256i v;

		const int *pSrcI = (const int *)pSrc;

		v = _mm256_castsi128_si256(_mm_castps_si128( _mm_load_ss( pSrc ) ));
		v = _mm256_insert_epi32(v, pSrcI[stride], 1);
		v = _mm256_insert_epi32(v, pSrcI[stride*2], 2);
		v = _mm256_insert_epi32(v, pSrcI[stride*3], 3);

		v = _mm256_insert_epi32(v, pSrcI[stride*4], 4);
		v = _mm256_insert_epi32(v, pSrcI[stride*5], 5);
		v = _mm256_insert_epi32(v, pSrcI[stride*6], 6);
		v = _mm256_insert_epi32(v, pSrcI[stride*7], 7);

		return vfloat{ _mm256_castsi256_ps(v) };
	}
		
	CPPSPMD_FORCE_INLINE const vfloat_vref& store(const vfloat_vref& dst, const vfloat& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 0)]))[0] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 1)]))[1] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 2)]))[2] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 3)]))[3] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

		if (mask & 16) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 0)]))[4] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
		if (mask & 32) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 1)]))[5] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
		if (mask & 64) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 2)]))[6] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
		if (mask & 128) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 3)]))[7] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		__m256i k = _mm256_setzero_si256();

		if (mask & 1) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)]))[0], 0);
		if (mask & 2) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)]))[1], 1);
		if (mask & 4) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)]))[2], 2);
		if (mask & 8) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)]))[3], 3);

		if (mask & 16) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)]))[4], 4);
		if (mask & 32) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)]))[5], 5);
		if (mask & 64) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)]))[6], 6);
		if (mask & 128) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)]))[7], 7);

		return vfloat{ _mm256_castsi256_ps(k) };
	}

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 0)]))[0] = _mm_extract_epi32(src.m_value_l, 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 1)]))[1] = _mm_extract_epi32(src.m_value_l, 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 2)]))[2] = _mm_extract_epi32(src.m_value_l, 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_l, 3)]))[3] = _mm_extract_epi32(src.m_value_l, 3);

		if (mask & 16) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 0)]))[4] = _mm_extract_epi32(src.m_value_h, 0);
		if (mask & 32) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 1)]))[5] = _mm_extract_epi32(src.m_value_h, 1);
		if (mask & 64) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 2)]))[6] = _mm_extract_epi32(src.m_value_h, 2);
		if (mask & 128) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex_h, 3)]))[7] = _mm_extract_epi32(src.m_value_h, 3);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const vint_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		__m256i k = _mm256_setzero_si256();

		if (mask & 1) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 0)]))[0], 0);
		if (mask & 2) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 1)]))[1], 1);
		if (mask & 4) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 2)]))[2], 2);
		if (mask & 8) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_l, 3)]))[3], 3);

		if (mask & 16) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 0)]))[4], 4);
		if (mask & 32) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 1)]))[5], 5);
		if (mask & 64) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 2)]))[6], 6);
		if (mask & 128) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex_h, 3)]))[7], 7);

		return vint{ get_lo_i(k), get_hi_i(k) };
	}
	
	// Linear integer
	struct lint
	{
		__m128i m_value_l;
		__m128i m_value_h;
						
		CPPSPMD_FORCE_INLINE explicit lint(const __m128i& l, const __m128i &h) : m_value_l(l), m_value_h(h)	{ }
				
		CPPSPMD_FORCE_INLINE explicit operator vfloat() const	{ return vfloat{ _mm256_cvtepi32_ps(get_value_256i()) }; }

		CPPSPMD_FORCE_INLINE explicit operator vint() const {	return vint{ m_value_l, m_value_h }; }

		int get_first_value() const { return _mm_cvtsi128_si32(m_value_l); }

		CPPSPMD_FORCE_INLINE float_lref operator[](float* ptr) const { return float_lref{ ptr + get_first_value() }; }

		CPPSPMD_FORCE_INLINE int_lref operator[](int* ptr) const { return int_lref{ ptr + get_first_value() }; }

		CPPSPMD_FORCE_INLINE int16_lref operator[](int16_t* ptr) const { return int16_lref{ ptr + get_first_value() }; }

		CPPSPMD_FORCE_INLINE cint_lref operator[](const int* ptr) const { return cint_lref{ ptr + get_first_value() }; }

		CPPSPMD_FORCE_INLINE __m256i get_value_256i() const { return combine_i(m_value_l, m_value_h); }

	private:
		lint& operator=(const lint&);
	};

	CPPSPMD_FORCE_INLINE lint& store_all(lint& dst, const lint& src)
	{
		dst.m_value_l = src.m_value_l;
		dst.m_value_h = src.m_value_h;
		return dst;
	}
	
	const lint program_index = lint{ _mm_set_epi32(3, 2, 1, 0), _mm_set_epi32(7, 6, 5, 4) };
	
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

	CPPSPMD_FORCE_INLINE float reduce_add(vfloat v)	
	{ 
		__m256 k = _mm256_blendv_ps(_mm256_setzero_ps(), v.m_value, _mm256_castsi256_ps(m_exec.m_mask));

#if 0
		__m256 a = _mm256_hadd_ps(k, k);
		__m256 b = _mm256_hadd_ps(a, a);
		
		__m128 c = _mm_add_ps(get_lo(b), get_hi(b));
		
		float d;
		_MM_EXTRACT_FLOAT(d, c, 0);
		return d;
#else
		// See https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
		// hiQuad = ( x7, x6, x5, x4 )
		const __m128 hiQuad = _mm256_extractf128_ps(k, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const __m128 loQuad = _mm256_castps256_ps128(k);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const __m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const __m128 sumDual = _mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const __m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const __m128 sum = _mm_add_ss(lo, hi);
		return _mm_cvtss_f32(sum);
#endif
	}

	CPPSPMD_FORCE_INLINE void swap(vint &a, vint &b) { vint temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vfloat &a, vfloat &b) { vfloat temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vbool &a, vbool &b) { vbool temp = a; store(a, b); store(b, temp); }

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
	return vfloat { _mm256_and_ps( _mm256_castsi256_ps(m_value), *(const __m256 *)g_onef_256 ) }; 
}

// Returns UINT32_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint() const 
{ 
	return vint { get_lo_i(m_value), get_hi_i(m_value) };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
	return vbool{ _mm256_castps_si256(_mm256_xor_ps(_mm256_load_ps((const float*)g_allones_256), _mm256_castsi256_ps(v.m_value))) };
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ xor_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) {	return exec_mask{ and_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ or_si256(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_mask)) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_mask)) != 0; }

// Bad pattern - doesn't factor in the current exec mask. Prefer spmd_any() instead.
CPPSPMD_FORCE_INLINE bool all(const vbool& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_value)) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const vbool& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_value)) != 0; }

CPPSPMD_FORCE_INLINE exec_mask andnot(const exec_mask& a, const exec_mask& b) { return exec_mask{ andnot_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE vbool operator||(const vbool& a, const vbool& b) { return vbool{ or_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator&&(const vbool& a, const vbool& b) { return vbool{ and_si256(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_add_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_sub_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const vfloat& b) { return vfloat(a) + b; }
CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, float b) { return a + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vint& b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vint& a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, int b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(int a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, float b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(float a, const vfloat& b) { return vfloat(a) - b; }

CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_mul_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, float b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float a, const vfloat& b) { return vfloat(a) * b; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, int b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(int a, const vfloat& b) { return vfloat(a) * b; }

CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_div_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, int b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(int a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, float b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(float a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& v) { return vfloat{ _mm256_sub_ps(_mm256_xor_ps(v.m_value, v.m_value), v.m_value) }; }

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, const vfloat& b) { return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_EQ_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, float b) { return a == vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, const vfloat& b) { return !vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_EQ_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, float b) { return a != vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, const vfloat& b) { return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_LT_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, float b) { return a < vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, const vfloat& b) { return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_GT_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, float b) { return a > vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, const vfloat& b) { return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_LE_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, float b) { return a <= vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, const vfloat& b) { return vbool{ _mm256_castps_si256(_mm256_cmp_ps(a.m_value, b.m_value, _CMP_GE_OQ)) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, float b) { return a >= vfloat(b); }

CPPSPMD_FORCE_INLINE vfloat spmd_ternaryf(const vbool& cond, const vfloat& a, const vfloat& b) { return vfloat{ _mm256_blendv_ps(b.m_value, a.m_value, _mm256_castsi256_ps(cond.m_value)) }; }

CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) 
{ 
	return vint{ 
		_mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b.m_value_l), _mm_castsi128_ps(a.m_value_l), _mm_castsi128_ps( get_lo_i(cond.m_value) ) ) ),
		_mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b.m_value_h), _mm_castsi128_ps(a.m_value_h), _mm_castsi128_ps( get_hi_i(cond.m_value) ) ) ) 
	}; 
}

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ _mm256_sqrt_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_max_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_min_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ _mm256_ceil_ps(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ _mm256_floor_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC ) }; }
CPPSPMD_FORCE_INLINE vfloat frac(const vfloat& a) { return a - floor(a); }
CPPSPMD_FORCE_INLINE vfloat fmod(const vfloat &a, const vfloat &b) { vfloat c = frac(abs(a / b)) * abs(b); return spmd_ternaryf(a < 0, -c, c); }
CPPSPMD_FORCE_INLINE vfloat sign(const vfloat& a) { return spmd_ternaryf(a < 0.0f, 1.0f, 1.0f); }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b) 
{ 
	return vint{ _mm_max_epi32(a.m_value_l, b.m_value_l), _mm_max_epi32(a.m_value_h, b.m_value_h) };
}

CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b) 
{	
	return vint{ _mm_min_epi32(a.m_value_l, b.m_value_l), _mm_min_epi32(a.m_value_h, b.m_value_h) };
}

CPPSPMD_FORCE_INLINE vint maxu(const vint& a, const vint& b)
{
	return vint{ _mm_max_epu32(a.m_value_l, b.m_value_l), _mm_max_epu32(a.m_value_h, b.m_value_h) };
}

CPPSPMD_FORCE_INLINE vint minu(const vint& a, const vint& b)
{
	return vint{ _mm_min_epu32(a.m_value_l, b.m_value_l), _mm_min_epu32(a.m_value_h, b.m_value_h) };
}

CPPSPMD_FORCE_INLINE vint abs(const vint& v) { return vint{ _mm_abs_epi32(v.m_value_l), _mm_abs_epi32(v.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint byteswap(const vint& v) 
{ 
	__m128i mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3); 
	return vint{ _mm_shuffle_epi8(v.m_value_l, mask), _mm_shuffle_epi8(v.m_value_h, mask) }; 
}

CPPSPMD_FORCE_INLINE vint cast_vfloat_to_vint(const vfloat& v) 
{ 
	__m256i k = _mm256_castps_si256(v.m_value);
	return vint{ get_lo_i(k), get_hi_i(k) }; 
}

CPPSPMD_FORCE_INLINE vfloat cast_vint_to_vfloat(const vint& v) 
{ 
	__m256i k = combine_i(v.m_value_l, v.m_value_h);
	return vfloat{ _mm256_castsi256_ps(k) }; 
}

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ _mm256_min_ps(b.m_value, _mm256_max_ps(v.m_value, a.m_value) ) };
}

CPPSPMD_FORCE_INLINE vint clamp(const vint& v, const vint& a, const vint& b)
{
	return vint{ _mm_max_epi32(a.m_value_l, _mm_min_epi32(v.m_value_l, b.m_value_l) ), _mm_max_epi32(a.m_value_h, _mm_min_epi32(v.m_value_h, b.m_value_h) ) }; 
}

CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_add_ps(_mm256_mul_ps(a.m_value, b.m_value), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_sub_ps(_mm256_mul_ps(a.m_value, b.m_value), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_sub_ps(c.m_value, _mm256_mul_ps(a.m_value, b.m_value)) };
}

CPPSPMD_FORCE_INLINE vfloat fnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm256_sub_ps(_mm256_sub_ps(_mm256_xor_ps(a.m_value, a.m_value), _mm256_mul_ps(a.m_value, b.m_value)), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { __m128i k = _mm_set1_epi32(a); return lint{ _mm_add_epi32(k, b.m_value_l), _mm_add_epi32(k, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { __m128i k = _mm_set1_epi32(b); return lint{ _mm_add_epi32(a.m_value_l, k), _mm_add_epi32(a.m_value_h, k) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ _mm_and_si128(a.m_value_l, b.m_value_l), _mm_and_si128(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint andnot(const vint& a, const vint& b) { return vint{ _mm_andnot_si128(a.m_value_l, b.m_value_l), _mm_andnot_si128(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, const vint& b) { return vint{ _mm_or_si128(a.m_value_l, b.m_value_l), _mm_or_si128(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, int b) { return a | vint(b); }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, const vint& b) { return vint{ _mm_xor_si128(a.m_value_l, b.m_value_l), _mm_xor_si128(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, int b) { return a ^ vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(const vint& a, const vint& b) { return vbool{ combine_i(_mm_cmpeq_epi32(a.m_value_l, b.m_value_l), _mm_cmpeq_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vint& a, const vint& b) { return !vbool{ combine_i(_mm_cmpeq_epi32(a.m_value_l, b.m_value_l), _mm_cmpeq_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vint& a, const vint& b) { return vbool{ combine_i(_mm_cmplt_epi32(a.m_value_l, b.m_value_l), _mm_cmplt_epi32(a.m_value_h, b.m_value_h)) }; }

CPPSPMD_FORCE_INLINE vbool operator<=(const vint& a, const vint& b) { return !vbool{ combine_i(_mm_cmplt_epi32(b.m_value_l, a.m_value_l), _mm_cmplt_epi32(b.m_value_h, a.m_value_h)) }; }

CPPSPMD_FORCE_INLINE vbool operator>=(const vint& a, const vint& b) { return !vbool{ combine_i(_mm_cmpgt_epi32(b.m_value_l, a.m_value_l), _mm_cmpgt_epi32(b.m_value_h, a.m_value_h)) }; }

CPPSPMD_FORCE_INLINE vbool operator>(const vint& a, const vint& b) { return vbool{ combine_i(_mm_cmpgt_epi32(a.m_value_l, b.m_value_l), _mm_cmpgt_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, const vint& b) { return vint{ _mm_add_epi32(a.m_value_l, b.m_value_l), _mm_add_epi32(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, const vint& b) { return vint{ _mm_sub_epi32(a.m_value_l, b.m_value_l), _mm_sub_epi32(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, int b) { return a + vint(b); }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, int b) { return a - vint(b); }
CPPSPMD_FORCE_INLINE vint operator+(int a, const vint& b) { return vint(a) + b; }
CPPSPMD_FORCE_INLINE vint operator-(int a, const vint& b) { return vint(a) - b; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, const vint& b) { return vint{ _mm_mullo_epi32(a.m_value_l, b.m_value_l), _mm_mullo_epi32(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, int b) { return a * vint(b); }
CPPSPMD_FORCE_INLINE vint operator*(int a, const vint& b) { return vint(a) * b; }

CPPSPMD_FORCE_INLINE vint mulhiu(const vint& a, const vint& b) { return vint{ mulhi_epu32(a.m_value_l, b.m_value_l), mulhi_epu32(a.m_value_h, b.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ _mm_sub_epi32(_mm_setzero_si128(), v.m_value_l), _mm_sub_epi32(_mm_setzero_si128(), v.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint operator~(const vint& a) { return vint{ -a - 1 }; }

// A few of these break the lane-based abstraction model. They are supported in SSE2, so it makes sense to support them and let the user figure it out.
CPPSPMD_FORCE_INLINE vint adds_epu8(const vint& a, const vint& b) { return vint{ _mm_adds_epu8(a.m_value_l, b.m_value_l), _mm_adds_epu8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint subs_epu8(const vint& a, const vint& b) { return vint{ _mm_subs_epu8(a.m_value_l, b.m_value_l), _mm_subs_epu8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint avg_epu8(const vint& a, const vint& b) { return vint{ _mm_avg_epu8(a.m_value_l, b.m_value_l), _mm_avg_epu8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint max_epu8(const vint& a, const vint& b) { return vint{ _mm_max_epu8(a.m_value_l, b.m_value_l), _mm_max_epu8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint min_epu8(const vint& a, const vint& b) { return vint{ _mm_min_epu8(a.m_value_l, b.m_value_l), _mm_min_epu8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint sad_epu8(const vint& a, const vint& b) { return vint{ _mm_sad_epu8(a.m_value_l, b.m_value_l), _mm_sad_epu8(a.m_value_h, b.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint add_epi8(const vint& a, const vint& b) { return vint{ _mm_add_epi8(a.m_value_l, b.m_value_l), _mm_add_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint adds_epi8(const vint& a, const vint& b) { return vint{ _mm_adds_epi8(a.m_value_l, b.m_value_l), _mm_adds_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint sub_epi8(const vint& a, const vint& b) { return vint{ _mm_sub_epi8(a.m_value_l, b.m_value_l), _mm_sub_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint subs_epi8(const vint& a, const vint& b) { return vint{ _mm_subs_epi8(a.m_value_l, b.m_value_l), _mm_subs_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi8(const vint& a, const vint& b) { return vint{ _mm_cmpeq_epi8(a.m_value_l, b.m_value_l), _mm_cmpeq_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi8(const vint& a, const vint& b) { return vint{ _mm_cmpgt_epi8(a.m_value_l, b.m_value_l), _mm_cmpgt_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi8(const vint& a, const vint& b) { return vint{ _mm_cmplt_epi8(a.m_value_l, b.m_value_l), _mm_cmplt_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint unpacklo_epi8(const vint& a, const vint& b) { return vint{ _mm_unpacklo_epi8(a.m_value_l, b.m_value_l), _mm_unpacklo_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint unpackhi_epi8(const vint& a, const vint& b) { return vint{ _mm_unpackhi_epi8(a.m_value_l, b.m_value_l), _mm_unpackhi_epi8(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE int movemask_epi8(const vint& a) { return _mm_movemask_epi8(a.m_value_l) | (_mm_movemask_epi8(a.m_value_h) << 16); }
CPPSPMD_FORCE_INLINE int movemask_epi32(const vint& a) { return _mm_movemask_ps(_mm_castsi128_ps(a.m_value_l)) | (_mm_movemask_ps(_mm_castsi128_ps(a.m_value_h)) << 4); }

CPPSPMD_FORCE_INLINE vint cmple_epu8(const vint& a, const vint& b) { return vint{ _mm_cmpeq_epi8(_mm_min_epu8(a.m_value_l, b.m_value_l), a.m_value_l), _mm_cmpeq_epi8(_mm_min_epu8(a.m_value_h, b.m_value_h), a.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmpge_epu8(const vint& a, const vint& b) { return vint{ cmple_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epu8(const vint& a, const vint& b) { return vint{ _mm_andnot_si128(_mm_cmpeq_epi8(a.m_value_l, b.m_value_l), _mm_cmpeq_epi8(_mm_max_epu8(a.m_value_l, b.m_value_l), a.m_value_l)), _mm_andnot_si128(_mm_cmpeq_epi8(a.m_value_h, b.m_value_h), _mm_cmpeq_epi8(_mm_max_epu8(a.m_value_h, b.m_value_h), a.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epu8(const vint& a, const vint& b) { return vint{ cmpgt_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint absdiff_epu8(const vint& a, const vint& b) { return vint{ _mm_or_si128(_mm_subs_epu8(a.m_value_l, b.m_value_l), _mm_subs_epu8(b.m_value_l, a.m_value_l)), _mm_or_si128(_mm_subs_epu8(a.m_value_h, b.m_value_h), _mm_subs_epu8(b.m_value_h, a.m_value_h)) }; }

CPPSPMD_FORCE_INLINE vint blendv_epi8(const vint& a, const vint& b, const vint &mask) { return vint{ _mm_blendv_epi8(a.m_value_l, b.m_value_l, mask.m_value_l), _mm_blendv_epi8(a.m_value_h, b.m_value_h, mask.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint blendv_epi32(const vint& a, const vint& b, const vint &mask) { return vint{ _mm_blendv_epi32(a.m_value_l, b.m_value_l, mask.m_value_l), _mm_blendv_epi8(a.m_value_h, b.m_value_h, mask.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint undefined_vint() { return vint{ _mm_undefined_si128(), _mm_undefined_si128() }; }
CPPSPMD_FORCE_INLINE vfloat undefined_vfloat() { return vfloat{ _mm256_undefined_ps() }; }

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int32's in each 128-bit lane.
#define VINT_LANE_SHUFFLE_EPI32(a, control) vint(_mm_shuffle_epi32((a).m_value_l, control), _mm_shuffle_epi32((a).m_value_h, control))

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int16's in either the high or low 64-bit lane.
#define VINT_LANE_SHUFFLELO_EPI16(a, control) vint(_mm_shufflelo_epi16((a).m_value_l, control), _mm_shufflelo_epi16((a).m_value_h, control))
#define VINT_LANE_SHUFFLEHI_EPI16(a, control) vint(_mm_shufflehi_epi16((a).m_value_l, control), _mm_shufflehi_epi16((a).m_value_h, control))

#define VINT_LANE_SHUFFLE_MASK(a, b, c, d) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))
#define VINT_LANE_SHUFFLE_MASK_R(d, c, b, a) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))

#define VINT_LANE_SHIFT_LEFT_BYTES(a, l) vint(_mm_slli_si128((a).m_value_l, l), _mm_slli_si128((a).m_value_h, l))
#define VINT_LANE_SHIFT_RIGHT_BYTES(a, l) vint(_mm_srli_si128((a).m_value_l, l), _mm_srli_si128((a).m_value_h, l))

// Unpack and interleave 8-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi8(const vint& a, const vint& b) { return vint(_mm_unpacklo_epi8(a.m_value_l, b.m_value_l), _mm_unpacklo_epi8(a.m_value_h, b.m_value_h)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi8(const vint& a, const vint& b) { return vint(_mm_unpackhi_epi8(a.m_value_l, b.m_value_l), _mm_unpackhi_epi8(a.m_value_h, b.m_value_h)); }

// Unpack and interleave 16-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi16(const vint& a, const vint& b) { return vint(_mm_unpacklo_epi16(a.m_value_l, b.m_value_l), _mm_unpacklo_epi16(a.m_value_h, b.m_value_h)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi16(const vint& a, const vint& b) { return vint(_mm_unpackhi_epi16(a.m_value_l, b.m_value_l), _mm_unpackhi_epi16(a.m_value_h, b.m_value_h)); }

// Unpack and interleave 32-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi32(const vint& a, const vint& b) { return vint(_mm_unpacklo_epi32(a.m_value_l, b.m_value_l), _mm_unpacklo_epi32(a.m_value_h, b.m_value_h)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi32(const vint& a, const vint& b) { return vint(_mm_unpackhi_epi32(a.m_value_l, b.m_value_l), _mm_unpackhi_epi32(a.m_value_h, b.m_value_h)); }

// Unpack and interleave 64-bit integers from the low or high half of a and b
CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi64(const vint& a, const vint& b) { return vint(_mm_unpacklo_epi64(a.m_value_l, b.m_value_l), _mm_unpacklo_epi64(a.m_value_h, b.m_value_h)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi64(const vint& a, const vint& b) { return vint(_mm_unpackhi_epi64(a.m_value_l, b.m_value_l), _mm_unpackhi_epi64(a.m_value_h, b.m_value_h)); }

CPPSPMD_FORCE_INLINE vint vint_set1_epi8(int8_t a) { return vint(_mm_set1_epi8(a), _mm_set1_epi8(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi16(int16_t a) { return vint(_mm_set1_epi16(a), _mm_set1_epi16(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi32(int32_t a) { return vint(_mm_set1_epi32(a), _mm_set1_epi32(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi64(int64_t a) { return vint(_mm_set1_epi64x(a), _mm_set1_epi64x(a)); }

CPPSPMD_FORCE_INLINE vint add_epi16(const vint& a, const vint& b) { return vint{ _mm_add_epi16(a.m_value_l, b.m_value_l), _mm_add_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint adds_epi16(const vint& a, const vint& b) { return vint{ _mm_adds_epi16(a.m_value_l, b.m_value_l), _mm_adds_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint adds_epu16(const vint& a, const vint& b) { return vint{ _mm_adds_epu16(a.m_value_l, b.m_value_l), _mm_adds_epu16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint avg_epu16(const vint& a, const vint& b) { return vint{ _mm_avg_epu16(a.m_value_l, b.m_value_l), _mm_avg_epu16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint sub_epi16(const vint& a, const vint& b) { return vint{ _mm_sub_epi16(a.m_value_l, b.m_value_l), _mm_sub_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint subs_epi16(const vint& a, const vint& b) { return vint{ _mm_subs_epi16(a.m_value_l, b.m_value_l), _mm_subs_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint subs_epu16(const vint& a, const vint& b) { return vint{ _mm_subs_epu16(a.m_value_l, b.m_value_l), _mm_subs_epu16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint mullo_epi16(const vint& a, const vint& b) { return vint{ _mm_mullo_epi16(a.m_value_l, b.m_value_l), _mm_mullo_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epi16(const vint& a, const vint& b) { return vint{ _mm_mulhi_epi16(a.m_value_l, b.m_value_l), _mm_mulhi_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epu16(const vint& a, const vint& b) { return vint{ _mm_mulhi_epu16(a.m_value_l, b.m_value_l), _mm_mulhi_epu16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint min_epi16(const vint& a, const vint& b) { return vint{ _mm_min_epi16(a.m_value_l, b.m_value_l), _mm_min_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint max_epi16(const vint& a, const vint& b) { return vint{ _mm_max_epi16(a.m_value_l, b.m_value_l), _mm_max_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint madd_epi16(const vint& a, const vint& b) { return vint{ _mm_madd_epi16(a.m_value_l, b.m_value_l), _mm_madd_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi16(const vint& a, const vint& b) { return vint{ _mm_cmpeq_epi16(a.m_value_l, b.m_value_l), _mm_cmpeq_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi16(const vint& a, const vint& b) { return vint{ _mm_cmpgt_epi16(a.m_value_l, b.m_value_l), _mm_cmpgt_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi16(const vint& a, const vint& b) { return vint{ _mm_cmplt_epi16(a.m_value_l, b.m_value_l), _mm_cmplt_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint packs_epi16(const vint& a, const vint& b) { return vint{ _mm_packs_epi16(a.m_value_l, b.m_value_l), _mm_packs_epi16(a.m_value_h, b.m_value_h) }; }
CPPSPMD_FORCE_INLINE vint packus_epi16(const vint& a, const vint& b) { return vint{ _mm_packus_epi16(a.m_value_l, b.m_value_l), _mm_packus_epi16(a.m_value_h, b.m_value_h) }; }

CPPSPMD_FORCE_INLINE vint uniform_shift_left_epi16(const vint& a, const vint& b) { return vint{ _mm_sll_epi16(a.m_value_l, b.m_value_l), _mm_sll_epi16(a.m_value_h, b.m_value_l) }; }
CPPSPMD_FORCE_INLINE vint uniform_arith_shift_right_epi16(const vint& a, const vint& b) { return vint{ _mm_sra_epi16(a.m_value_l, b.m_value_l), _mm_sra_epi16(a.m_value_h, b.m_value_l) }; }
CPPSPMD_FORCE_INLINE vint uniform_shift_right_epi16(const vint& a, const vint& b) { return vint{ _mm_srl_epi16(a.m_value_l, b.m_value_l), _mm_srl_epi16(a.m_value_h, b.m_value_l) }; }

#define VINT_SHIFT_LEFT_EPI16(a, b) vint(_mm_slli_epi16((a).m_value_l, b), _mm_slli_epi16((a).m_value_h, b))
#define VINT_SHIFT_RIGHT_EPI16(a, b) vint(_mm_srai_epi16((a).m_value_l, b), _mm_srai_epi16((a).m_value_h, b))
#define VUINT_SHIFT_RIGHT_EPI16(a, b) vint(_mm_srli_epi16((a).m_value_l, b), _mm_srli_epi16((a).m_value_h, b))

CPPSPMD_FORCE_INLINE vint mul_epu32(const vint &a, const vint& b) { return vint(_mm_mul_epu32(a.m_value_l, b.m_value_l), _mm_mul_epu32(a.m_value_h, b.m_value_h)); }

CPPSPMD_FORCE_INLINE vint div_epi32(const vint &a, const vint& b)
{
	__m256d al = _mm256_cvtepi32_pd(a.m_value_l);
	__m256d ah = _mm256_cvtepi32_pd(a.m_value_h);

	__m256d bl = _mm256_cvtepi32_pd(b.m_value_l);
	__m256d bh = _mm256_cvtepi32_pd(b.m_value_h);

	__m256d rl = _mm256_div_pd(al, bl);
	__m256d rh = _mm256_div_pd(ah, bh);

	__m128i rli = _mm256_cvttpd_epi32(rl);
	__m128i rhi = _mm256_cvttpd_epi32(rh);

	return vint(rli, rhi);
}

CPPSPMD_FORCE_INLINE vint mod_epi32(const vint &a, const vint& b)
{
	vint aa = abs(a), ab = abs(b);
	vint q = div_epi32(aa, ab);
	vint r = aa - q * ab;
	return spmd_ternaryi(a < 0, -r, r);
}

CPPSPMD_FORCE_INLINE vint operator/(const vint& a, const vint& b) 
{ 
	return div_epi32(a, b);
}

CPPSPMD_FORCE_INLINE vint operator/(const vint& a, int b) 
{ 
	return div_epi32(a, vint(b));
}

CPPSPMD_FORCE_INLINE vint operator%(const vint& a, const vint& b) 
{ 
	return mod_epi32(a, b);
}

CPPSPMD_FORCE_INLINE vint operator%(const vint& a, int b) 
{ 
	return mod_epi32(a, vint(b));
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[8];
	result[0] = _mm_extract_epi32(a.m_value_l, 0) << _mm_extract_epi32(b.m_value_l, 0);
	result[1] = _mm_extract_epi32(a.m_value_l, 1) << _mm_extract_epi32(b.m_value_l, 1);
	result[2] = _mm_extract_epi32(a.m_value_l, 2) << _mm_extract_epi32(b.m_value_l, 2);
	result[3] = _mm_extract_epi32(a.m_value_l, 3) << _mm_extract_epi32(b.m_value_l, 3);

	result[4] = _mm_extract_epi32(a.m_value_h, 0) << _mm_extract_epi32(b.m_value_h, 0);
	result[5] = _mm_extract_epi32(a.m_value_h, 1) << _mm_extract_epi32(b.m_value_h, 1);
	result[6] = _mm_extract_epi32(a.m_value_h, 2) << _mm_extract_epi32(b.m_value_h, 2);
	result[7] = _mm_extract_epi32(a.m_value_h, 3) << _mm_extract_epi32(b.m_value_h, 3);

	return vint{ _mm_load_si128((__m128i*)result), _mm_load_si128(((__m128i*)result) + 1) };
}

// uniform shift left
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_sll_epi32(a.m_value_l, bv), _mm_sll_epi32(a.m_value_h, bv) };
}

// uniform arithmetic shift right
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_sra_epi32(a.m_value_l, bv), _mm_sra_epi32(a.m_value_h, bv) };
}

// uniform shift right
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_srl_epi32(a.m_value_l, bv), _mm_srl_epi32(a.m_value_h, bv) };
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[8];
	result[0] = _mm_extract_epi32(a.m_value_l, 0) >> _mm_extract_epi32(b.m_value_l, 0);
	result[1] = _mm_extract_epi32(a.m_value_l, 1) >> _mm_extract_epi32(b.m_value_l, 1);
	result[2] = _mm_extract_epi32(a.m_value_l, 2) >> _mm_extract_epi32(b.m_value_l, 2);
	result[3] = _mm_extract_epi32(a.m_value_l, 3) >> _mm_extract_epi32(b.m_value_l, 3);

	result[4] = _mm_extract_epi32(a.m_value_h, 0) >> _mm_extract_epi32(b.m_value_h, 0);
	result[5] = _mm_extract_epi32(a.m_value_h, 1) >> _mm_extract_epi32(b.m_value_h, 1);
	result[6] = _mm_extract_epi32(a.m_value_h, 2) >> _mm_extract_epi32(b.m_value_h, 2);
	result[7] = _mm_extract_epi32(a.m_value_h, 3) >> _mm_extract_epi32(b.m_value_h, 3);

	return vint{ _mm_load_si128((__m128i*)result), _mm_load_si128(((__m128i*)result) + 1) };
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) uint32_t result[8];
	result[0] = ((uint32_t)_mm_extract_epi32(a.m_value_l, 0)) >> _mm_extract_epi32(b.m_value_l, 0);
	result[1] = ((uint32_t)_mm_extract_epi32(a.m_value_l, 1)) >> _mm_extract_epi32(b.m_value_l, 1);
	result[2] = ((uint32_t)_mm_extract_epi32(a.m_value_l, 2)) >> _mm_extract_epi32(b.m_value_l, 2);
	result[3] = ((uint32_t)_mm_extract_epi32(a.m_value_l, 3)) >> _mm_extract_epi32(b.m_value_l, 3);

	result[4] = ((uint32_t)_mm_extract_epi32(a.m_value_h, 0)) >> _mm_extract_epi32(b.m_value_h, 0);
	result[5] = ((uint32_t)_mm_extract_epi32(a.m_value_h, 1)) >> _mm_extract_epi32(b.m_value_h, 1);
	result[6] = ((uint32_t)_mm_extract_epi32(a.m_value_h, 2)) >> _mm_extract_epi32(b.m_value_h, 2);
	result[7] = ((uint32_t)_mm_extract_epi32(a.m_value_h, 3)) >> _mm_extract_epi32(b.m_value_h, 3);

	return vint{ _mm_load_si128((__m128i*)result), _mm_load_si128(((__m128i*)result) + 1) };
}

CPPSPMD_FORCE_INLINE vint vuint_shift_right_not_zero(const vint& a, const vint& b) { return vuint_shift_right(a, b); }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) vint { _mm_slli_epi32( (a).m_value_l, (b) ), _mm_slli_epi32( (a).m_value_h, (b) ) }
#define VINT_SHIFT_RIGHT(a, b) vint { _mm_srai_epi32( (a).m_value_l, (b) ), _mm_srai_epi32( (a).m_value_h, (b) ) }
#define VUINT_SHIFT_RIGHT(a, b) vint { _mm_srli_epi32( (a).m_value_l, (b) ), _mm_srli_epi32( (a).m_value_h, (b) ) }
#define VINT_ROT(x, k) (VINT_SHIFT_LEFT((x), (k)) | VUINT_SHIFT_RIGHT((x), 32 - (k)))

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, const lint& b) { return vbool{ combine_i(_mm_cmpeq_epi32(a.m_value_l, b.m_value_l), _mm_cmpeq_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, int b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint& b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint& a, const lint& b) { return vbool{ combine_i(_mm_cmpgt_epi32(b.m_value_l, a.m_value_l), _mm_cmpgt_epi32(b.m_value_h, a.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const lint& a, const lint& b) { return vbool{ combine_i(_mm_cmpgt_epi32(a.m_value_l, b.m_value_l), _mm_cmpgt_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const lint& a, const lint& b) { return !vbool{ combine_i(_mm_cmpgt_epi32(a.m_value_l, b.m_value_l), _mm_cmpgt_epi32(a.m_value_h, b.m_value_h)) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const lint& a, const lint& b) { return !vbool{ combine_i(_mm_cmpgt_epi32(b.m_value_l, a.m_value_l), _mm_cmpgt_epi32(b.m_value_h, a.m_value_h)) }; }

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) float values[8]; _mm256_store_ps(values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const vint& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm_store_si128((__m128i*)values, v.m_value_l); _mm_store_si128(((__m128i*)values) + 1, v.m_value_h); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const lint& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm_store_si128((__m128i*)values, v.m_value_l); _mm_store_si128(((__m128i*)values) + 1, v.m_value_h); return values[instance]; }
CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm256_store_si256((__m256i*)values, v.m_value);	return values[instance] != 0; }

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

CPPSPMD_FORCE_INLINE float cast_int_to_float(int i) {	return *(const float*)&i; }

#define VINT_EXTRACT(v, instance) (((instance) < 4) ? _mm_extract_epi32((v).m_value_l, (instance)) : _mm_extract_epi32((v).m_value_h, (instance) - 4))
#define VBOOL_EXTRACT(v, instance) _mm256_extract_epi32((v).m_value, instance)
#define VFLOAT_EXTRACT(v, instance) cast_int_to_float(_mm256_extract_epi32(_mm256_castps_si256(v.m_value), instance))

CPPSPMD_FORCE_INLINE vfloat &insert(vfloat& v, int instance, float f)
{
	assert(instance < 8);
	CPPSPMD_ALIGN(32) float values[8];
	_mm256_store_ps(values, v.m_value);
	values[instance] = f;
	v.m_value = _mm256_load_ps(values);
	return v;
}

CPPSPMD_FORCE_INLINE vint &insert(vint& v, int instance, int i)
{
	assert(instance < 8);
	CPPSPMD_ALIGN(16) int values[8];
	_mm_store_si128((__m128i *)values, v.m_value_l);
	_mm_store_si128((__m128i *)values + 1, v.m_value_h);
	values[instance] = i;
	v.m_value_l = _mm_load_si128((__m128i *)values);
	v.m_value_h = _mm_load_si128((__m128i *)values + 1);
	return v;
}

CPPSPMD_FORCE_INLINE vint init_lookup4(const uint8_t pTab[16])
{
	__m128i l = _mm_loadu_si128((const __m128i*)pTab);
	return vint{ l, l };
}

CPPSPMD_FORCE_INLINE vint table_lookup4_8(const vint& a, const vint& table)
{
	return vint{ _mm_shuffle_epi8(table.m_value_l, a.m_value_l), _mm_shuffle_epi8(table.m_value_h, a.m_value_h) };
}

CPPSPMD_FORCE_INLINE void init_lookup5(const uint8_t pTab[32], vint& table_0, vint& table_1)
{
	__m128i l = _mm_loadu_si128((const __m128i*)pTab);
	__m128i h = _mm_loadu_si128((const __m128i*)(pTab + 16));
	table_0.m_value_l = l;
	table_0.m_value_h = l;
	table_1.m_value_l = h;
	table_1.m_value_h = h;
}

CPPSPMD_FORCE_INLINE vint table_lookup5_8(const vint& a, const vint& table_0, const vint& table_1)
{
	__m128i l_0 = _mm_shuffle_epi8(table_0.m_value_l, a.m_value_l);
	__m128i l_1 = _mm_shuffle_epi8(table_0.m_value_h, a.m_value_h);

	__m128i h_0 = _mm_shuffle_epi8(table_1.m_value_l, a.m_value_l);
	__m128i h_1 = _mm_shuffle_epi8(table_1.m_value_h, a.m_value_h);

	__m128i m_0 = _mm_slli_epi32(a.m_value_l, 31 - 4);
	__m128i m_1 = _mm_slli_epi32(a.m_value_h, 31 - 4);

	__m128 v_0 = _mm_blendv_ps(_mm_castsi128_ps(l_0), _mm_castsi128_ps(h_0), _mm_castsi128_ps(m_0));
	__m128 v_1 = _mm_blendv_ps(_mm_castsi128_ps(l_1), _mm_castsi128_ps(h_1), _mm_castsi128_ps(m_1));

	return vint{ _mm_castps_si128(v_0), _mm_castps_si128(v_1) };
}

CPPSPMD_FORCE_INLINE void init_lookup6(const uint8_t pTab[64], vint& table_0, vint& table_1, vint& table_2, vint& table_3)
{
	__m128i a = _mm_loadu_si128((const __m128i*)pTab);
	__m128i b = _mm_loadu_si128((const __m128i*)(pTab + 16));
	__m128i c = _mm_loadu_si128((const __m128i*)(pTab + 32));
	__m128i d = _mm_loadu_si128((const __m128i*)(pTab + 48));

	table_0.m_value_l = a;
	table_0.m_value_h = a;

	table_1.m_value_l = b;
	table_1.m_value_h = b;

	table_2.m_value_l = c;
	table_2.m_value_h = c;

	table_3.m_value_l = d;
	table_3.m_value_h = d;
}

CPPSPMD_FORCE_INLINE vint table_lookup6_8(const vint& a, const vint& table_0, const vint& table_1, const vint& table_2, const vint& table_3)
{
	__m128i m_0 = _mm_slli_epi32(a.m_value_l, 31 - 4);
	__m128i m_1 = _mm_slli_epi32(a.m_value_h, 31 - 4);

	__m128 av_0, av_1;
	{
		__m128i al_0 = _mm_shuffle_epi8(table_0.m_value_l, a.m_value_l);
		__m128i al_1 = _mm_shuffle_epi8(table_0.m_value_h, a.m_value_h);
		__m128i ah_0 = _mm_shuffle_epi8(table_1.m_value_l, a.m_value_l);
		__m128i ah_1 = _mm_shuffle_epi8(table_1.m_value_h, a.m_value_h);
		av_0 = _mm_blendv_ps(_mm_castsi128_ps(al_0), _mm_castsi128_ps(ah_0), _mm_castsi128_ps(m_0));
		av_1 = _mm_blendv_ps(_mm_castsi128_ps(al_1), _mm_castsi128_ps(ah_1), _mm_castsi128_ps(m_1));
	}
	
	__m128 bv_0, bv_1;
	{
		__m128i bl_0 = _mm_shuffle_epi8(table_2.m_value_l, a.m_value_l);
		__m128i bl_1 = _mm_shuffle_epi8(table_2.m_value_h, a.m_value_h);
		__m128i bh_0 = _mm_shuffle_epi8(table_3.m_value_l, a.m_value_l);
		__m128i bh_1 = _mm_shuffle_epi8(table_3.m_value_h, a.m_value_h);
		bv_0 = _mm_blendv_ps(_mm_castsi128_ps(bl_0), _mm_castsi128_ps(bh_0), _mm_castsi128_ps(m_0));
		bv_1 = _mm_blendv_ps(_mm_castsi128_ps(bl_1), _mm_castsi128_ps(bh_1), _mm_castsi128_ps(m_1));
	}
	
	__m128i m2_0 = _mm_slli_epi32(a.m_value_l, 31 - 5);
	__m128i m2_1 = _mm_slli_epi32(a.m_value_h, 31 - 5);
	__m128 v2_0 = _mm_blendv_ps(av_0, bv_0, _mm_castsi128_ps(m2_0));
	__m128 v2_1 = _mm_blendv_ps(av_1, bv_1, _mm_castsi128_ps(m2_1));

	return vint{ _mm_castps_si128(v2_0), _mm_castps_si128(v2_1) };
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
	int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
	
	int *pDstI = (int *)dst.m_pValue;
	if (mask & 1) pDstI[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
	if (mask & 2) pDstI[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
	if (mask & 4) pDstI[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
	if (mask & 8) pDstI[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

	if (mask & 16) pDstI[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
	if (mask & 32) pDstI[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
	if (mask & 64) pDstI[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
	if (mask & 128) pDstI[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	int *pDstI = (int *)dst.m_pValue;
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

	pDstI[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);

	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
	
	int *pDstI = (int *)dst.m_pValue;
	if (mask & 1) pDstI[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
	if (mask & 2) pDstI[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
	if (mask & 4) pDstI[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
	if (mask & 8) pDstI[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

	if (mask & 16) pDstI[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
	if (mask & 32) pDstI[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
	if (mask & 64) pDstI[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
	if (mask & 128) pDstI[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	int *pDstI = (int *)dst.m_pValue;
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
	pDstI[_mm_extract_epi32(dst.m_vindex_l, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

	pDstI[_mm_extract_epi32(dst.m_vindex_h, 0)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 1)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 2)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
	pDstI[_mm_extract_epi32(dst.m_vindex_h, 3)] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);

	return dst;
}

#include "cppspmd_flow.h"
#include "cppspmd_math.h"

} // namespace cppspmd_avx1
