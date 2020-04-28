// cppspmd_avx2.h
// The module is intended for AVX2, but it also supports AVX1. Also supports optional FMA support.
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
#include <immintrin.h>

// By default this header is for AVX2, but I've left the older AVX1 code in place for benchmarking purposes.
#ifndef CPPSPMD_USE_AVX2
#define CPPSPMD_USE_AVX2 1
#endif

#ifndef CPPSPMD_USE_FMA
#define CPPSPMD_USE_FMA 0
#endif

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

#undef CPPSPMD_SSE
#undef CPPSPMD_AVX1
#undef CPPSPMD_AVX2
#undef CPPSPMD_AVX
#undef CPPSPMD_FLOAT4

#define CPPSPMD_SSE 0
#define CPPSPMD_AVX 1
#define CPPSPMD_FLOAT4 0

#if CPPSPMD_USE_FMA
	#if CPPSPMD_USE_AVX2
		#define CPPSPMD cppspmd_avx2_fma
		#define CPPSPMD_ARCH _avx2_fma
		#define CPPSPMD_AVX1 0
		#define CPPSPMD_AVX2 1
	#else
		#define CPPSPMD cppspmd_avx1_fma
		#define CPPSPMD_ARCH _avx1_fma
		#define CPPSPMD_AVX1 1
		#define CPPSPMD_AVX2 0
	#endif
#else
	#if CPPSPMD_USE_AVX2
		#define CPPSPMD cppspmd_avx2
		#define CPPSPMD_ARCH _avx2
		#define CPPSPMD_AVX1 0
		#define CPPSPMD_AVX2 1
	#else
		#define CPPSPMD cppspmd_avx1
		#define CPPSPMD_ARCH _avx1
		#define CPPSPMD_AVX1 1
		#define CPPSPMD_AVX2 0
	#endif
#endif

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

const int PROGRAM_COUNT_SHIFT = 3;
const int PROGRAM_COUNT = 1 << PROGRAM_COUNT_SHIFT;

CPPSPMD_DECL(uint32_t, g_allones_256[8]) = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
CPPSPMD_DECL(float, g_onef_256[8]) = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(uint32_t, g_oneu_256[8]) = { 1, 1, 1, 1, 1, 1, 1, 1 };
CPPSPMD_DECL(uint32_t, g_x_128[4]) = { UINT32_MAX, 0, 0, 0 };

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

CPPSPMD_FORCE_INLINE __m256i compare_gt_epi32(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_cmpgt_epi32(a, b);
#else
	return combine_i(_mm_cmpgt_epi32(get_lo_i(a), get_lo_i(b)), _mm_cmpgt_epi32(get_hi_i(a), get_hi_i(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i compare_eq_epi32(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_cmpeq_epi32(a, b);
#else
	return combine_i(_mm_cmpeq_epi32(get_lo_i(a), get_lo_i(b)), _mm_cmpeq_epi32(get_hi_i(a), get_hi_i(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i add_epi32(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_add_epi32(a, b);
#else
	return combine_i(_mm_add_epi32(get_lo_i(a), get_lo_i(b)), _mm_add_epi32(get_hi_i(a), get_hi_i(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i sub_epi32(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_sub_epi32(a, b);
#else
	return combine_i(_mm_sub_epi32(get_lo_i(a), get_lo_i(b)), _mm_sub_epi32(get_hi_i(a), get_hi_i(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i and_si256(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_and_si256(a, b);
#else
	return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i or_si256(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_or_si256(a, b);
#else
	return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i xor_si256(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_xor_si256(a, b);
#else
	return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i andnot_si256(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_andnot_si256(a, b);
#else
	return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

CPPSPMD_FORCE_INLINE __m256i mullo_epi32(__m256i a, __m256i b)
{
#if CPPSPMD_USE_AVX2
	return _mm256_mullo_epi32(a, b);
#else
	return combine_i(_mm_mullo_epi32(get_lo_i(a), get_lo_i(b)), _mm_mullo_epi32(get_hi_i(a), get_hi_i(b)));
#endif
}

const uint32_t ALL_ON_MOVEMASK = 0xFF;

struct spmd_kernel
{
	struct vint;
	struct vbool;
	struct vfloat;
		
	// Exec mask
	struct exec_mask
	{
		__m256i m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const __m256i& mask) : m_mask(mask) { }
				
		static CPPSPMD_FORCE_INLINE exec_mask all_on()	{ return exec_mask{ _mm256_load_si256((const __m256i*)g_allones_256) };	}
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ _mm256_setzero_si256() }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return _mm256_movemask_ps(_mm256_castsi256_ps(m_mask)); }
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
		__m256i m_value;

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
	struct vint_vref
	{
		__m256i m_vindex;
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
#if CPPSPMD_USE_AVX2
		return vfloat{ _mm256_mask_i32gather_ps(_mm256_undefined_ps(),
															 src.m_pValue, src.m_vindex,
															 _mm256_castsi256_ps(m_exec.m_mask),
															 4) };
#else
		CPPSPMD_ALIGN(32) int vindex[8];
		_mm256_store_si256((__m256i*)vindex, src.m_vindex);

		CPPSPMD_ALIGN(32) float loaded[8];

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		for (int i = 0; i < 8; i++)
		{
			if (mask & (1 << i))
				loaded[i] = src.m_pValue[vindex[i]];
		}
		return vfloat{ _mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)loaded)) };
#endif
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
#if CPPSPMD_USE_AVX2
		return vfloat{ _mm256_mask_i32gather_ps(_mm256_undefined_ps(),
															 src.m_pValue, src.m_vindex,
															 _mm256_castsi256_ps(m_exec.m_mask),
															 4) };
#else
		CPPSPMD_ALIGN(32) float loaded[8];

#if 0
		CPPSPMD_ALIGN(32) int vindex[8];
		_mm256_store_si256((__m256i*)vindex, src.m_vindex);
				
		for (int i = 0; i < 8; i++)
			loaded[i] = src.m_pValue[vindex[i]];
#else
		const float *pSrc = src.m_pValue;
		loaded[0] = pSrc[_mm256_extract_epi32(src.m_vindex, 0)];
		loaded[1] = pSrc[_mm256_extract_epi32(src.m_vindex, 1)];
		loaded[2] = pSrc[_mm256_extract_epi32(src.m_vindex, 2)];
		loaded[3] = pSrc[_mm256_extract_epi32(src.m_vindex, 3)];
		loaded[4] = pSrc[_mm256_extract_epi32(src.m_vindex, 4)];
		loaded[5] = pSrc[_mm256_extract_epi32(src.m_vindex, 5)];
		loaded[6] = pSrc[_mm256_extract_epi32(src.m_vindex, 6)];
		loaded[7] = pSrc[_mm256_extract_epi32(src.m_vindex, 7)];
#endif

		return vfloat{ _mm256_load_ps((const float*)loaded) };
#endif
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
			_mm256_storeu_si256((__m256i*)dst.m_pValue, src.m_value);
		}
		else
		{
#if CPPSPMD_USE_AVX2
			_mm256_maskstore_epi32(dst.m_pValue, m_exec.m_mask, src.m_value);
#else
			if (mask & 1) dst.m_pValue[0] = _mm256_extract_epi32(src.m_value, 0);
			if (mask & 2) dst.m_pValue[1] = _mm256_extract_epi32(src.m_value, 1);
			if (mask & 4) dst.m_pValue[2] = _mm256_extract_epi32(src.m_value, 2);
			if (mask & 8) dst.m_pValue[3] = _mm256_extract_epi32(src.m_value, 3);

			if (mask & 16) dst.m_pValue[4] = _mm256_extract_epi32(src.m_value, 4);
			if (mask & 32) dst.m_pValue[5] = _mm256_extract_epi32(src.m_value, 5);
			if (mask & 64) dst.m_pValue[6] = _mm256_extract_epi32(src.m_value, 6);
			if (mask & 128) dst.m_pValue[7] = _mm256_extract_epi32(src.m_value, 7);
#endif
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_lref& src)
	{
#if CPPSPMD_USE_AVX2
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			return vint{ _mm256_loadu_si256((__m256i*)src.m_pValue) };
		else
			return vint{ _mm256_maskload_epi32(src.m_pValue, m_exec.m_mask) };
#else
		__m256i v = _mm256_loadu_si256((const __m256i*)src.m_pValue);

		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ v };
#endif
	}

	// Linear ref to int16's
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};

	CPPSPMD_FORCE_INLINE int16_lref& store(int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(32) int stored[8];
		_mm256_store_si256((__m256i*)stored, src.m_value);

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
		_mm256_store_si256((__m256i*)stored, src.m_value);

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

		return vint{ _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps( t ), _mm256_castsi256_ps(m_exec.m_mask))) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const int16_lref& src)
	{
		CPPSPMD_ALIGN(32) int values[8];

		for (int i = 0; i < 8; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		__m256i t = _mm256_load_si256( (const __m256i *)values );

		return vint{ t };
	}

	// Linear ref to constant int's
	struct cint_lref
	{
		const int* m_pValue;

	private:
		cint_lref& operator=(const cint_lref&);
	};

	CPPSPMD_FORCE_INLINE vint load(const cint_lref& src)
	{
#if CPPSPMD_USE_AVX2
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			return vint{ _mm256_loadu_si256((const __m256i*)src.m_pValue) };
		else
			return vint{ _mm256_maskload_epi32(src.m_pValue, m_exec.m_mask) };
#else
		__m256i v = _mm256_loadu_si256((const __m256i*)src.m_pValue);
		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));
		return vint{ v };
#endif
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_lref& src)
	{
		return vint{ _mm256_loadu_si256((const __m256i*)src.m_pValue) };
	}
		
	// Varying ref to ints
	struct int_vref
	{
		__m256i m_vindex;
		int* m_pValue;

	private:
		int_vref& operator=(const int_vref&);
	};

	// Varying ref to constant ints
	struct cint_vref
	{
		__m256i m_vindex;
		const int* m_pValue;

	private:
		cint_vref& operator=(const cint_vref&);
	};

	// Varying int
	struct vint
	{
		__m256i m_value;

		vint() = default;

		CPPSPMD_FORCE_INLINE explicit vint(const __m256i& value) : m_value(value)	{ }

		CPPSPMD_FORCE_INLINE vint(int value) : m_value(_mm256_set1_epi32(value)) { }

		CPPSPMD_FORCE_INLINE explicit vint(float value) : m_value(_mm256_set1_epi32((int)value))	{ }

		CPPSPMD_FORCE_INLINE explicit vint(const vfloat& other) : m_value(_mm256_cvttps_epi32(other.m_value)) { }

		CPPSPMD_FORCE_INLINE explicit operator vbool() const 
		{
			return vbool{ xor_si256( _mm256_load_si256((const __m256i*)g_allones_256), compare_eq_epi32(m_value, _mm256_setzero_si256())) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm256_cvtepi32_ps(m_value) };
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
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm256_storeu_si256((__m256i*)pDst, src.m_value);
		}
		else
		{
			if (mask & 1) pDst[0] = _mm256_extract_epi32(src.m_value, 0);
			if (mask & 2) pDst[1] = _mm256_extract_epi32(src.m_value, 1);
			if (mask & 4) pDst[2] = _mm256_extract_epi32(src.m_value, 2);
			if (mask & 8) pDst[3] = _mm256_extract_epi32(src.m_value, 3);

			if (mask & 16) pDst[4] = _mm256_extract_epi32(src.m_value, 4);
			if (mask & 32) pDst[5] = _mm256_extract_epi32(src.m_value, 5);
			if (mask & 64) pDst[6] = _mm256_extract_epi32(src.m_value, 6);
			if (mask & 128) pDst[7] = _mm256_extract_epi32(src.m_value, 7);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int *pDst, const vint& src)
	{
		_mm256_storeu_si256((__m256i*)pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int *pDst, const vint& src)
	{
		_mm256_store_si256((__m256i*)pDst, src.m_value);
	}
	
	CPPSPMD_FORCE_INLINE vint loadu_linear(const int *pSrc)
	{
#if CPPSPMD_USE_AVX2
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			return vint{ _mm256_loadu_si256((__m256i*)pSrc) };
		else
			return vint{ _mm256_maskload_epi32(pSrc, m_exec.m_mask) };
#else
		__m256i v = _mm256_loadu_si256((const __m256i*)pSrc);

		v = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v), _mm256_castsi256_ps(m_exec.m_mask)));

		return vint{ v };
#endif
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear_all(const int *pSrc)
	{
		return vint{ _mm256_loadu_si256((__m256i*)pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint load_linear_all(const int *pSrc)
	{
		return vint{ _mm256_load_si256((__m256i*)pSrc) };
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
#if CPPSPMD_USE_AVX2
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			return vfloat{ _mm256_loadu_ps((float*)pSrc) };
		else
			return vfloat{ _mm256_maskload_ps(pSrc, m_exec.m_mask) };
#else
		__m256 v = _mm256_loadu_ps((const float*)pSrc);

		v = _mm256_and_ps(v, _mm256_castsi256_ps(m_exec.m_mask));

		return vfloat{ v };
#endif
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
		dst.m_value = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dst.m_value), _mm256_castsi256_ps(src.m_value), _mm256_castsi256_ps(m_exec.m_mask)));
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store(const int_vref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(32) int vindex[8];
		_mm256_store_si256((__m256i*)vindex, dst.m_vindex);

		CPPSPMD_ALIGN(32) int stored[8];
		_mm256_store_si256((__m256i*)stored, src.m_value);

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		for (int i = 0; i < 8; i++)
		{
			if (mask & (1 << i))
				dst.m_pValue[vindex[i]] = stored[i];
		}
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE vint& store_all(vint& dst, const vint& src)
	{
		dst.m_value = src.m_value;
		return dst;
	}
				
	CPPSPMD_FORCE_INLINE const int_vref& store_all(const int_vref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(32) int stored[8];
		_mm256_store_si256((__m256i*)stored, src.m_value);

#if 0
		CPPSPMD_ALIGN(32) int vindex[8];
		_mm256_store_si256((__m256i*)vindex, dst.m_vindex);

		for (int i = 0; i < 8; i++)
			dst.m_pValue[vindex[i]] = stored[i];
#else
		int *pDst = dst.m_pValue;
		pDst[_mm256_extract_epi32(dst.m_vindex, 0)] = stored[0];
		pDst[_mm256_extract_epi32(dst.m_vindex, 1)] = stored[1];
		pDst[_mm256_extract_epi32(dst.m_vindex, 2)] = stored[2];
		pDst[_mm256_extract_epi32(dst.m_vindex, 3)] = stored[3];
		pDst[_mm256_extract_epi32(dst.m_vindex, 4)] = stored[4];
		pDst[_mm256_extract_epi32(dst.m_vindex, 5)] = stored[5];
		pDst[_mm256_extract_epi32(dst.m_vindex, 6)] = stored[6];
		pDst[_mm256_extract_epi32(dst.m_vindex, 7)] = stored[7];
#endif

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_vref& src)
	{
#if CPPSPMD_USE_AVX2
		return vint{ _mm256_mask_i32gather_epi32(_mm256_undefined_si256(), src.m_pValue, src.m_vindex, m_exec.m_mask, 4) };
#else
		CPPSPMD_ALIGN(32) int values[8];

		CPPSPMD_ALIGN(32) int indices[8];
		_mm256_store_si256((__m256i *)indices, src.m_vindex);

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		for (int i = 0; i < 8; i++)
		{
			if (mask & (1 << i))
				values[i] = src.m_pValue[indices[i]];
		}

		return vint{ _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values))) };
#endif
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const int_vref& src)
	{
#if CPPSPMD_USE_AVX2
		return vint{ _mm256_mask_i32gather_epi32(_mm256_undefined_si256(), src.m_pValue, src.m_vindex, m_exec.m_mask, 4) };
#else
		CPPSPMD_ALIGN(32) int values[8];

#if 0
		CPPSPMD_ALIGN(32) int indices[8];
		_mm256_store_si256((__m256i *)indices, src.m_vindex);

		for (int i = 0; i < 8; i++)
			values[i] = src.m_pValue[indices[i]];
#endif
		const int *pSrc = src.m_pValue;
		values[0] = pSrc[_mm256_extract_epi32(src.m_vindex, 0)];
		values[1] = pSrc[_mm256_extract_epi32(src.m_vindex, 1)];
		values[2] = pSrc[_mm256_extract_epi32(src.m_vindex, 2)];
		values[3] = pSrc[_mm256_extract_epi32(src.m_vindex, 3)];
		values[4] = pSrc[_mm256_extract_epi32(src.m_vindex, 4)];
		values[5] = pSrc[_mm256_extract_epi32(src.m_vindex, 5)];
		values[6] = pSrc[_mm256_extract_epi32(src.m_vindex, 6)];
		values[7] = pSrc[_mm256_extract_epi32(src.m_vindex, 7)];

		return vint{ _mm256_castps_si256( _mm256_load_ps((const float*)values)) };
#endif
	}
		
	CPPSPMD_FORCE_INLINE vint load(const cint_vref& src)
	{
#if CPPSPMD_USE_AVX2
		return vint{ _mm256_mask_i32gather_epi32(_mm256_undefined_si256(), src.m_pValue, src.m_vindex, m_exec.m_mask, 4) };
#else
		CPPSPMD_ALIGN(32) int values[8];

		CPPSPMD_ALIGN(32) int indices[8];
		_mm256_store_si256((__m256i *)indices, src.m_vindex);

		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		for (int i = 0; i < 8; i++)
		{
			if (mask & (1 << i))
				values[i] = src.m_pValue[indices[i]];
		}

		return vint{ _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(m_exec.m_mask), _mm256_load_ps((const float*)values))) };
#endif
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const cint_vref& src)
	{
#if CPPSPMD_USE_AVX2
		return vint{ _mm256_mask_i32gather_epi32(_mm256_undefined_si256(), src.m_pValue, src.m_vindex, m_exec.m_mask, 4) };
#else
		CPPSPMD_ALIGN(32) int values[8];
				
#if 0
		CPPSPMD_ALIGN(32) int indices[8];
		_mm256_store_si256((__m256i *)indices, src.m_vindex);

		for (int i = 0; i < 8; i++)
			values[i] = src.m_pValue[indices[i]];
#else
		const int *pSrc = src.m_pValue;
		values[0] = pSrc[_mm256_extract_epi32(src.m_vindex, 0)];
		values[1] = pSrc[_mm256_extract_epi32(src.m_vindex, 1)];
		values[2] = pSrc[_mm256_extract_epi32(src.m_vindex, 2)];
		values[3] = pSrc[_mm256_extract_epi32(src.m_vindex, 3)];
		values[4] = pSrc[_mm256_extract_epi32(src.m_vindex, 4)];
		values[5] = pSrc[_mm256_extract_epi32(src.m_vindex, 5)];
		values[6] = pSrc[_mm256_extract_epi32(src.m_vindex, 6)];
		values[7] = pSrc[_mm256_extract_epi32(src.m_vindex, 7)];

		return vint{ _mm256_castps_si256( _mm256_load_ps((const float*)values)) };
#endif

		return vint{ _mm256_castps_si256( _mm256_load_ps((const float*)values)) };
#endif
	}

	CPPSPMD_FORCE_INLINE void store_strided(int *pDst, uint32_t stride, const vint &v)
	{
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) pDst[0] = _mm256_extract_epi32(v.m_value, 0);
		if (mask & 2) pDst[stride] = _mm256_extract_epi32(v.m_value, 1);
		if (mask & 4) pDst[stride*2] = _mm256_extract_epi32(v.m_value, 2);
		if (mask & 8) pDst[stride*3] = _mm256_extract_epi32(v.m_value, 3);
		if (mask & 16) pDst[stride*4] = _mm256_extract_epi32(v.m_value, 4);
		if (mask & 32) pDst[stride*5] = _mm256_extract_epi32(v.m_value, 5);
		if (mask & 64) pDst[stride*6] = _mm256_extract_epi32(v.m_value, 6);
		if (mask & 128) pDst[stride*7] = _mm256_extract_epi32(v.m_value, 7);
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
		pDst[0] = _mm256_extract_epi32(v.m_value, 0);
		pDst[stride] = _mm256_extract_epi32(v.m_value, 1);
		pDst[stride*2] = _mm256_extract_epi32(v.m_value, 2);
		pDst[stride*3] = _mm256_extract_epi32(v.m_value, 3);
		pDst[stride*4] = _mm256_extract_epi32(v.m_value, 4);
		pDst[stride*5] = _mm256_extract_epi32(v.m_value, 5);
		pDst[stride*6] = _mm256_extract_epi32(v.m_value, 6);
		pDst[stride*7] = _mm256_extract_epi32(v.m_value, 7);
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
		
		__m256i v = _mm256_castps_si256(_mm256_setzero_ps());

		if (mask & 1) v = _mm256_castsi128_si256(_mm_castps_si128( _mm_load_ss( (const float *)pSrc ) ));
		if (mask & 2) v = _mm256_insert_epi32(v, pSrc[stride], 1);
		if (mask & 4) v = _mm256_insert_epi32(v, pSrc[stride*2], 2);
		if (mask & 8) v = _mm256_insert_epi32(v, pSrc[stride*3], 3);

		if (mask & 16) v = _mm256_insert_epi32(v, pSrc[stride*4], 4);
		if (mask & 32) v = _mm256_insert_epi32(v, pSrc[stride*5], 5);
		if (mask & 64) v = _mm256_insert_epi32(v, pSrc[stride*6], 6);
		if (mask & 128) v = _mm256_insert_epi32(v, pSrc[stride*7], 7);

		return vint{ v };
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
		__m256i v;

		v = _mm256_castsi128_si256(_mm_castps_si128( _mm_load_ss( (const float *)pSrc ) ));
		v = _mm256_insert_epi32(v, pSrc[stride], 1);
		v = _mm256_insert_epi32(v, pSrc[stride*2], 2);
		v = _mm256_insert_epi32(v, pSrc[stride*3], 3);

		v = _mm256_insert_epi32(v, pSrc[stride*4], 4);
		v = _mm256_insert_epi32(v, pSrc[stride*5], 5);
		v = _mm256_insert_epi32(v, pSrc[stride*6], 6);
		v = _mm256_insert_epi32(v, pSrc[stride*7], 7);

		return vint{ v };
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
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 0)]))[0] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 1)]))[1] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 2)]))[2] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 3)]))[3] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 3);

		if (mask & 16) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 4)]))[4] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 4);
		if (mask & 32) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 5)]))[5] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 5);
		if (mask & 64) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 6)]))[6] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 6);
		if (mask & 128) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 7)]))[7] = _mm256_extract_epi32(_mm256_castps_si256(src.m_value), 7);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		__m256i k = _mm256_setzero_si256();

		if (mask & 1) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 0)]))[0], 0);
		if (mask & 2) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 1)]))[1], 1);
		if (mask & 4) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 2)]))[2], 2);
		if (mask & 8) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 3)]))[3], 3);

		if (mask & 16) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 4)]))[4], 4);
		if (mask & 32) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 5)]))[5], 5);
		if (mask & 64) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 6)]))[6], 6);
		if (mask & 128) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 7)]))[7], 7);

		return vfloat{ _mm256_castsi256_ps(k) };
	}

	//

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 0)]))[0] = _mm256_extract_epi32(src.m_value, 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 1)]))[1] = _mm256_extract_epi32(src.m_value, 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 2)]))[2] = _mm256_extract_epi32(src.m_value, 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 3)]))[3] = _mm256_extract_epi32(src.m_value, 3);

		if (mask & 16) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 4)]))[4] = _mm256_extract_epi32(src.m_value, 4);
		if (mask & 32) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 5)]))[5] = _mm256_extract_epi32(src.m_value, 5);
		if (mask & 64) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 6)]))[6] = _mm256_extract_epi32(src.m_value, 6);
		if (mask & 128) ((int *)(&dst.m_pValue[_mm256_extract_epi32(dst.m_vindex, 7)]))[7] = _mm256_extract_epi32(src.m_value, 7);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const vint_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));

		__m256i k = _mm256_setzero_si256();

		if (mask & 1) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 0)]))[0], 0);
		if (mask & 2) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 1)]))[1], 1);
		if (mask & 4) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 2)]))[2], 2);
		if (mask & 8) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 3)]))[3], 3);

		if (mask & 16) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 4)]))[4], 4);
		if (mask & 32) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 5)]))[5], 5);
		if (mask & 64) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 6)]))[6], 6);
		if (mask & 128) k = _mm256_insert_epi32(k, ((int *)(&src.m_pValue[_mm256_extract_epi32(src.m_vindex, 7)]))[7], 7);

		return vint{ k };
	}
	
	// Linear integer
	struct lint
	{
		__m256i m_value;

		CPPSPMD_FORCE_INLINE explicit lint(__m256i value)
			: m_value(value)
		{ }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm256_cvtepi32_ps(m_value) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vint() const
		{
			return vint{ m_value };
		}

		int get_first_value() const 
		{
			return _mm_cvtsi128_si32(_mm256_castsi256_si128(m_value));
		}

		CPPSPMD_FORCE_INLINE float_lref operator[](float* ptr) const
		{
			return float_lref{ ptr + _mm_cvtsi128_si32(_mm256_castsi256_si128(m_value)) };
		}

		CPPSPMD_FORCE_INLINE int_lref operator[](int* ptr) const
		{
			return int_lref{ ptr + _mm_cvtsi128_si32(_mm256_castsi256_si128(m_value)) };
		}

		CPPSPMD_FORCE_INLINE int16_lref operator[](int16_t* ptr) const
		{
			return int16_lref{ ptr + _mm_cvtsi128_si32(_mm256_castsi256_si128(m_value)) };
		}

		CPPSPMD_FORCE_INLINE cint_lref operator[](const int* ptr) const
		{
			return cint_lref{ ptr + _mm_cvtsi128_si32(_mm256_castsi256_si128(m_value)) };
		}

	private:
		lint& operator=(const lint&);
	};
	
	const lint program_index = lint{ _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ) };
	
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
	return vfloat { _mm256_and_ps( _mm256_castsi256_ps(m_value), *(const __m256 *)g_onef_256 ) }; 
}

// Returns UINT32_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint() const 
{ 
	return vint { m_value };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
#if CPPSPMD_USE_AVX2
	return vbool{ _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), v.m_value) };
#else
	return vbool{ _mm256_castps_si256(_mm256_xor_ps(_mm256_load_ps((const float*)g_allones_256), _mm256_castsi256_ps(v.m_value))) };
#endif
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ xor_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) {	return exec_mask{ and_si256(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ or_si256(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_mask)) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return _mm256_movemask_ps(_mm256_castsi256_ps(e.m_mask)) != 0; }

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
CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) { return vint{ _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(b.m_value), _mm256_castsi256_ps(a.m_value), _mm256_castsi256_ps(cond.m_value))) }; }

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ _mm256_sqrt_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ _mm256_max_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ _mm256_min_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ _mm256_ceil_ps(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ _mm256_floor_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ _mm256_round_ps(a.m_value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC ) }; }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b) 
{ 
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_max_epi32(a.m_value, b.m_value) }; 
#else
	return vint{ combine_i(_mm_max_epi32(get_lo_i(a.m_value), get_lo_i(b.m_value)), _mm_max_epi32(get_hi_i(a.m_value), get_hi_i(b.m_value))) };
#endif
}

CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b) 
{	
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_min_epi32(a.m_value, b.m_value) }; 
#else
	return vint{ combine_i(_mm_min_epi32(get_lo_i(a.m_value), get_lo_i(b.m_value)), _mm_min_epi32(get_hi_i(a.m_value), get_hi_i(b.m_value))) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ _mm256_min_ps(b.m_value, _mm256_max_ps(v.m_value, a.m_value) ) };
}

CPPSPMD_FORCE_INLINE vint clamp(const vint& v, const vint& a, const vint& b)
{
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_max_epi32(a.m_value, _mm256_min_epi32(v.m_value, b.m_value) ) };
#else
	// TODO: Probably better to compute lo/hi 128-bit values, then combine, instead of using these helpers.
	__m256i lomask = compare_gt_epi32(a.m_value, v.m_value);
	__m256i himask = compare_gt_epi32(v.m_value, b.m_value);
	__m256i okmask = andnot_si256(or_si256(lomask, himask), _mm256_load_si256((const __m256i*)g_allones_256));
	return vint{ or_si256(and_si256(okmask, v.m_value), or_si256(and_si256(lomask, a.m_value), and_si256(himask, b.m_value))) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
#if CPPSPMD_USE_FMA
	return vfloat{ _mm256_fmadd_ps(a.m_value, b.m_value, c.m_value) };
#else
	return vfloat{ _mm256_add_ps(_mm256_mul_ps(a.m_value, b.m_value), c.m_value) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
#if CPPSPMD_USE_FMA
	return vfloat{ _mm256_fmsub_ps(a.m_value, b.m_value, c.m_value) };
#else
	return vfloat{ _mm256_sub_ps(_mm256_mul_ps(a.m_value, b.m_value), c.m_value) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
#if CPPSPMD_USE_FMA
	return vfloat{ _mm256_fnmadd_ps(a.m_value, b.m_value, c.m_value) };
#else
	return vfloat{ _mm256_sub_ps(c.m_value, _mm256_mul_ps(a.m_value, b.m_value)) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat vfnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
#if CPPSPMD_USE_FMA
	return vfloat{ _mm256_fnmsub_ps(a.m_value, b.m_value, c.m_value) };
#else
	return vfloat{ _mm256_sub_ps(_mm256_sub_ps(_mm256_xor_ps(a.m_value, a.m_value), _mm256_mul_ps(a.m_value, b.m_value)), c.m_value) };
#endif
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { return lint{ add_epi32(_mm256_set1_epi32(a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { return lint{ add_epi32(a.m_value, _mm256_set1_epi32(b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ and_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, const vint& b) { return vint{ or_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, int b) { return a | vint(b); }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, const vint& b) { return vint{ xor_si256(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, int b) { return a ^ vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(const vint& a, const vint& b) { return vbool{ compare_eq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vint& a, const vint& b) { return !vbool{ compare_eq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vint& a, const vint& b) { return vbool{ compare_gt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vint& a, const vint& b) { return !vbool{ compare_gt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vint& a, const vint& b) { return !vbool{ compare_gt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vint& a, const vint& b) { return vbool{ compare_gt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, const vint& b) { return vint{ add_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, const vint& b) { return vint{ sub_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, int b) { return a + vint(b); }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, int b) { return a - vint(b); }
CPPSPMD_FORCE_INLINE vint operator+(int a, const vint& b) { return vint(a) + b; }
CPPSPMD_FORCE_INLINE vint operator-(int a, const vint& b) { return vint(a) - b; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, const vint& b) { return vint{ mullo_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, int b) { return a * vint(b); }
CPPSPMD_FORCE_INLINE vint operator*(int a, const vint& b) { return vint(a) * b; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ sub_epi32(_mm256_setzero_si256(), v.m_value) }; }

CPPSPMD_FORCE_INLINE int safe_div(int a, int b) { return b ? (a / b) : 0; }
CPPSPMD_FORCE_INLINE int safe_mod(int a, int b) { return b ? (a % b) : 0; }

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[8];
	
	// Why safe_div()? Because of dead lanes. And we don't have the exec mask here.
	result[0] = safe_div(_mm256_extract_epi32(a.m_value, 0), _mm256_extract_epi32(b.m_value, 0));
	result[1] = safe_div(_mm256_extract_epi32(a.m_value, 1), _mm256_extract_epi32(b.m_value, 1));
	result[2] = safe_div(_mm256_extract_epi32(a.m_value, 2), _mm256_extract_epi32(b.m_value, 2));
	result[3] = safe_div(_mm256_extract_epi32(a.m_value, 3), _mm256_extract_epi32(b.m_value, 3));

	result[4] = safe_div(_mm256_extract_epi32(a.m_value, 4), _mm256_extract_epi32(b.m_value, 4));
	result[5] = safe_div(_mm256_extract_epi32(a.m_value, 5), _mm256_extract_epi32(b.m_value, 5));
	result[6] = safe_div(_mm256_extract_epi32(a.m_value, 6), _mm256_extract_epi32(b.m_value, 6));
	result[7] = safe_div(_mm256_extract_epi32(a.m_value, 7), _mm256_extract_epi32(b.m_value, 7));

	return vint{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, int b)
{
	CPPSPMD_ALIGN(32) int result[8];

	if (!b)
		return vint{ _mm256_setzero_si256() };
		
	result[0] = _mm256_extract_epi32(a.m_value, 0) / b;
	result[1] = _mm256_extract_epi32(a.m_value, 1) / b;
	result[2] = _mm256_extract_epi32(a.m_value, 2) / b;
	result[3] = _mm256_extract_epi32(a.m_value, 3) / b;

	result[4] = _mm256_extract_epi32(a.m_value, 4) / b;
	result[5] = _mm256_extract_epi32(a.m_value, 5) / b;
	result[6] = _mm256_extract_epi32(a.m_value, 6) / b;
	result[7] = _mm256_extract_epi32(a.m_value, 7) / b;

	return vint{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[8];
	result[0] = safe_mod(_mm256_extract_epi32(a.m_value, 0), _mm256_extract_epi32(b.m_value, 0));
	result[1] = safe_mod(_mm256_extract_epi32(a.m_value, 1), _mm256_extract_epi32(b.m_value, 1));
	result[2] = safe_mod(_mm256_extract_epi32(a.m_value, 2), _mm256_extract_epi32(b.m_value, 2));
	result[3] = safe_mod(_mm256_extract_epi32(a.m_value, 3), _mm256_extract_epi32(b.m_value, 3));

	result[4] = safe_mod(_mm256_extract_epi32(a.m_value, 4), _mm256_extract_epi32(b.m_value, 4));
	result[5] = safe_mod(_mm256_extract_epi32(a.m_value, 5), _mm256_extract_epi32(b.m_value, 5));
	result[6] = safe_mod(_mm256_extract_epi32(a.m_value, 6), _mm256_extract_epi32(b.m_value, 6));
	result[7] = safe_mod(_mm256_extract_epi32(a.m_value, 7), _mm256_extract_epi32(b.m_value, 7));

	return vint{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b)
{
	CPPSPMD_ALIGN(32) int result[8];

	if (!b)
		return vint{ _mm256_setzero_si256() };

	result[0] = _mm256_extract_epi32(a.m_value, 0) % b;
	result[1] = _mm256_extract_epi32(a.m_value, 1) % b;
	result[2] = _mm256_extract_epi32(a.m_value, 2) % b;
	result[3] = _mm256_extract_epi32(a.m_value, 3) % b;

	result[4] = _mm256_extract_epi32(a.m_value, 4) % b;
	result[5] = _mm256_extract_epi32(a.m_value, 5) % b;
	result[6] = _mm256_extract_epi32(a.m_value, 6) % b;
	result[7] = _mm256_extract_epi32(a.m_value, 7) % b;

	return vint{ _mm256_load_si256((__m256i*)result) };
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b)
{
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_sllv_epi32(a.m_value, b.m_value) };
#else
	CPPSPMD_ALIGN(32) int a_values[8];
	CPPSPMD_ALIGN(32) int b_values[8];

	_mm256_store_si256((__m256i*)a_values, a.m_value);
	_mm256_store_si256((__m256i*)b_values, b.m_value);

	int result[8];
	for (uint32_t i = 0; i < 8; i++)
		result[i] = a_values[i] << b_values[i];

	return vint{ _mm256_load_si256((__m256i*)result) };
#endif
}

// uniform shift left
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ combine_i(_mm_sll_epi32(get_lo_i(a.m_value), bv), _mm_sll_epi32(get_hi_i(a.m_value), bv)) };
}

// uniform arithmetic shift right
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ combine_i(_mm_sra_epi32(get_lo_i(a.m_value), bv), _mm_sra_epi32(get_hi_i(a.m_value), bv)) };
}

// uniform shift right
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ combine_i(_mm_srl_epi32(get_lo_i(a.m_value), bv), _mm_srl_epi32(get_hi_i(a.m_value), bv)) };
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b)
{
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_srav_epi32(a.m_value, b.m_value) };
#else
	CPPSPMD_ALIGN(32) int a_values[8];
	CPPSPMD_ALIGN(32) int b_values[8];

	_mm256_store_si256((__m256i*)a_values, a.m_value);
	_mm256_store_si256((__m256i*)b_values, b.m_value);

	int result[8];
	for (uint32_t i = 0; i < 8; i++)
		result[i] = a_values[i] >> b_values[i];

	return vint{ _mm256_load_si256((__m256i*)result) };
#endif
}

// This is very slow without AVX2
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b)
{
#if CPPSPMD_USE_AVX2
	return vint{ _mm256_srlv_epi32(a.m_value, b.m_value) };
#else
	CPPSPMD_ALIGN(32) uint32_t a_values[8];
	CPPSPMD_ALIGN(32) uint32_t b_values[8];

	_mm256_store_si256((__m256i*)a_values, a.m_value);
	_mm256_store_si256((__m256i*)b_values, b.m_value);

	uint32_t result[8];
	for (uint32_t i = 0; i < 8; i++)
		result[i] = a_values[i] >> b_values[i];

	return vint{ _mm256_load_si256((__m256i*)result) };
#endif
}

CPPSPMD_FORCE_INLINE vint create_vint(__m256i v) { return vint{ v }; }
CPPSPMD_FORCE_INLINE vint create_vint(__m128i lo, __m128i hi) { return vint{ _mm256_setr_m128i(lo, hi) }; }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) CPPSPMD::create_vint( _mm_slli_epi32( CPPSPMD::get_lo_i((a).m_value), (b) ), _mm_slli_epi32( CPPSPMD::get_hi_i((a).m_value), (b) ) )
#define VINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( _mm_srai_epi32( CPPSPMD::get_lo_i((a).m_value), (b) ), _mm_srai_epi32( CPPSPMD::get_hi_i((a).m_value), (b) ) )
#define VUINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( _mm_srli_epi32( CPPSPMD::get_lo_i((a).m_value), (b) ), _mm_srli_epi32( CPPSPMD::get_hi_i((a).m_value), (b) ) )

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, const lint& b) { return vbool{ compare_eq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, int b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint& b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint& a, const lint& b) { return vbool{ compare_gt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const lint& a, const lint& b) { return vbool{ compare_gt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const lint& a, const lint& b) { return !vbool{ compare_gt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const lint& a, const lint& b) { return !vbool{ compare_gt_epi32(b.m_value, a.m_value) }; }

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) float values[8]; _mm256_store_ps(values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const vint& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm256_store_si256((__m256i*)values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const lint& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm256_store_si256((__m256i*)values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) { assert(instance < 8); CPPSPMD_ALIGN(32) int values[8]; _mm256_store_si256((__m256i*)values, v.m_value);	return values[instance] != 0; }

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

#define VINT_EXTRACT(v, instance) _mm256_extract_epi32((v).m_value, instance)
#define VBOOL_EXTRACT(v, instance) _mm256_extract_epi32((v).m_value, instance)
#define VFLOAT_EXTRACT(result, v, instance) do { int _v = _mm256_extract_epi32(_mm256_castps_si256(v.m_value), instance); result = *(const float *)&_v; } while(0)

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
	CPPSPMD_ALIGN(32) int vindex[8];
	_mm256_store_si256((__m256i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(32) float stored[8];
	_mm256_store_ps(stored, src.m_value);

	int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
	for (int i = 0; i < 8; i++)
	{
		if (mask & (1 << i))
			dst.m_pValue[vindex[i]] = stored[i];
	}
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(32) float stored[8];
	_mm256_store_ps(stored, src.m_value);

#if 0
	CPPSPMD_ALIGN(32) int vindex[8];
	_mm256_store_si256((__m256i*)vindex, dst.m_vindex);
		
	for (int i = 0; i < 8; i++)
		dst.m_pValue[vindex[i]] = stored[i];
#else
	float *pDst = dst.m_pValue;
	pDst[_mm256_extract_epi32(dst.m_vindex, 0)] = stored[0];
	pDst[_mm256_extract_epi32(dst.m_vindex, 1)] = stored[1];
	pDst[_mm256_extract_epi32(dst.m_vindex, 2)] = stored[2];
	pDst[_mm256_extract_epi32(dst.m_vindex, 3)] = stored[3];
	pDst[_mm256_extract_epi32(dst.m_vindex, 4)] = stored[4];
	pDst[_mm256_extract_epi32(dst.m_vindex, 5)] = stored[5];
	pDst[_mm256_extract_epi32(dst.m_vindex, 6)] = stored[6];
	pDst[_mm256_extract_epi32(dst.m_vindex, 7)] = stored[7];
#endif

	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(32) int vindex[8];
	_mm256_store_si256((__m256i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(32) float stored[8];
	_mm256_store_ps(stored, src.m_value);

	int mask = _mm256_movemask_ps(_mm256_castsi256_ps(m_exec.m_mask));
	for (int i = 0; i < 8; i++)
	{
		if (mask & (1 << i))
			dst.m_pValue[vindex[i]] = stored[i];
	}
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(32) float stored[8];
	_mm256_store_ps(stored, src.m_value);

#if 0
	CPPSPMD_ALIGN(32) int vindex[8];
	_mm256_store_si256((__m256i*)vindex, dst.m_vindex);

	for (int i = 0; i < 8; i++)
		dst.m_pValue[vindex[i]] = stored[i];
#else
	float *pDst = dst.m_pValue;
	pDst[_mm256_extract_epi32(dst.m_vindex, 0)] = stored[0];
	pDst[_mm256_extract_epi32(dst.m_vindex, 1)] = stored[1];
	pDst[_mm256_extract_epi32(dst.m_vindex, 2)] = stored[2];
	pDst[_mm256_extract_epi32(dst.m_vindex, 3)] = stored[3];
	pDst[_mm256_extract_epi32(dst.m_vindex, 4)] = stored[4];
	pDst[_mm256_extract_epi32(dst.m_vindex, 5)] = stored[5];
	pDst[_mm256_extract_epi32(dst.m_vindex, 6)] = stored[6];
	pDst[_mm256_extract_epi32(dst.m_vindex, 7)] = stored[7];
#endif

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
	if (mask == ALL_ON_MOVEMASK)
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

	if (mask == ALL_ON_MOVEMASK)
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
		all_flag = (mask == ALL_ON_MOVEMASK);

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

		all_flag = (mask == ALL_ON_MOVEMASK);
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

		if (mask == ALL_ON_MOVEMASK)
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

		all_flag = (mask == ALL_ON_MOVEMASK);
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
			exec_mask partial_mask = exec_mask{ compare_gt_epi32(_mm256_set1_epi32(total_partial), program_index.m_value) };
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

} // namespace cppspmd_avx2
