// cppspmd_sse.h
// SSE 4.1
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

#undef CPPSPMD_SSE
#undef CPPSPMD_AVX1
#undef CPPSPMD_AVX2
#undef CPPSPMD_AVX
#undef CPPSPMD_FLOAT4

#define CPPSPMD_SSE 1
#define CPPSPMD_AVX 0
#define CPPSPMD_AVX1 0
#define CPPSPMD_AVX2 0
#define CPPSPMD_FLOAT4 0

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

#define CPPSPMD cppspmd_sse41
#define CPPSPMD_ARCH _sse41

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
CPPSPMD_DECL(uint32_t, g_x_128[4]) = { UINT32_MAX, 0, 0, 0 };
CPPSPMD_DECL(float, g_onef_128[4]) = { 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(uint32_t, g_oneu_128[4]) = { 1, 1, 1, 1 };

const uint32_t ALL_ON_MOVEMASK = 0xF;

struct spmd_kernel
{
	struct vint;
	struct vbool;
	struct vfloat;
		
	// Exec mask
	struct exec_mask
	{
		__m128i m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);
		CPPSPMD_FORCE_INLINE explicit exec_mask(const __m128i& mask) : m_mask(mask) { }
				
		static CPPSPMD_FORCE_INLINE exec_mask all_on()	{ return exec_mask{ _mm_load_si128((const __m128i*)g_allones_128) };	}
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ _mm_setzero_si128() }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const { return _mm_movemask_ps(_mm_castsi128_ps(m_mask)); }
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
		__m128i m_value;

		CPPSPMD_FORCE_INLINE vbool(bool value) : m_value(_mm_set1_epi32(value ? UINT32_MAX : 0)) { }

		CPPSPMD_FORCE_INLINE explicit vbool(const __m128i& value) : m_value(value) { }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const;
		CPPSPMD_FORCE_INLINE explicit operator vint() const;
								
	private:
		vbool& operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(dst.m_value), _mm_castsi128_ps(src.m_value), _mm_castsi128_ps(m_exec.m_mask)));
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
		__m128 m_value;

		vfloat() = default;

		CPPSPMD_FORCE_INLINE explicit vfloat(const __m128& v) : m_value(v) { }

		CPPSPMD_FORCE_INLINE vfloat(float value) : m_value(_mm_set1_ps(value)) { }

		CPPSPMD_FORCE_INLINE explicit vfloat(int value) : m_value(_mm_set1_ps((float)value)) { }

	private:
		vfloat& operator=(const vfloat&);
	};

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat& dst, const vfloat& src)
	{
		dst.m_value = _mm_blendv_ps(dst.m_value, src.m_value, _mm_castsi128_ps(m_exec.m_mask));
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat&& dst, const vfloat& src)
	{
		dst.m_value = _mm_blendv_ps(dst.m_value, src.m_value, _mm_castsi128_ps(m_exec.m_mask));
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
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm_storeu_ps(dst.m_pValue, src.m_value);
		else
			_mm_storeu_ps(dst.m_pValue, _mm_blendv_ps(_mm_loadu_ps(dst.m_pValue), src.m_value, _mm_castsi128_ps(m_exec.m_mask)));
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref&& dst, const vfloat& src)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm_storeu_ps(dst.m_pValue, src.m_value);
		else
			_mm_storeu_ps(dst.m_pValue, _mm_blendv_ps(_mm_loadu_ps(dst.m_pValue), src.m_value, _mm_castsi128_ps(m_exec.m_mask)));
		return dst;
	}
	
	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref& dst, const vfloat& src)
	{
		_mm_storeu_ps(dst.m_pValue, src.m_value);
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref&& dst, const vfloat& src)
	{
		_mm_storeu_ps(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const float_lref& src)
	{
		return vfloat{ _mm_and_ps(_mm_loadu_ps(src.m_pValue), _mm_castsi128_ps(m_exec.m_mask)) };
	}
		
	// Varying ref to floats
	struct float_vref
	{
		__m128i m_vindex;
		float* m_pValue;
		
	private:
		float_vref& operator=(const float_vref&);
	};

	// Varying ref to varying float
	struct vfloat_vref
	{
		__m128i m_vindex;
		vfloat* m_pValue;
		
	private:
		vfloat_vref& operator=(const vfloat_vref&);
	};

	// Varying ref to varying int
	struct vint_vref
	{
		__m128i m_vindex;
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
		CPPSPMD_ALIGN(16) int vindex[4];
		_mm_store_si128((__m128i *)vindex, src.m_vindex);

		CPPSPMD_ALIGN(16) float loaded[4];

		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		for (int i = 0; i < 4; i++)
		{
			if (mask & (1 << i))
				loaded[i] = src.m_pValue[vindex[i]];
		}
		return vfloat{ _mm_and_ps(_mm_castsi128_ps(m_exec.m_mask), _mm_load_ps((const float*)loaded)) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
		CPPSPMD_ALIGN(16) int vindex[4];
		_mm_store_si128((__m128i *)vindex, src.m_vindex);

		CPPSPMD_ALIGN(16) float loaded[4];

		for (int i = 0; i < 4; i++)
			loaded[i] = src.m_pValue[vindex[i]];
		return vfloat{ _mm_load_ps((const float*)loaded) };
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
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
		{
			_mm_storeu_si128((__m128i *)dst.m_pValue, src.m_value);
		}
		else
		{
			CPPSPMD_ALIGN(16) int stored[4];
			_mm_store_si128((__m128i *)stored, src.m_value);

			for (int i = 0; i < 4; i++)
			{
				if (mask & (1 << i))
					dst.m_pValue[i] = stored[i];
			}
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_lref& src)
	{
		__m128i v = _mm_loadu_si128((const __m128i*)src.m_pValue);

		v = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(m_exec.m_mask)));

		return vint{ v };
	}

	// Linear ref to int16's
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};

	CPPSPMD_FORCE_INLINE const int16_lref& store(const int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(16) int stored[4];
		_mm_store_si128((__m128i *)stored, src.m_value);

		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		for (int i = 0; i < 4; i++)
		{
			if (mask & (1 << i))
				dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		}
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int16_lref& store_all(const int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(16) int stored[4];
		_mm_store_si128((__m128i *)stored, src.m_value);

		for (int i = 0; i < 4; i++)
			dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		return dst;
	}
		
	CPPSPMD_FORCE_INLINE vint load(const int16_lref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		for (int i = 0; i < 4; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		__m128i t = _mm_load_si128( (const __m128i *)values );

		return vint{ _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps( t ), _mm_castsi128_ps(m_exec.m_mask))) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const int16_lref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		for (int i = 0; i < 4; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		__m128i t = _mm_load_si128( (const __m128i *)values );

		return vint{ t };
	}
		
	// Linear ref to constant ints
	struct cint_lref
	{
		const int* m_pValue;

	private:
		cint_lref& operator=(const cint_lref&);
	};

	CPPSPMD_FORCE_INLINE vint load(const cint_lref& src)
	{
		__m128i v = _mm_loadu_si128((const __m128i *)src.m_pValue);
		v = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(m_exec.m_mask)));
		return vint{ v };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_lref& src)
	{
		return vint{ _mm_loadu_si128((const __m128i *)src.m_pValue) };
	}
	
	// Varying ref to ints
	struct int_vref
	{
		__m128i m_vindex;
		int* m_pValue;

	private:
		int_vref& operator=(const int_vref&);
	};

	// Varying ref to constant ints
	struct cint_vref
	{
		__m128i m_vindex;
		const int* m_pValue;

	private:
		cint_vref& operator=(const cint_vref&);
	};

	// Varying int
	struct vint
	{
		__m128i m_value;

		vint() = default;

		CPPSPMD_FORCE_INLINE explicit vint(const __m128i& value) : m_value(value)	{ }

		CPPSPMD_FORCE_INLINE vint(int value) : m_value(_mm_set1_epi32(value)) { }

		CPPSPMD_FORCE_INLINE explicit vint(float value) : m_value(_mm_set1_epi32((int)value))	{ }

		CPPSPMD_FORCE_INLINE explicit vint(const vfloat& other) : m_value(_mm_cvttps_epi32(other.m_value)) { }

		CPPSPMD_FORCE_INLINE explicit operator vbool() const 
		{
			return vbool{ _mm_xor_si128( _mm_load_si128((const __m128i*)g_allones_128), _mm_cmpeq_epi32(m_value, _mm_setzero_si128())) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm_cvtepi32_ps(m_value) };
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

	// Load/store linear int
	CPPSPMD_FORCE_INLINE void storeu_linear(int *pDst, const vint& src)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm_storeu_si128((__m128i *)pDst, src.m_value);
		else
		{
			if (mask & 1) pDst[0] = _mm_extract_epi32(src.m_value, 0);
			if (mask & 2) pDst[1] = _mm_extract_epi32(src.m_value, 1);
			if (mask & 4) pDst[2] = _mm_extract_epi32(src.m_value, 2);
			if (mask & 8) pDst[3] = _mm_extract_epi32(src.m_value, 3);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int *pDst, const vint& src)
	{
		_mm_storeu_si128((__m128i*)pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int *pDst, const vint& src)
	{
		_mm_store_si128((__m128i*)pDst, src.m_value);
	}
		
	CPPSPMD_FORCE_INLINE vint loadu_linear(const int *pSrc)
	{
		__m128i v = _mm_loadu_si128((const __m128i*)pSrc);

		v = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(m_exec.m_mask)));

		return vint{ v };
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear_all(const int *pSrc)
	{
		return vint{ _mm_loadu_si128((__m128i*)pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint load_linear_all(const int *pSrc)
	{
		return vint{ _mm_load_si128((__m128i*)pSrc) };
	}

	// Load/store linear float
	CPPSPMD_FORCE_INLINE void storeu_linear(float *pDst, const vfloat& src)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		if (mask == ALL_ON_MOVEMASK)
			_mm_storeu_ps((float*)pDst, src.m_value);
		else
		{
			int *pDstI = (int *)pDst;
			if (mask & 1) pDstI[0] = _mm_extract_ps(src.m_value, 0);
			if (mask & 2) pDstI[1] = _mm_extract_ps(src.m_value, 1);
			if (mask & 4) pDstI[2] = _mm_extract_ps(src.m_value, 2);
			if (mask & 8) pDstI[3] = _mm_extract_ps(src.m_value, 3);
		}
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(float *pDst, const vfloat& src)
	{
		_mm_storeu_ps((float*)pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(float *pDst, const vfloat& src)
	{
		_mm_store_ps((float*)pDst, src.m_value);
	}
		
	CPPSPMD_FORCE_INLINE vfloat loadu_linear(const float *pSrc)
	{
		__m128 v = _mm_loadu_ps((const float*)pSrc);

		v = _mm_and_ps(v, _mm_castsi128_ps(m_exec.m_mask));

		return vfloat{ v };
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear_all(const float *pSrc)
	{
		return vfloat{ _mm_loadu_ps((float*)pSrc) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_linear_all(const float *pSrc)
	{
		return vfloat{ _mm_load_ps((float*)pSrc) };
	}
	
	CPPSPMD_FORCE_INLINE vint& store(vint& dst, const vint& src)
	{
		dst.m_value = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(dst.m_value), _mm_castsi128_ps(src.m_value), _mm_castsi128_ps(m_exec.m_mask)));
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store(const int_vref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(16) int vindex[4];
		_mm_store_si128((__m128i*)vindex, dst.m_vindex);

		CPPSPMD_ALIGN(16) int stored[4];
		_mm_store_si128((__m128i*)stored, src.m_value);

		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		for (int i = 0; i < 4; i++)
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
		CPPSPMD_ALIGN(16) int vindex[4];
		_mm_store_si128((__m128i*)vindex, dst.m_vindex);

		CPPSPMD_ALIGN(16) int stored[4];
		_mm_store_si128((__m128i*)stored, src.m_value);

		for (int i = 0; i < 4; i++)
			dst.m_pValue[vindex[i]] = stored[i];

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_vref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		CPPSPMD_ALIGN(16) int indices[4];
		_mm_store_si128((__m128i *)indices, src.m_vindex);

		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		for (int i = 0; i < 4; i++)
		{
			if (mask & (1 << i))
				values[i] = src.m_pValue[indices[i]];
		}

		return vint{ _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(m_exec.m_mask), _mm_load_ps((const float*)values))) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const int_vref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		CPPSPMD_ALIGN(16) int indices[4];
		_mm_store_si128((__m128i *)indices, src.m_vindex);

		for (int i = 0; i < 4; i++)
			values[i] = src.m_pValue[indices[i]];

		return vint{ _mm_castps_si128( _mm_load_ps((const float*)values)) };
	}
		
	CPPSPMD_FORCE_INLINE vint load(const cint_vref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		CPPSPMD_ALIGN(16) int indices[4];
		_mm_store_si128((__m128i *)indices, src.m_vindex);

		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		for (int i = 0; i < 4; i++)
		{
			if (mask & (1 << i))
				values[i] = src.m_pValue[indices[i]];
		}

		return vint{ _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(m_exec.m_mask), _mm_load_ps((const float*)values))) };
	}
		
	CPPSPMD_FORCE_INLINE vint load_all(const cint_vref& src)
	{
		CPPSPMD_ALIGN(16) int values[4];

		CPPSPMD_ALIGN(16) int indices[4];
		_mm_store_si128((__m128i *)indices, src.m_vindex);

		for (int i = 0; i < 4; i++)
			values[i] = src.m_pValue[indices[i]];

		return vint{ _mm_castps_si128( _mm_load_ps((const float*)values)) };
	}

	CPPSPMD_FORCE_INLINE void store_strided(int *pDst, uint32_t stride, const vint &v)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		
		if (mask & 1) pDst[0] = _mm_extract_epi32(v.m_value, 0);
		if (mask & 2) pDst[stride] = _mm_extract_epi32(v.m_value, 1);
		if (mask & 4) pDst[stride*2] = _mm_extract_epi32(v.m_value, 2);
		if (mask & 8) pDst[stride*3] = _mm_extract_epi32(v.m_value, 3);
	}

	CPPSPMD_FORCE_INLINE void store_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));

		if (mask & 1) ((int *)pDstF)[0] = _mm_extract_ps(v.m_value, 0);
		if (mask & 2) ((int *)pDstF)[stride] = _mm_extract_ps(v.m_value, 1);
		if (mask & 4) ((int *)pDstF)[stride*2] = _mm_extract_ps(v.m_value, 2);
		if (mask & 8) ((int *)pDstF)[stride*3] = _mm_extract_ps(v.m_value, 3);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(int *pDst, uint32_t stride, const vint &v)
	{
		pDst[0] = _mm_extract_epi32(v.m_value, 0);
		pDst[stride] = _mm_extract_epi32(v.m_value, 1);
		pDst[stride*2] = _mm_extract_epi32(v.m_value, 2);
		pDst[stride*3] = _mm_extract_epi32(v.m_value, 3);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(float *pDstF, uint32_t stride, const vfloat &v)
	{
		((int *)pDstF)[0] = _mm_extract_ps(v.m_value, 0);
		((int *)pDstF)[stride] = _mm_extract_ps(v.m_value, 1);
		((int *)pDstF)[stride*2] = _mm_extract_ps(v.m_value, 2);
		((int *)pDstF)[stride*3] = _mm_extract_ps(v.m_value, 3);
	}

	CPPSPMD_FORCE_INLINE vint load_strided(const int *pSrc, uint32_t stride)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		
		const float *pSrcF = (const float *)pSrc;
		
		__m128 v = _mm_setzero_ps();
		if (mask & 1) v = _mm_load_ss(pSrcF);
		if (mask & 2) v = _mm_insert_ps(v, _mm_load_ss(pSrcF + stride), 0x10);
		if (mask & 4) v = _mm_insert_ps(v, _mm_load_ss(pSrcF + 2 * stride), 0x20);
		if (mask & 8) v = _mm_insert_ps(v, _mm_load_ss(pSrcF + 3 * stride), 0x30);

		return vint{ _mm_castps_si128(v) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_strided(const float *pSrc, uint32_t stride)
	{
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		
		__m128 v = _mm_setzero_ps();
		if (mask & 1) v = _mm_load_ss(pSrc);
		if (mask & 2) v = _mm_insert_ps(v, _mm_load_ss(pSrc + stride), 0x10);
		if (mask & 4) v = _mm_insert_ps(v, _mm_load_ss(pSrc + 2 * stride), 0x20);
		if (mask & 8) v = _mm_insert_ps(v, _mm_load_ss(pSrc + 3 * stride), 0x30);

		return vfloat{ v };
	}

	CPPSPMD_FORCE_INLINE vint load_all_strided(const int *pSrc, uint32_t stride)
	{
		const float *pSrcF = (const float *)pSrc;
		
		__m128 v = _mm_load_ss(pSrcF);
		v = _mm_insert_ps(v, _mm_load_ss(pSrcF + stride), 0x10);
		v = _mm_insert_ps(v, _mm_load_ss(pSrcF + 2 * stride), 0x20);
		v = _mm_insert_ps(v, _mm_load_ss(pSrcF + 3 * stride), 0x30);

		return vint{ _mm_castps_si128(v) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all_strided(const float *pSrc, uint32_t stride)
	{
		__m128 v = _mm_load_ss(pSrc);
		v = _mm_insert_ps(v, _mm_load_ss(pSrc + stride), 0x10);
		v = _mm_insert_ps(v, _mm_load_ss(pSrc + 2 * stride), 0x20);
		v = _mm_insert_ps(v, _mm_load_ss(pSrc + 3 * stride), 0x30);

		return vfloat{ v };
	}

	CPPSPMD_FORCE_INLINE const vfloat_vref& store(const vfloat_vref& dst, const vfloat& src)
	{
		// TODO: There's surely a better way
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 0)]))[0] = _mm_extract_epi32(_mm_castps_si128(src.m_value), 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 1)]))[1] = _mm_extract_epi32(_mm_castps_si128(src.m_value), 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 2)]))[2] = _mm_extract_epi32(_mm_castps_si128(src.m_value), 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 3)]))[3] = _mm_extract_epi32(_mm_castps_si128(src.m_value), 3);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));

		__m128i k = _mm_setzero_si128();

		if (mask & 1) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 0)]))[0], 0);
		if (mask & 2) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 1)]))[1], 1);
		if (mask & 4) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 2)]))[2], 2);
		if (mask & 8) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 3)]))[3], 3);

		return vfloat{ _mm_castsi128_ps(k) };
	}

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		// TODO: There's surely a better way
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
		
		if (mask & 1) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 0)]))[0] = _mm_extract_epi32(src.m_value, 0);
		if (mask & 2) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 1)]))[1] = _mm_extract_epi32(src.m_value, 1);
		if (mask & 4) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 2)]))[2] = _mm_extract_epi32(src.m_value, 2);
		if (mask & 8) ((int *)(&dst.m_pValue[_mm_extract_epi32(dst.m_vindex, 3)]))[3] = _mm_extract_epi32(src.m_value, 3);

		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(vint_vref& src)
	{
		// TODO: There's surely a better way
		int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));

		__m128i k = _mm_setzero_si128();

		if (mask & 1) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 0)]))[0], 0);
		if (mask & 2) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 1)]))[1], 1);
		if (mask & 4) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 2)]))[2], 2);
		if (mask & 8) k = _mm_insert_epi32(k, ((int *)(&src.m_pValue[_mm_extract_epi32(src.m_vindex, 3)]))[3], 3);

		return vint{ k };
	}
			
	// Linear integer
	struct lint
	{
		__m128i m_value;

		CPPSPMD_FORCE_INLINE explicit lint(__m128i value)
			: m_value(value)
		{ }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm_cvtepi32_ps(m_value) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vint() const
		{
			return vint{ m_value };
		}

		CPPSPMD_FORCE_INLINE int get_first_value() const 
		{
			return _mm_cvtsi128_si32(m_value);
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
	
	const lint program_index = lint{ _mm_set_epi32( 3, 2, 1, 0 ) };
	
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
			m_pKernel->m_in_loop = m_prev_in_loop;
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
	return vfloat { _mm_and_ps( _mm_castsi128_ps(m_value), *(const __m128 *)g_onef_128 ) }; 
}
	
// Returns UINT32_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint() const 
{ 
	return vint { m_value };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
	return vbool{ _mm_castps_si128(_mm_xor_ps(_mm_load_ps((const float*)g_allones_128), _mm_castsi128_ps(v.m_value))) };
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm_xor_si128(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) {	return exec_mask{ _mm_and_si128(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm_or_si128(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return _mm_movemask_ps(_mm_castsi128_ps(e.m_mask)) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return _mm_movemask_ps(_mm_castsi128_ps(e.m_mask)) != 0; }

CPPSPMD_FORCE_INLINE bool all(const vbool& e) { return _mm_movemask_ps(_mm_castsi128_ps(e.m_value)) == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const vbool& e) { return _mm_movemask_ps(_mm_castsi128_ps(e.m_value)) != 0; }

CPPSPMD_FORCE_INLINE exec_mask andnot(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm_andnot_si128(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE vbool operator||(const vbool& a, const vbool& b) { return vbool{ _mm_or_si128(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator&&(const vbool& a, const vbool& b) { return vbool{ _mm_and_si128(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, const vfloat& b) { return vfloat{ _mm_add_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vfloat& b) {	return vfloat{ _mm_sub_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const vfloat& b) { return vfloat(a) + b; }
CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, float b) { return a + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vint& b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vint& a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, int b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(int a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, float b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(float a, const vfloat& b) { return vfloat(a) - b; }

CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, const vfloat& b) { return vfloat{ _mm_mul_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, float b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float a, const vfloat& b) { return vfloat(a) * b; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, int b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(int a, const vfloat& b) { return vfloat(a) * b; }

CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, const vfloat& b) {	return vfloat{ _mm_div_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, int b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(int a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, float b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(float a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& v) { return vfloat{ _mm_sub_ps(_mm_xor_ps(v.m_value, v.m_value), v.m_value) }; }

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, const vfloat& b) { return vbool{ _mm_castps_si128(_mm_cmpeq_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, float b) { return a == vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, const vfloat& b) { return !vbool{ _mm_castps_si128(_mm_cmpeq_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, float b) { return a != vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, const vfloat& b) { return vbool{ _mm_castps_si128(_mm_cmplt_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, float b) { return a < vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, const vfloat& b) { return vbool{ _mm_castps_si128(_mm_cmpgt_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, float b) { return a > vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, const vfloat& b) { return vbool{ _mm_castps_si128(_mm_cmple_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, float b) { return a <= vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, const vfloat& b) { return vbool{ _mm_castps_si128(_mm_cmpge_ps(a.m_value, b.m_value)) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, float b) { return a >= vfloat(b); }

CPPSPMD_FORCE_INLINE vfloat spmd_ternaryf(const vbool& cond, const vfloat& a, const vfloat& b) { return vfloat{ _mm_blendv_ps(b.m_value, a.m_value, _mm_castsi128_ps(cond.m_value)) }; }
CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) { return vint{ _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b.m_value), _mm_castsi128_ps(a.m_value), _mm_castsi128_ps(cond.m_value))) }; }

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ _mm_sqrt_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ _mm_andnot_ps(_mm_set1_ps(-0.0f), v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ _mm_max_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) {	return vfloat{ _mm_min_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ _mm_ceil_ps(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ _mm_floor_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat &a) { return vfloat{ _mm_round_ps(a.m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat &a) { return vfloat{ _mm_round_ps(a.m_value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC ) }; }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b) { return vint{ _mm_max_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b) {	return vint{ _mm_min_epi32(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint cast_vfloat_to_vint(const vfloat& v) { return vint{ _mm_castps_si128(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat cast_vint_to_vfloat(const vint& v) { return vfloat{ _mm_castsi128_ps(v.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ _mm_min_ps(b.m_value, _mm_max_ps(v.m_value, a.m_value) ) };
}

CPPSPMD_FORCE_INLINE vint clamp(const vint& v, const vint& a, const vint& b)
{
	return vint{ _mm_min_epi32(b.m_value, _mm_max_epi32(v.m_value, a.m_value) ) };
}

CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm_add_ps(_mm_mul_ps(a.m_value, b.m_value), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm_sub_ps(_mm_mul_ps(a.m_value, b.m_value), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm_sub_ps(c.m_value, _mm_mul_ps(a.m_value, b.m_value)) };
}

CPPSPMD_FORCE_INLINE vfloat vfnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm_sub_ps(_mm_sub_ps(_mm_xor_ps(a.m_value, a.m_value), _mm_mul_ps(a.m_value, b.m_value)), c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat &x, const vfloat &y, const vfloat &s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { return lint{ _mm_add_epi32(_mm_set1_epi32(a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { return lint{ _mm_add_epi32(a.m_value, _mm_set1_epi32(b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ _mm_and_si128(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, const vint& b) { return vint{ _mm_or_si128(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, int b) { return a | vint(b); }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, const vint& b) { return vint{ _mm_xor_si128(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, int b) { return a ^ vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(const vint& a, const vint& b) { return vbool{ _mm_cmpeq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator!=(const vint& a, const vint& b) { return !vbool{ _mm_cmpeq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<(const vint& a, const vint& b) { return vbool{ _mm_cmpgt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const vint& a, const vint& b) { return !vbool{ _mm_cmpgt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const vint& a, const vint& b) { return !vbool{ _mm_cmpgt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const vint& a, const vint& b) { return vbool{ _mm_cmpgt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, const vint& b) { return vint{ _mm_add_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, const vint& b) { return vint{ _mm_sub_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, int b) { return a + vint(b); }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, int b) { return a - vint(b); }
CPPSPMD_FORCE_INLINE vint operator+(int a, const vint& b) { return vint(a) + b; }
CPPSPMD_FORCE_INLINE vint operator-(int a, const vint& b) { return vint(a) - b; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, const vint& b) { return vint{ _mm_mullo_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, int b) { return a * vint(b); }
CPPSPMD_FORCE_INLINE vint operator*(int a, const vint& b) { return vint(a) * b; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ _mm_sub_epi32(_mm_setzero_si128(), v.m_value) }; }

CPPSPMD_FORCE_INLINE int safe_div(int a, int b) { return b ? (a / b) : 0; }
CPPSPMD_FORCE_INLINE int safe_mod(int a, int b) { return b ? (a % b) : 0; }

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[4];
	result[0] = safe_div(_mm_extract_epi32(a.m_value, 0), _mm_extract_epi32(b.m_value, 0));
	result[1] = safe_div(_mm_extract_epi32(a.m_value, 1), _mm_extract_epi32(b.m_value, 1));
	result[2] = safe_div(_mm_extract_epi32(a.m_value, 2), _mm_extract_epi32(b.m_value, 2));
	result[3] = safe_div(_mm_extract_epi32(a.m_value, 3), _mm_extract_epi32(b.m_value, 3));

	return vint{ _mm_load_si128((__m128i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, int b)
{
	CPPSPMD_ALIGN(32) int result[4];
	if (!b)
		return vint{ _mm_setzero_si128() };

	result[0] = _mm_extract_epi32(a.m_value, 0) / b;
	result[1] = _mm_extract_epi32(a.m_value, 1) / b;
	result[2] = _mm_extract_epi32(a.m_value, 2) / b;
	result[3] = _mm_extract_epi32(a.m_value, 3) / b;

	return vint{ _mm_load_si128((__m128i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[4];
	result[0] = safe_mod(_mm_extract_epi32(a.m_value, 0), _mm_extract_epi32(b.m_value, 0));
	result[1] = safe_mod(_mm_extract_epi32(a.m_value, 1), _mm_extract_epi32(b.m_value, 1));
	result[2] = safe_mod(_mm_extract_epi32(a.m_value, 2), _mm_extract_epi32(b.m_value, 2));
	result[3] = safe_mod(_mm_extract_epi32(a.m_value, 3), _mm_extract_epi32(b.m_value, 3));

	return vint{ _mm_load_si128((__m128i*)result) };
}

// This is very slow, it's here for completeness. Don't use it.
CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b)
{
	CPPSPMD_ALIGN(32) int result[4];
	if (!b)
		return vint{ _mm_setzero_si128() };

	result[0] = safe_mod(_mm_extract_epi32(a.m_value, 0), b);
	result[1] = safe_mod(_mm_extract_epi32(a.m_value, 1), b);
	result[2] = safe_mod(_mm_extract_epi32(a.m_value, 2), b);
	result[3] = safe_mod(_mm_extract_epi32(a.m_value, 3), b);

	return vint{ _mm_load_si128((__m128i*)result) };
}

CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[4];
	result[0] = _mm_extract_epi32(a.m_value, 0) << _mm_extract_epi32(b.m_value, 0);
	result[1] = _mm_extract_epi32(a.m_value, 1) << _mm_extract_epi32(b.m_value, 1);
	result[2] = _mm_extract_epi32(a.m_value, 2) << _mm_extract_epi32(b.m_value, 2);
	result[3] = _mm_extract_epi32(a.m_value, 3) << _mm_extract_epi32(b.m_value, 3);

	return vint{ _mm_load_si128((__m128i*)result) };
}

// uniform shift left
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_sll_epi32(a.m_value, bv) };
}

// uniform arithmetic shift right
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_sra_epi32(a.m_value, bv) };
}

// uniform shift right
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i *)g_x_128))));
	return vint{ _mm_srl_epi32(a.m_value, bv) };
}

CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[4];
	result[0] = _mm_extract_epi32(a.m_value, 0) >> _mm_extract_epi32(b.m_value, 0);
	result[1] = _mm_extract_epi32(a.m_value, 1) >> _mm_extract_epi32(b.m_value, 1);
	result[2] = _mm_extract_epi32(a.m_value, 2) >> _mm_extract_epi32(b.m_value, 2);
	result[3] = _mm_extract_epi32(a.m_value, 3) >> _mm_extract_epi32(b.m_value, 3);

	return vint{ _mm_load_si128((__m128i*)result) };
}

CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b)
{
	CPPSPMD_ALIGN(32) int result[4];
	result[0] = ((uint32_t)_mm_extract_epi32(a.m_value, 0)) >> _mm_extract_epi32(b.m_value, 0);
	result[1] = ((uint32_t)_mm_extract_epi32(a.m_value, 1)) >> _mm_extract_epi32(b.m_value, 1);
	result[2] = ((uint32_t)_mm_extract_epi32(a.m_value, 2)) >> _mm_extract_epi32(b.m_value, 2);
	result[3] = ((uint32_t)_mm_extract_epi32(a.m_value, 3)) >> _mm_extract_epi32(b.m_value, 3);

	return vint{ _mm_load_si128((__m128i*)result) };
}

CPPSPMD_FORCE_INLINE vint create_vint(__m128i v) { return vint{ v }; }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) CPPSPMD::create_vint( _mm_slli_epi32( (a).m_value, (b) ) )
#define VINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( _mm_srai_epi32( (a).m_value, (b) ) ) 
#define VUINT_SHIFT_RIGHT(a, b) CPPSPMD::create_vint( _mm_srli_epi32( (a).m_value, (b) ) )

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, const lint& b) { return vbool{ _mm_cmpeq_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, int b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint& b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint& a, const lint& b) { return vbool{ _mm_cmpgt_epi32(b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>(const lint& a, const lint& b) { return vbool{ _mm_cmpgt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator<=(const lint& a, const lint& b) { return !vbool{ _mm_cmpgt_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vbool operator>=(const lint& a, const lint& b) { return !vbool{ _mm_cmpgt_epi32(b.m_value, a.m_value) }; }

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) { assert(instance < 4); CPPSPMD_ALIGN(16) float values[4]; _mm_store_ps(values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const vint& v, int instance) { assert(instance < 4); CPPSPMD_ALIGN(16) int values[4]; _mm_store_si128((__m128i*)values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE int extract(const lint& v, int instance) { assert(instance < 4); CPPSPMD_ALIGN(16) int values[4]; _mm_store_si128((__m128i*)values, v.m_value); return values[instance]; }
CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) { assert(instance < 4); CPPSPMD_ALIGN(16) int values[4]; _mm_store_si128((__m128i*)values, v.m_value); return values[instance] != 0; }

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

#define VINT_EXTRACT(v, instance) _mm_extract_epi32((v).m_value, instance)
#define VBOOL_EXTRACT(v, instance) _mm_extract_epi32((v).m_value, instance)
#define VFLOAT_EXTRACT(result, v, instance) _mm_extract_ps((v).m_value, instance)

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
	CPPSPMD_ALIGN(16) int vindex[4];
	_mm_store_si128((__m128i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(16) float stored[4];
	_mm_store_ps(stored, src.m_value);

	int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
	for (int i = 0; i < 4; i++)
	{
		if (mask & (1 << i))
			dst.m_pValue[vindex[i]] = stored[i];
	}
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(16) int vindex[4];
	_mm_store_si128((__m128i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(16) float stored[4];
	_mm_store_ps(stored, src.m_value);

	for (int i = 0; i < 4; i++)
		dst.m_pValue[vindex[i]] = stored[i];
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(16) int vindex[4];
	_mm_store_si128((__m128i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(16) float stored[4];
	_mm_store_ps(stored, src.m_value);

	int mask = _mm_movemask_ps(_mm_castsi128_ps(m_exec.m_mask));
	for (int i = 0; i < 4; i++)
	{
		if (mask & (1 << i))
			dst.m_pValue[vindex[i]] = stored[i];
	}
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	CPPSPMD_ALIGN(16) int vindex[4];
	_mm_store_si128((__m128i*)vindex, dst.m_vindex);

	CPPSPMD_ALIGN(16) float stored[4];
	_mm_store_ps(stored, src.m_value);

	for (int i = 0; i < 4; i++)
		dst.m_pValue[vindex[i]] = stored[i];
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
			exec_mask partial_mask = exec_mask{ _mm_cmpgt_epi32(_mm_set1_epi32(total_partial), program_index.m_value) };
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

} // namespace cppspmd_sse41

