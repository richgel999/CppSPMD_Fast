// cppspmd_avx512.h
// AVX-512 support.
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
#define CPPSPMD_DECL(type, name) __declspec(align(64)) type name
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
#undef CPPSPMD_AVX512

#define CPPSPMD_SSE 0
#define CPPSPMD_AVX 1
#define CPPSPMD_FLOAT4 0
#define CPPSPMD_INT16 0
#define CPPSPMD_AVX1 0
#define CPPSPMD_AVX2 0

#define CPPSPMD cppspmd_avx512
#define CPPSPMD_ARCH _avx512
#define CPPSPMD_AVX512 1 

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
#define CPPSPMD_ALIGNMENT (64)

namespace CPPSPMD
{
const int PROGRAM_COUNT_SHIFT = 4;
const int PROGRAM_COUNT = 1 << PROGRAM_COUNT_SHIFT;

template <typename N> inline N* aligned_new() { void* p = _mm_malloc(sizeof(N), 64); new (p) N;	return static_cast<N*>(p); }
template <typename N> void aligned_delete(N* p) { if (p) { p->~N(); _mm_free(p); } }

CPPSPMD_DECL(const uint32_t, g_allones_512[16]) = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,   UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
CPPSPMD_DECL(const uint32_t, g_bit7_512[16]) = { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,   0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
CPPSPMD_DECL(const float, g_onef_512[16]) = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,   1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
CPPSPMD_DECL(const uint32_t, g_oneu_512[16]) = { 1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1 };
CPPSPMD_DECL(const uint32_t, g_x_128[4]) = { UINT32_MAX, 0, 0, 0 };

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

const uint32_t ALL_ON_MOVEMASK = 0xFFFF;

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
		__mmask16 m_mask;

		exec_mask() = default;

		CPPSPMD_FORCE_INLINE explicit exec_mask(const vbool& b);

		CPPSPMD_FORCE_INLINE explicit exec_mask(const __m512i& mask) : m_mask( _mm512_cmpge_epu32_mask(mask, _mm512_loadu_epi32(g_bit7_512)) ) { }

		CPPSPMD_FORCE_INLINE void enable_lane(uint32_t lane) { m_mask = (__mmask16)(1U << lane); }

		static CPPSPMD_FORCE_INLINE exec_mask all_on() { return exec_mask{ _mm512_int2mask(ALL_ON_MOVEMASK) }; }
		static CPPSPMD_FORCE_INLINE exec_mask all_off() { return exec_mask{ _mm512_int2mask(0) }; }

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const 
		{ 
			uint32_t mask = _mm512_mask2int(m_mask); 
			return mask;
		}
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
		__mmask16 m_value;

		vbool() = default;

		CPPSPMD_FORCE_INLINE vbool(bool value) : m_value(_mm512_int2mask(value ? ALL_ON_MOVEMASK : 0)) { }

		CPPSPMD_FORCE_INLINE vbool(__mmask16 value) : m_value(value) { }

		CPPSPMD_FORCE_INLINE explicit vbool(const __m512i& value) : m_value(_mm512_cmpge_epu32_mask(value, _mm512_loadu_epi32(g_bit7_512))) { }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const;
		CPPSPMD_FORCE_INLINE explicit operator vint() const;

		CPPSPMD_FORCE_INLINE uint32_t get_movemask() const 
		{ 
			uint32_t mask = _mm512_mask2int(m_value); 
			return mask;
		}

	private:
		vbool& operator=(const vbool&);
	};

	friend vbool operator!(const vbool& v);

	CPPSPMD_FORCE_INLINE vbool& store(vbool& dst, const vbool& src)
	{
		dst.m_value = _mm512_kor( _mm512_kand(m_exec.m_mask, src.m_value), _mm512_kandn(m_exec.m_mask, dst.m_value) );
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
		__m512 m_value;

		vfloat() = default;

		CPPSPMD_FORCE_INLINE explicit vfloat(const __m512& v) : m_value(v) { }

		CPPSPMD_FORCE_INLINE vfloat(float value) : m_value(_mm512_set1_ps(value)) { }

		CPPSPMD_FORCE_INLINE explicit vfloat(int value) : m_value(_mm512_set1_ps((float)value)) { }

	private:
		vfloat& operator=(const vfloat&);
	};

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat& dst, const vfloat& src)
	{
		dst.m_value = _mm512_mask_mov_ps(dst.m_value, m_exec.m_mask, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat& store(vfloat&& dst, const vfloat& src)
	{
		dst.m_value = _mm512_mask_mov_ps(dst.m_value, m_exec.m_mask, src.m_value);
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
		_mm512_mask_storeu_ps(dst.m_pValue, m_exec.m_mask, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store(const float_lref&& dst, const vfloat& src)
	{
		_mm512_mask_storeu_ps(dst.m_pValue, m_exec.m_mask, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref& dst, const vfloat& src)
	{
		_mm512_storeu_ps(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const float_lref& store_all(const float_lref&& dst, const vfloat& src)
	{
		_mm512_storeu_ps(dst.m_pValue, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const float_lref& src)
	{
		return vfloat{ _mm512_mask_loadu_ps(_mm512_setzero_ps(), m_exec.m_mask, src.m_pValue) };
	}

	// Varying ref to floats
	struct float_vref
	{
		__m512i m_vindex;
		float* m_pValue;

	private:
		float_vref& operator=(const float_vref&);
	};

	// Varying ref to varying float
	struct vfloat_vref
	{
		__m512i m_vindex;
		vfloat* m_pValue;

	private:
		vfloat_vref& operator=(const vfloat_vref&);
	};

	// Varying ref to varying int
	struct vint_vref
	{
		__m512i m_vindex;
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
		return vfloat{ _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m_exec.m_mask, src.m_vindex, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all(const float_vref& src)
	{
		return vfloat{ _mm512_i32gather_ps(src.m_vindex, src.m_pValue, 4) };
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
		_mm512_mask_storeu_epi32(dst.m_pValue, m_exec.m_mask, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_lref& src)
	{
		return vint{ _mm512_mask_loadu_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_pValue) };
	}

	// Linear ref to int16's
	struct int16_lref
	{
		int16_t* m_pValue;

	private:
		int16_lref& operator=(const int16_lref&);
	};

	// TODO - Optimize
	CPPSPMD_FORCE_INLINE int16_lref& store(int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(64) int stored[16];
		_mm512_store_si512(stored, src.m_value);

		int mask = m_exec.get_movemask();
		for (int i = 0; i < 16; i++)
		{
			if (mask & (1 << i))
				dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		}
		return dst;
	}
		
	// TODO - Optimize
	CPPSPMD_FORCE_INLINE const int16_lref& store_all(const int16_lref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(64) int stored[16];
		_mm512_store_si512(stored, src.m_value);

		for (int i = 0; i < 16; i++)
			dst.m_pValue[i] = static_cast<int16_t>(stored[i]);
		return dst;
	}

	// TODO - Optimize
	CPPSPMD_FORCE_INLINE vint load(const int16_lref& src)
	{
		CPPSPMD_ALIGN(64) int values[16];

		for (int i = 0; i < 16; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		return vint{ _mm512_mask_loadu_epi32(_mm512_setzero_si512(), m_exec.m_mask, values) };
	}

	// TODO - Optimize
	CPPSPMD_FORCE_INLINE vint load_all(const int16_lref& src)
	{
		CPPSPMD_ALIGN(64) int values[16];

		for (int i = 0; i < 16; i++)
			values[i] = static_cast<int16_t>(src.m_pValue[i]);

		return vint{ _mm512_loadu_epi32(values) };
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
		return vint{ _mm512_mask_loadu_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_pValue) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_lref& src)
	{
		return vint{ _mm512_loadu_epi32(src.m_pValue) };
	}

	// Varying ref to ints
	struct int_vref
	{
		__m512i m_vindex;
		int* m_pValue;

	private:
		int_vref& operator=(const int_vref&);
	};

	// Varying ref to constant ints
	struct cint_vref
	{
		__m512i m_vindex;
		const int* m_pValue;

	private:
		cint_vref& operator=(const cint_vref&);
	};

	// Varying int
	struct vint
	{
		__m512i m_value;

		vint() = default;

		CPPSPMD_FORCE_INLINE explicit vint(const __m512i& value) : m_value(value) { }

		CPPSPMD_FORCE_INLINE vint(int value) : m_value(_mm512_set1_epi32(value)) { }

		CPPSPMD_FORCE_INLINE explicit vint(float value) : m_value(_mm512_set1_epi32((int)value)) { }

		CPPSPMD_FORCE_INLINE explicit vint(const vfloat& other) : m_value(_mm512_cvttps_epi32(other.m_value)) { }

		CPPSPMD_FORCE_INLINE explicit operator vbool() const
		{
			__mmask16 v = _mm512_cmpneq_epi32_mask(m_value, _mm512_setzero_si512());
			return vbool{ v };
		}

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm512_cvtepi32_ps(m_value) };
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
	CPPSPMD_FORCE_INLINE void storeu_linear(int* pDst, const vint& src)
	{
		_mm512_mask_storeu_epi32(pDst, m_exec.m_mask, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(int* pDst, const vint& src)
	{
		_mm512_storeu_epi32(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(int* pDst, const vint& src)
	{
		_mm512_store_epi32(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear(const int* pSrc)
	{
		return vint{ _mm512_mask_loadu_epi32(_mm512_setzero_si512(), m_exec.m_mask, pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint loadu_linear_all(const int* pSrc)
	{
		return vint{ _mm512_loadu_epi32(pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint load_linear_all(const int* pSrc)
	{
		return vint{ _mm512_load_epi32(pSrc) };
	}

	// load/store linear float
	CPPSPMD_FORCE_INLINE void storeu_linear(float* pDst, const vfloat& src)
	{
		_mm512_mask_storeu_ps(pDst, m_exec.m_mask, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void storeu_linear_all(float* pDst, const vfloat& src)
	{
		_mm512_storeu_ps(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE void store_linear_all(float* pDst, const vfloat& src)
	{
		_mm512_store_ps(pDst, src.m_value);
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear(const float* pSrc)
	{
		return vfloat{ _mm512_mask_loadu_ps(_mm512_setzero_ps(), m_exec.m_mask, pSrc) };
	}

	CPPSPMD_FORCE_INLINE vfloat loadu_linear_all(const float* pSrc)
	{
		return vfloat{ _mm512_loadu_ps(pSrc) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_linear_all(const float* pSrc)
	{
		return vfloat{ _mm512_load_ps(pSrc) };
	}

	CPPSPMD_FORCE_INLINE vint& store(vint& dst, const vint& src)
	{
		dst.m_value = _mm512_mask_blend_epi32(m_exec.m_mask, dst.m_value, src.m_value);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store(const int_vref& dst, const vint& src)
	{
		_mm512_mask_i32scatter_epi32(dst.m_pValue, m_exec.m_mask, dst.m_vindex, src.m_value, 4);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store_bytes(const int_vref& dst, const vint& src)
	{
		_mm512_mask_i32scatter_epi32(dst.m_pValue, m_exec.m_mask, dst.m_vindex, src.m_value, 1);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store_bytes_precise(const int_vref& dst, const vint& src)
	{
		CPPSPMD_ALIGN(64) int vindex[16];
		_mm512_store_si512((__m512i*)vindex, dst.m_vindex);

		CPPSPMD_ALIGN(64) int stored[16];
		_mm512_store_si512((__m512i*)stored, src.m_value);

		uint32_t mask = get_movemask();
		for (int i = 0; i < 16; i++)
		{
			if (mask & (1 << i))
				*((uint8_t *)dst.m_pValue + vindex[i]) = (uint8_t)stored[i];
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
		_mm512_i32scatter_epi32(dst.m_pValue, dst.m_vindex, src.m_value, 4);
		return dst;
	}

	CPPSPMD_FORCE_INLINE const int_vref& store_bytes_all(const int_vref& dst, const vint& src)
	{
		_mm512_i32scatter_epi32(dst.m_pValue, dst.m_vindex, src.m_value, 1);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const int_vref& src)
	{
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_vindex, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vint load_bytes(const int_vref& src)
	{
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_vindex, src.m_pValue, 1) };
	}

	CPPSPMD_FORCE_INLINE vint load_words(const int_vref& src)
	{
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_vindex, src.m_pValue, 2) };
	}

	CPPSPMD_FORCE_INLINE vint load_bytes_all(const int_vref& src)
	{
		return vint{ _mm512_i32gather_epi32(src.m_vindex, src.m_pValue, 1) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const int_vref& src)
	{
		return vint{ _mm512_i32gather_epi32(src.m_vindex, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vint load(const cint_vref& src)
	{
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_vindex, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vint load_bytes(const cint_vref& src)
	{
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, src.m_vindex, src.m_pValue, 1) };
	}

	CPPSPMD_FORCE_INLINE vint load_all(const cint_vref& src)
	{
		return vint{ _mm512_i32gather_epi32(src.m_vindex, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vint load_bytes_all(const cint_vref& src)
	{
		return vint{ _mm512_i32gather_epi32(src.m_vindex, src.m_pValue, 1) };
	}

	CPPSPMD_FORCE_INLINE vint load_words_all(const cint_vref& src)
	{
		return vint{ _mm512_i32gather_epi32(src.m_vindex, src.m_pValue, 2) };
	}

	CPPSPMD_FORCE_INLINE void store_strided(int* pDst, uint32_t stride, const vint& v)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		_mm512_mask_i32scatter_epi32(pDst, m_exec.m_mask, vstride, v.m_value, 4);
	}

	CPPSPMD_FORCE_INLINE void store_strided(float* pDstF, uint32_t stride, const vfloat& v)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		_mm512_mask_i32scatter_ps(pDstF, m_exec.m_mask, vstride, v.m_value, 4);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(int* pDst, uint32_t stride, const vint& v)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		_mm512_i32scatter_epi32(pDst, vstride, v.m_value, 4);
	}

	CPPSPMD_FORCE_INLINE void store_all_strided(float* pDstF, uint32_t stride, const vfloat& v)
	{
		__m512i m = _mm512_set1_epi32(stride);
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, m);
		_mm512_i32scatter_ps(pDstF, vstride, v.m_value, 4);
	}

	CPPSPMD_FORCE_INLINE vint load_strided(const int* pSrc, uint32_t stride)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, vstride, pSrc, 4) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_strided(const float* pSrc, uint32_t stride)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		return vfloat{ _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m_exec.m_mask, vstride, pSrc, 4) };
	}

	CPPSPMD_FORCE_INLINE vint load_all_strided(const int* pSrc, uint32_t stride)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		return vint{ _mm512_i32gather_epi32(vstride, pSrc, 4) };
	}

	CPPSPMD_FORCE_INLINE vfloat load_all_strided(const float* pSrc, uint32_t stride)
	{
		__m512i vstride = _mm512_mullo_epi32(program_index.m_value, _mm512_set1_epi32(stride));
		return vfloat{ _mm512_i32gather_ps(vstride, pSrc, 4) };
	}

	CPPSPMD_FORCE_INLINE const vfloat_vref& store(const vfloat_vref& dst, const vfloat& src)
	{
		__m512i dstv = _mm512_add_epi32(_mm512_slli_epi32(dst.m_vindex, 4), program_index.m_value);
		_mm512_mask_i32scatter_ps(dst.m_pValue, m_exec.m_mask, dstv, src.m_value, 4);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vfloat load(const vfloat_vref& src)
	{
		__m512i srcv = _mm512_add_epi32(_mm512_slli_epi32(src.m_vindex, 4), program_index.m_value);
		return vfloat{ _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m_exec.m_mask, srcv, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE const vint_vref& store(const vint_vref& dst, const vint& src)
	{
		__m512i dstv = _mm512_add_epi32(_mm512_slli_epi32(dst.m_vindex, 4), program_index.m_value);
		_mm512_mask_i32scatter_epi32(dst.m_pValue, m_exec.m_mask, dstv, src.m_value, 4);
		return dst;
	}

	CPPSPMD_FORCE_INLINE vint load(const vint_vref& src)
	{
		__m512i srcv = _mm512_add_epi32(_mm512_slli_epi32(src.m_vindex, 4), program_index.m_value);
		return vint{ _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), m_exec.m_mask, srcv, src.m_pValue, 4) };
	}

	CPPSPMD_FORCE_INLINE vint& add(vint &dst, vint a, vint b) { dst.m_value = _mm512_mask_add_epi32(dst.m_value, m_exec.m_mask, a.m_value, b.m_value); return dst; }
	CPPSPMD_FORCE_INLINE vint& sub(vint& dst, vint a, vint b) { dst.m_value = _mm512_mask_sub_epi32(dst.m_value, m_exec.m_mask, a.m_value, b.m_value); return dst;	}
	CPPSPMD_FORCE_INLINE vint& mul(vint& dst, vint a, vint b) { dst.m_value = _mm512_mask_mul_epi32(dst.m_value, m_exec.m_mask, a.m_value, b.m_value); return dst; }

	// Linear integer
	struct lint
	{
		__m512i m_value;

		CPPSPMD_FORCE_INLINE explicit lint(__m512i value)
			: m_value(value)
		{ }

		CPPSPMD_FORCE_INLINE explicit operator vfloat() const
		{
			return vfloat{ _mm512_cvtepi32_ps(m_value) };
		}

		CPPSPMD_FORCE_INLINE explicit operator vint() const
		{
			return vint{ m_value };
		}

		int get_first_value() const
		{
			return _mm_cvtsi128_si32(_mm512_castsi512_si128(m_value));
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

	const lint program_index = lint{ _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) };

	// SPMD condition helpers

	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_if(const vbool& cond, const IfBody& ifBody);

	CPPSPMD_FORCE_INLINE void spmd_if_break(const vbool& cond);
	CPPSPMD_FORCE_INLINE void spmd_simple_if_break(const vbool& cond);

	// No breaks, continues, etc. allowed
	template<typename IfBody>
	CPPSPMD_FORCE_INLINE void spmd_sif(const vbool& cond, const IfBody& ifBody);
		
	// No breaks, continues, etc. allowed
	template<typename IfBody, typename ElseBody>
	CPPSPMD_FORCE_INLINE void spmd_sifelse(const vbool& cond, const IfBody& ifBody, const ElseBody& elseBody);

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

	CPPSPMD_FORCE_INLINE float reduce_add(vfloat v)	{ return _mm512_mask_reduce_add_ps(m_exec.m_mask, v.m_value); }

	CPPSPMD_FORCE_INLINE void swap(vint& a, vint& b) { vint temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vfloat& a, vfloat& b) { vfloat temp = a; store(a, b); store(b, temp); }
	CPPSPMD_FORCE_INLINE void swap(vbool& a, vbool& b) { vbool temp = a; store(a, b); store(b, temp); }
		
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
	return vfloat{ _mm512_mask_blend_ps(m_value, _mm512_setzero_ps(), *(const __m512*)g_onef_512) };
}

// Returns UINT32_MAX's for true, 0 for false. (Should it return 1's?)
CPPSPMD_FORCE_INLINE spmd_kernel::vbool::operator vint() const
{
	return vint{ _mm512_mask_blend_epi32(m_value, _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) };
}

CPPSPMD_FORCE_INLINE vbool operator!(const vbool& v)
{
	__mmask16 b = v.m_value;
	b = _knot_mask16(b);
	return vbool{ b };
}

CPPSPMD_FORCE_INLINE exec_mask::exec_mask(const vbool& b) { m_mask = b.m_value; }

CPPSPMD_FORCE_INLINE exec_mask operator^(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm512_kxor(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator&(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm512_kand(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE exec_mask operator|(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm512_kor(a.m_mask, b.m_mask) }; }

CPPSPMD_FORCE_INLINE bool all(const exec_mask& e) { return e.get_movemask() == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const exec_mask& e) { return e.get_movemask() != 0; }

// Bad pattern - doesn't factor in the current exec mask. Prefer spmd_any() instead.
CPPSPMD_FORCE_INLINE bool all(const vbool& e) { return e.get_movemask() == ALL_ON_MOVEMASK; }
CPPSPMD_FORCE_INLINE bool any(const vbool& e) { return e.get_movemask() != 0; }

CPPSPMD_FORCE_INLINE exec_mask andnot(const exec_mask& a, const exec_mask& b) { return exec_mask{ _mm512_kandn(a.m_mask, b.m_mask) }; }
CPPSPMD_FORCE_INLINE vbool operator||(const vbool& a, const vbool& b) 
{ 
	__mmask16 v = _mm512_kor(a.m_value, b.m_value);
	return vbool{ v };
}
	
CPPSPMD_FORCE_INLINE vbool operator&&(const vbool& a, const vbool& b) 
{ 
	__mmask16 v = _mm512_kand(a.m_value, b.m_value);
	return vbool{ v };
}

CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_add_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_sub_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const vfloat& b) { return vfloat(a) + b; }
CPPSPMD_FORCE_INLINE vfloat operator+(const vfloat& a, float b) { return a + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, const vint& b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(const vint& a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, int b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(int a, const vfloat& b) { return vfloat(a) - b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& a, float b) { return a - vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator-(float a, const vfloat& b) { return vfloat(a) - b; }

CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_mul_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, float b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float a, const vfloat& b) { return vfloat(a) * b; }
CPPSPMD_FORCE_INLINE vfloat operator*(const vfloat& a, int b) { return a * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(int a, const vfloat& b) { return vfloat(a) * b; }

CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_div_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, int b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(int a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator/(const vfloat& a, float b) { return a / vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator/(float a, const vfloat& b) { return vfloat(a) / b; }
CPPSPMD_FORCE_INLINE vfloat operator-(const vfloat& v) { return vfloat{ _mm512_sub_ps(_mm512_xor_ps(v.m_value, v.m_value), v.m_value) }; }

CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, const vfloat& b) 
{ 
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_EQ_OQ);
	return vbool{ v }; 
}
CPPSPMD_FORCE_INLINE vbool operator==(const vfloat& a, float b) { return a == vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, const vfloat& b) 
{ 
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_EQ_OQ);
	return !vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator!=(const vfloat& a, float b) { return a != vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, const vfloat& b) 
{ 
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_LT_OQ);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator<(const vfloat& a, float b) { return a < vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, const vfloat& b) 
{ 
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_GT_OQ);
	return vbool{ v }; 
}
CPPSPMD_FORCE_INLINE vbool operator>(const vfloat& a, float b) { return a > vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, const vfloat& b) 
{ 
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_LE_OQ);
	return vbool{ v }; 
}
CPPSPMD_FORCE_INLINE vbool operator<=(const vfloat& a, float b) { return a <= vfloat(b); }

CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, const vfloat& b) 
{
	__mmask16 v = _mm512_cmp_ps_mask(a.m_value, b.m_value, _CMP_GE_OQ);
	return vbool{ v }; 
}
CPPSPMD_FORCE_INLINE vbool operator>=(const vfloat& a, float b) { return a >= vfloat(b); }

CPPSPMD_FORCE_INLINE vfloat spmd_ternaryf(const vbool& cond, const vfloat& a, const vfloat& b) { return vfloat{ _mm512_mask_blend_ps(cond.m_value, b.m_value, a.m_value) }; }
CPPSPMD_FORCE_INLINE vint spmd_ternaryi(const vbool& cond, const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi32(cond.m_value, b.m_value, a.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat sqrt(const vfloat& v) { return vfloat{ _mm512_sqrt_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat abs(const vfloat& v) { return vfloat{ _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat max(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_max_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat min(const vfloat& a, const vfloat& b) { return vfloat{ _mm512_min_ps(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat ceil(const vfloat& a) { return vfloat{ _mm512_ceil_ps(a.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat floor(const vfloat& v) { return vfloat{ _mm512_floor_ps(v.m_value) }; }
CPPSPMD_FORCE_INLINE vfloat round_nearest(const vfloat& a) { return vfloat{ _mm512_roundscale_ps(a.m_value, _MM_FROUND_TO_NEAREST_INT) }; }
CPPSPMD_FORCE_INLINE vfloat round_truncate(const vfloat& a) { return vfloat{ _mm512_roundscale_ps(a.m_value, _MM_FROUND_TO_ZERO) }; }
CPPSPMD_FORCE_INLINE vfloat frac(const vfloat& a) { return a - floor(a); }
CPPSPMD_FORCE_INLINE vfloat fmod(const vfloat &a, const vfloat &b) { vfloat c = frac(abs(a / b)) * abs(b); return spmd_ternaryf(a < 0, -c, c); }
CPPSPMD_FORCE_INLINE vfloat sign(const vfloat& a) { return spmd_ternaryf(a < 0.0f, 1.0f, 1.0f); }

CPPSPMD_FORCE_INLINE vint max(const vint& a, const vint& b)
{
	return vint{ _mm512_max_epi32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint min(const vint& a, const vint& b)
{
	return vint{ _mm512_min_epi32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint maxu(const vint& a, const vint& b)
{
	return vint{ _mm512_max_epu32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint minu(const vint& a, const vint& b)
{
	return vint{ _mm512_min_epu32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint abs(const vint& v) { return vint{ _mm512_abs_epi32(v.m_value) }; }

CPPSPMD_FORCE_INLINE vint byteswap(const vint& v)
{
	CPPSPMD_DECL(const uint8_t, s_smask[64]) = { 
		3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,  3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,
		3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12,  3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12
	};
	return vint{ _mm512_shuffle_epi8(v.m_value, _mm512_loadu_si512(s_smask)) };
}

CPPSPMD_FORCE_INLINE vint cast_vfloat_to_vint(const vfloat& v) { return vint{ _mm512_castps_si512(v.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat cast_vint_to_vfloat(const vint& v) { return vfloat{ _mm512_castsi512_ps(v.m_value) }; }

CPPSPMD_FORCE_INLINE vfloat clamp(const vfloat& v, const vfloat& a, const vfloat& b)
{
	return vfloat{ _mm512_min_ps(b.m_value, _mm512_max_ps(v.m_value, a.m_value)) };
}

CPPSPMD_FORCE_INLINE vint clamp(const vint& v, const vint& a, const vint& b)
{
	return vint{ _mm512_max_epi32(a.m_value, _mm512_min_epi32(v.m_value, b.m_value)) };
}

CPPSPMD_FORCE_INLINE vfloat vfma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm512_fmadd_ps(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm512_fmsub_ps(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnma(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm512_fnmadd_ps(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat vfnms(const vfloat& a, const vfloat& b, const vfloat& c)
{
	return vfloat{ _mm512_fnmsub_ps(a.m_value, b.m_value, c.m_value) };
}

CPPSPMD_FORCE_INLINE vfloat lerp(const vfloat& x, const vfloat& y, const vfloat& s) { return vfma(y - x, s, x); }

CPPSPMD_FORCE_INLINE lint operator+(int a, const lint& b) { return lint{ _mm512_add_epi32(_mm512_set1_epi32(a), b.m_value) }; }
CPPSPMD_FORCE_INLINE lint operator+(const lint& a, int b) { return lint{ _mm512_add_epi32(a.m_value, _mm512_set1_epi32(b)) }; }
CPPSPMD_FORCE_INLINE vfloat operator+(float a, const lint& b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator+(const lint& a, float b) { return vfloat(a) + vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(const lint& a, float b) { return vfloat(a) * vfloat(b); }
CPPSPMD_FORCE_INLINE vfloat operator*(float b, const lint& a) { return vfloat(a) * vfloat(b); }

CPPSPMD_FORCE_INLINE vint operator&(const vint& a, const vint& b) { return vint{ _mm512_and_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator&(const vint& a, int b) { return a & vint(b); }
CPPSPMD_FORCE_INLINE vint andnot(const vint& a, const vint& b) { return vint{ _mm512_andnot_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, const vint& b) { return vint{ _mm512_or_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator|(const vint& a, int b) { return a | vint(b); }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, const vint& b) { return vint{ _mm512_xor_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator^(const vint& a, int b) { return a ^ vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpeq_epi32_mask(a.m_value, b.m_value);
	return vbool{ v }; 
}
CPPSPMD_FORCE_INLINE vbool operator!=(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpneq_epi32_mask(a.m_value, b.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator<(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(b.m_value, a.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator<=(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(a.m_value, b.m_value);
	return !vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator>=(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(b.m_value, a.m_value);
	return !vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator>(const vint& a, const vint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(a.m_value, b.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vint operator+(const vint& a, const vint& b) { return vint{ _mm512_add_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, const vint& b) { return vint{ _mm512_sub_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator+(const vint& a, int b) { return a + vint(b); }
CPPSPMD_FORCE_INLINE vint operator-(const vint& a, int b) { return a - vint(b); }
CPPSPMD_FORCE_INLINE vint operator+(int a, const vint& b) { return vint(a) + b; }
CPPSPMD_FORCE_INLINE vint operator-(int a, const vint& b) { return vint(a) - b; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, const vint& b) { return vint{ _mm512_mullo_epi32(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint operator*(const vint& a, int b) { return a * vint(b); }
CPPSPMD_FORCE_INLINE vint operator*(int a, const vint& b) { return vint(a) * b; }

CPPSPMD_FORCE_INLINE __m512i mulhi_epu32(__m512i a, __m512i b)
{
	__m512i tmp1 = _mm512_mul_epu32(a, b);
	__m512i tmp2 = _mm512_mul_epu32(_mm512_bsrli_epi128(a, 4), _mm512_bsrli_epi128(b, 4));
	return _mm512_unpacklo_epi32(_mm512_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 3, 1)), _mm512_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 3, 1)));
}

CPPSPMD_FORCE_INLINE vint mulhiu(const vint& a, const vint& b) { return vint{ mulhi_epu32(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint operator-(const vint& v) { return vint{ _mm512_sub_epi32(_mm512_setzero_si512(), v.m_value) }; }

CPPSPMD_FORCE_INLINE vint operator~(const vint& a) { return vint{ -a - 1 }; }

// A few of these break the lane-based abstraction model. They are supported in SSE2, so it makes sense to support them and let the user figure it out.
CPPSPMD_FORCE_INLINE vint adds_epu8(const vint& a, const vint& b) { return vint{ _mm512_adds_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epu8(const vint& a, const vint& b) { return vint{ _mm512_subs_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint avg_epu8(const vint& a, const vint& b) { return vint{ _mm512_avg_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint max_epu8(const vint& a, const vint& b) { return vint{ _mm512_max_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min_epu8(const vint& a, const vint& b) { return vint{ _mm512_min_epu8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sad_epu8(const vint& a, const vint& b) { return vint{ _mm512_sad_epu8(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint add_epi8(const vint& a, const vint& b) { return vint{ _mm512_add_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epi8(const vint& a, const vint& b) { return vint{ _mm512_adds_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sub_epi8(const vint& a, const vint& b) { return vint{ _mm512_sub_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epi8(const vint& a, const vint& b) { return vint{ _mm512_subs_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi8(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(a.m_value, b.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi8(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(a.m_value, b.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi8(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi8(_mm512_cmplt_epi8_mask(a.m_value, b.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) }; }
CPPSPMD_FORCE_INLINE vint unpacklo_epi8(const vint& a, const vint& b) { return vint{ _mm512_unpacklo_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint unpackhi_epi8(const vint& a, const vint& b) { return vint{ _mm512_unpackhi_epi8(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE uint64_t movemask_epi8(const vint& a) { return _cvtmask64_u64(_mm512_cmpge_epu8_mask(a.m_value, _mm512_set1_epi8(0x80))); }
CPPSPMD_FORCE_INLINE int movemask_epi32(const vint& a) { return _mm512_mask2int(_mm512_cmpge_epu32_mask(a.m_value, _mm512_set1_epi32(0x80000000))); }

CPPSPMD_FORCE_INLINE vint cmple_epu8(const vint& a, const vint& b) { return vint{ cmpeq_epi8(vint{_mm512_min_epu8(a.m_value, b.m_value)}, a) }; }
CPPSPMD_FORCE_INLINE vint cmpge_epu8(const vint& a, const vint& b) { return vint{ cmple_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epu8(const vint& a, const vint& b) { return vint{ _mm512_andnot_si512(cmpeq_epi8(a, b).m_value, cmpeq_epi8(vint{_mm512_max_epu8(a.m_value, b.m_value)}, a).m_value) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epu8(const vint& a, const vint& b) { return vint{ cmpgt_epu8(b, a) }; }
CPPSPMD_FORCE_INLINE vint absdiff_epu8(const vint& a, const vint& b) { return vint{ _mm512_or_si512(_mm512_subs_epu8(a.m_value, b.m_value), _mm512_subs_epu8(b.m_value, a.m_value)) }; }

CPPSPMD_FORCE_INLINE vint blendv_epi8(const vint& a, const vint& b, const vint &mask) 
{
	__mmask64 k = _mm512_cmplt_epi8_mask(mask.m_value, _mm512_setzero_si512());
	return vint(_mm512_mask_blend_epi8(k, a.m_value, b.m_value));
}

CPPSPMD_FORCE_INLINE vint blendv_epi32(const vint& a, const vint& b, const vint &mask) 
{ 
	__mmask16 k = _mm512_cmplt_epi32_mask(mask.m_value, _mm512_setzero_si512());
	return vint(_mm512_mask_blend_epi32(k, a.m_value, b.m_value));
}

CPPSPMD_FORCE_INLINE vint undefined_vint() { return vint{ _mm512_undefined_epi32() }; }
CPPSPMD_FORCE_INLINE vfloat undefined_vfloat() { return vfloat{ _mm512_undefined_ps() }; }

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int32's in each 128-bit lane.
#define VINT_LANE_SHUFFLE_EPI32(a, control) vint(_mm512_shuffle_epi32((a).m_value, control))

// control is an 8-bit immediate value containing 4 2-bit indices which shuffles the int16's in either the high or low 64-bit lane.
#define VINT_LANE_SHUFFLELO_EPI16(a, control) vint(_mm512_shufflelo_epi16((a).m_value, control))
#define VINT_LANE_SHUFFLEHI_EPI16(a, control) vint(_mm512_shufflehi_epi16((a).m_value, control))

#define VINT_LANE_SHUFFLE_MASK(a, b, c, d) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))
#define VINT_LANE_SHUFFLE_MASK_R(d, c, b, a) ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))

#define VINT_LANE_SHIFT_LEFT_BYTES(a, l) vint(_mm512_bslli_epi128((a).m_value, l))
#define VINT_LANE_SHIFT_RIGHT_BYTES(a, l) vint(_mm512_bsrli_epi128((a).m_value, l))

CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi8(const vint& a, const vint& b) { return vint(_mm512_unpacklo_epi8(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi8(const vint& a, const vint& b) { return vint(_mm512_unpackhi_epi8(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi16(const vint& a, const vint& b) { return vint(_mm512_unpacklo_epi16(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi16(const vint& a, const vint& b) { return vint(_mm512_unpackhi_epi16(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi32(const vint& a, const vint& b) { return vint(_mm512_unpacklo_epi32(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi32(const vint& a, const vint& b) { return vint(_mm512_unpackhi_epi32(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint vint_lane_unpacklo_epi64(const vint& a, const vint& b) { return vint(_mm512_unpacklo_epi64(a.m_value, b.m_value)); }
CPPSPMD_FORCE_INLINE vint vint_lane_unpackhi_epi64(const vint& a, const vint& b) { return vint(_mm512_unpackhi_epi64(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint vint_set1_epi8(int8_t a) { return vint(_mm512_set1_epi8(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi16(int16_t a) { return vint(_mm512_set1_epi16(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi32(int32_t a) { return vint(_mm512_set1_epi32(a)); }
CPPSPMD_FORCE_INLINE vint vint_set1_epi64(int64_t a) { return vint(_mm512_set1_epi64(a)); }

CPPSPMD_FORCE_INLINE vint add_epi16(const vint& a, const vint& b) { return vint{ _mm512_add_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epi16(const vint& a, const vint& b) { return vint{ _mm512_adds_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint adds_epu16(const vint& a, const vint& b) { return vint{ _mm512_adds_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint avg_epu16(const vint& a, const vint& b) { return vint{ _mm512_avg_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint sub_epi16(const vint& a, const vint& b) { return vint{ _mm512_sub_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epi16(const vint& a, const vint& b) { return vint{ _mm512_subs_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint subs_epu16(const vint& a, const vint& b) { return vint{ _mm512_subs_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mullo_epi16(const vint& a, const vint& b) { return vint{ _mm512_mullo_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epi16(const vint& a, const vint& b) { return vint{ _mm512_mulhi_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint mulhi_epu16(const vint& a, const vint& b) { return vint{ _mm512_mulhi_epu16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint min_epi16(const vint& a, const vint& b) { return vint{ _mm512_min_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint max_epi16(const vint& a, const vint& b) { return vint{ _mm512_max_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint madd_epi16(const vint& a, const vint& b) { return vint{ _mm512_madd_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint cmpeq_epi16(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi16(_mm512_cmpeq_epi16_mask(a.m_value, b.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512)  }; }
CPPSPMD_FORCE_INLINE vint cmpgt_epi16(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi16(_mm512_cmpgt_epi16_mask(a.m_value, b.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) }; }
CPPSPMD_FORCE_INLINE vint cmplt_epi16(const vint& a, const vint& b) { return vint{ _mm512_mask_blend_epi16(_mm512_cmpgt_epi16_mask(b.m_value, a.m_value), _mm512_setzero_epi32(), *(const __m512i*)g_allones_512) }; }
CPPSPMD_FORCE_INLINE vint packs_epi16(const vint& a, const vint& b) { return vint{ _mm512_packs_epi16(a.m_value, b.m_value) }; }
CPPSPMD_FORCE_INLINE vint packus_epi16(const vint& a, const vint& b) { return vint{ _mm512_packus_epi16(a.m_value, b.m_value) }; }

CPPSPMD_FORCE_INLINE vint uniform_shift_left_epi16(const vint& a, const vint& b) { return vint{ _mm512_sll_epi16(a.m_value, _mm512_castsi512_si128(b.m_value)) }; }
CPPSPMD_FORCE_INLINE vint uniform_arith_shift_right_epi16(const vint& a, const vint& b) { return vint{ _mm512_sra_epi16(a.m_value, _mm512_castsi512_si128(b.m_value)) }; }
CPPSPMD_FORCE_INLINE vint uniform_shift_right_epi16(const vint& a, const vint& b) { return vint{ _mm512_srl_epi16(a.m_value, _mm512_castsi512_si128(b.m_value)) }; }

#define VINT_SHIFT_LEFT_EPI16(a, b) vint(_mm512_slli_epi16((a).m_value, b))
#define VINT_SHIFT_RIGHT_EPI16(a, b) vint(_mm512_srai_epi16((a).m_value, b))
#define VUINT_SHIFT_RIGHT_EPI16(a, b) vint(_mm512_srli_epi16((a).m_value, b))

CPPSPMD_FORCE_INLINE vint mul_epu32(const vint &a, const vint& b) { return vint(_mm512_mul_epu32(a.m_value, b.m_value)); }

CPPSPMD_FORCE_INLINE vint div_epi32(const vint &a, const vint& b)
{
	__m512d al = _mm512_cvtepi32_pd(_mm512_castsi512_si256(a.m_value));
	__m512d ah = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(a.m_value, 1));

	__m512d bl = _mm512_cvtepi32_pd(_mm512_castsi512_si256(b.m_value));
	__m512d bh = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(b.m_value, 1));

	__m512d rl = _mm512_div_pd(al, bl);
	__m512d rh = _mm512_div_pd(ah, bh);

	__m256i rli = _mm512_cvttpd_epi32(rl);
	__m256i rhi = _mm512_cvttpd_epi32(rh);
		
	return vint(_mm512_inserti32x8(_mm512_castsi256_si512(rli), rhi, 0x1));
}

CPPSPMD_FORCE_INLINE vint mod_epi32(const vint &a, const vint& b)
{
	vint aa = abs(a), ab = abs(b);
	vint q = div_epi32(aa, ab);
	vint r = aa - q * ab;
	return spmd_ternaryi(a < 0, -r, r);
}

CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, const vint& b)
{
	return div_epi32(a, b);
}

CPPSPMD_FORCE_INLINE vint operator/ (const vint& a, int b)
{
	return div_epi32(a, vint(b));
}

CPPSPMD_FORCE_INLINE vint operator% (const vint& a, const vint& b)
{
	return mod_epi32(a, b);
}

CPPSPMD_FORCE_INLINE vint operator% (const vint& a, int b)
{
	return mod_epi32(a, vint(b));
}

CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, const vint& b)
{
	return vint{ _mm512_sllv_epi32(a.m_value, b.m_value) };
}

// uniform shift left
CPPSPMD_FORCE_INLINE vint operator<< (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i*)g_x_128))));
	return vint{ _mm512_sll_epi32(a.m_value, bv) };
}

// uniform arithmetic shift right
CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i*)g_x_128))));
	return vint{ _mm512_sra_epi32(a.m_value, bv) };
}

// uniform shift right
CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, int b)
{
	__m128i bv = _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(b)), _mm_castsi128_ps(_mm_load_si128((const __m128i*)g_x_128))));
	return vint{ _mm512_srl_epi32(a.m_value, bv) };
}

CPPSPMD_FORCE_INLINE vint operator>> (const vint& a, const vint& b)
{
	return vint{ _mm512_srav_epi32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint vuint_shift_right(const vint& a, const vint& b)
{
	return vint{ _mm512_srlv_epi32(a.m_value, b.m_value) };
}

CPPSPMD_FORCE_INLINE vint vuint_shift_right_not_zero(const vint& a, const vint& b) { return vuint_shift_right(a, b); }

#undef VINT_SHIFT_LEFT
#undef VINT_SHIFT_RIGHT
#undef VUINT_SHIFT_RIGHT

// Shift left/right by a uniform immediate constant
#define VINT_SHIFT_LEFT(a, b) vint( _mm512_slli_epi32( (a).m_value, (b) ) ) 
#define VINT_SHIFT_RIGHT(a, b) vint( _mm512_srai_epi32( (a).m_value, (b) ) )
#define VUINT_SHIFT_RIGHT(a, b) vint( _mm512_srli_epi32( (a).m_value, (b) ) )
#define VINT_ROT(x, k) (VINT_SHIFT_LEFT((x), (k)) | VUINT_SHIFT_RIGHT((x), 32 - (k)))

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, const lint& b) 
{ 
	__mmask16 v = _mm512_cmpeq_epi32_mask(a.m_value, b.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator==(const lint& a, int b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator==(int a, const lint& b) { return vint(a) == vint(b); }
CPPSPMD_FORCE_INLINE vbool operator<(const lint& a, const lint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(b.m_value, a.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator>(const lint& a, const lint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(a.m_value, b.m_value);
	return vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator<=(const lint& a, const lint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(a.m_value, b.m_value);
	return !vbool{ v }; 
}

CPPSPMD_FORCE_INLINE vbool operator>=(const lint& a, const lint& b) 
{ 
	__mmask16 v = _mm512_cmpgt_epi32_mask(b.m_value, a.m_value);
	return !vbool{ v }; 
}

CPPSPMD_FORCE_INLINE float extract(const vfloat& v, int instance) 
{ 
	assert(instance < 16); 
	CPPSPMD_ALIGN(64) float values[16]; 
	_mm512_store_ps(values, v.m_value); 
	return values[instance]; 
}
	
CPPSPMD_FORCE_INLINE int extract(const vint& v, int instance) 
{ 
	assert(instance < 16); 
	CPPSPMD_ALIGN(64) int values[16]; 
	_mm512_store_epi32(values, v.m_value); 
	return values[instance]; 
}
	
CPPSPMD_FORCE_INLINE int extract(const lint& v, int instance) 
{ 
	assert(instance < 16); 
	CPPSPMD_ALIGN(64) int values[16]; 
	_mm512_store_epi32(values, v.m_value); 
	return values[instance]; 
}

CPPSPMD_FORCE_INLINE bool extract(const vbool& v, int instance) 
{ 
	assert(instance < 16); 
	return ((v.get_movemask() >> instance) & 1) != 0; 
}

#undef VINT_EXTRACT
#undef VBOOL_EXTRACT
#undef VFLOAT_EXTRACT

CPPSPMD_FORCE_INLINE float cast_int_to_float(int v) { return *(const float*)&v; }

#define CPPSPMD_EXTRACT_EPI32(v, idx) _mm_extract_epi32(_mm512_extracti32x4_epi32(v, (idx) >> 2), (idx) & 3)
#define CPPSPMD_EXTRACT_PS(v, idx) cast_int_to_float(_mm_extract_ps(_mm512_extractf32x4_ps(v, (idx) >> 2), (idx) & 3))

#define VINT_EXTRACT(v, instance) CPPSPMD_EXTRACT_EPI32((v).m_value, instance)
#define VBOOL_EXTRACT(v, instance) (((_cvtmask16_u32(v.m_value) >> (instance)) & 1) != 0)
#define VFLOAT_EXTRACT(v, instance) CPPSPMD_EXTRACT_PS((v).m_value, instance)

CPPSPMD_FORCE_INLINE vfloat& insert(vfloat& v, int instance, float f)
{
	assert(instance < 16);
	CPPSPMD_ALIGN(64) float values[16];
	_mm512_store_ps(values, v.m_value);
	values[instance] = f;
	v.m_value = _mm512_load_ps(values);
	return v;
}

CPPSPMD_FORCE_INLINE vint& insert(vint& v, int instance, int i)
{
	assert(instance < 16);
	CPPSPMD_ALIGN(64) int values[16];
	_mm512_store_si512(values, v.m_value);
	values[instance] = i;
	v.m_value = _mm512_load_si512(values);
	return v;
}

CPPSPMD_FORCE_INLINE vint init_lookup4(const uint8_t pTab[16])
{
	__m128i l = _mm_loadu_si128((const __m128i*)pTab);
		
	__m512i v = _mm512_inserti32x4(_mm512_undefined_epi32(), l, 0);
	v = _mm512_inserti32x4(v, l, 1);
	v = _mm512_inserti32x4(v, l, 2);
	v = _mm512_inserti32x4(v, l, 3);

	return vint{ v };
}

CPPSPMD_FORCE_INLINE vint table_lookup4_8(const vint& a, const vint& table)
{
	return vint{ _mm512_shuffle_epi8(table.m_value, a.m_value) };
}

CPPSPMD_FORCE_INLINE void init_lookup5(const uint8_t pTab[32], vint& table_0, vint& table_1)
{
	__m128i l = _mm_loadu_si128((const __m128i*)pTab);
	__m128i h = _mm_loadu_si128((const __m128i*)(pTab + 16));
		
	__m512i v = _mm512_inserti32x4(_mm512_undefined_epi32(), l, 0);
	v = _mm512_inserti32x4(v, h, 1);
		
	v = _mm512_inserti32x4(v, l, 2);
	v = _mm512_inserti32x4(v, h, 3);

	table_0.m_value = v;
	table_1.m_value = v;
}

CPPSPMD_FORCE_INLINE vint table_lookup5_8(const vint& a, const vint& table_0, const vint& table_1)
{
	(void)table_1;
	return vint{ _mm512_permutexvar_epi8(a.m_value, table_0.m_value) };
}

CPPSPMD_FORCE_INLINE void init_lookup6(const uint8_t pTab[64], vint& table_0, vint& table_1, vint& table_2, vint& table_3)
{
	__m128i a = _mm_loadu_si128((const __m128i*)pTab);
	__m128i b = _mm_loadu_si128((const __m128i*)(pTab + 16));
	__m128i c = _mm_loadu_si128((const __m128i*)(pTab + 32));
	__m128i d = _mm_loadu_si128((const __m128i*)(pTab + 48));

	__m512i v = _mm512_inserti32x4(_mm512_undefined_epi32(), a, 0);
	v = _mm512_inserti32x4(v, b, 1);
	v = _mm512_inserti32x4(v, c, 2);
	v = _mm512_inserti32x4(v, d, 3);

	table_0.m_value = v;
	table_1.m_value = v;
	table_2.m_value = v;
	table_3.m_value = v;
}

CPPSPMD_FORCE_INLINE vint table_lookup6_8(const vint& a, const vint& table_0, const vint& table_1, const vint& table_2, const vint& table_3)
{
	(void)table_1;
	(void)table_2;
	(void)table_3;
	return vint{ _mm512_permutexvar_epi8(a.m_value, table_0.m_value) };
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
	_mm512_mask_i32scatter_ps(dst.m_pValue, m_exec.m_mask, dst.m_vindex, src.m_value, 4);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref& dst, const vfloat& src)
{
	_mm512_i32scatter_ps(dst.m_pValue, dst.m_vindex, src.m_value, 4);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store(const float_vref&& dst, const vfloat& src)
{
	_mm512_mask_i32scatter_ps(dst.m_pValue, m_exec.m_mask, dst.m_vindex, src.m_value, 4);
	return dst;
}

CPPSPMD_FORCE_INLINE const float_vref& spmd_kernel::store_all(const float_vref&& dst, const vfloat& src)
{
	_mm512_i32scatter_ps(dst.m_pValue, dst.m_vindex, src.m_value, 4);
	return dst;
}

#include "cppspmd_flow.h"
#include "cppspmd_math.h"

} // namespace cppspmd_avx512

