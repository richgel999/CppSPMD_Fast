// test_kernel.h
// This is not an example kernel. It's here to help test all the overloads, SPMD conditonals, exec masking, etc.

#include <chrono>
class Stopwatch final
{
public:
	using elapsed_resolution = std::chrono::microseconds;
	Stopwatch() { Reset(); }
	void Reset() { reset_time = clock.now(); }
	double Elapsed() const { return (double)std::chrono::duration_cast<elapsed_resolution>(clock.now() - reset_time).count() * (1.0f / 1000000.0f); }

private:
	std::chrono::high_resolution_clock clock;
	std::chrono::high_resolution_clock::time_point reset_time;
};

#if CPPSPMD_SSE
//#define _XM_SSE4_INTRINSICS_ 1
//#include "DXM/Inc/DirectXMath.h"
//using namespace DirectX;
#endif

using namespace CPPSPMD;

// Must use unique struct/class names, or put them into uniquely named namespaces, otherwise the linker will get confused between the different variants. (At least with MSVC.)
namespace CPPSPMD_NAME(test_kernel_namespace)
{

const int MAX_LANES = 16;

struct test_kernel : spmd_kernel
{
#if CPPSPMD_INT16
	typedef int16_t int_t;
	typedef vint16 vint_t;
	typedef lint16 lint_t;
#else
	typedef int int_t;
	typedef vint vint_t;
	typedef lint lint_t;
#endif

	FILE* pFile;

	void print_vbool(vbool v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) 
			fprintf(pFile, "%i ", extract(v, i)); 
		fprintf(pFile, "\n"); 
	}

	void print_vint(vint_t v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) 
			fprintf(pFile, "%i ", extract(v, i)); 
		fprintf(pFile, "\n"); 
	}
	
	void print_vint16(vint_t v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT * 2; i++) 
			fprintf(pFile, "%i ", (extract(v, i/2) >> ((i & 1) * 16)) & 0xFFFF); 
		fprintf(pFile, "\n"); 
	}

	void print_vint_hex(vint_t v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) 
			fprintf(pFile, "0x%X ", extract(v, i)); 
		fprintf(pFile, "\n"); 
	}

	void print_active_lanes(const char *pPrefix) 
	{ 
		CPPSPMD_DECL(int_t, flags[PROGRAM_COUNT]);
		memset(flags, 0, sizeof(flags));
		storeu_linear(flags, vint_t(1));

		if (pPrefix)
			fprintf(pFile, "%s", pPrefix);

		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) 
		{
			if (flags[i])
				fprintf(pFile, "%u ", i);
		}
		fprintf(pFile, "\n");
	}
	
	void print_vint_u8_hex(vint_t v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT * 4; i++)
		{
			uint32_t t = (extract(v, i / 4) >> ((i & 3) * 8)) & 0xFF;
			fprintf(pFile, "0x%X ", t);
		}
		fprintf(pFile, "\n"); 
	}

	void print_vfloat(vfloat v) 
	{ 
		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) 
			fprintf(pFile, "%f ", extract(v, i)); 
		fprintf(pFile, "\n"); 
	}

	void print_float_ptr(const float* v, int n = -1) 
	{ 
		for (int i = 0; i < (n >= 0 ? n : PROGRAM_COUNT); i++) 
			fprintf(pFile, "%f ", v[i]); 
		fprintf(pFile, "\n"); 
	}

	void print_int_ptr(const int_t * v, int n = -1) 
	{ 
		for (int i = 0; i < (n >= 0 ? n : PROGRAM_COUNT); i++) 
			fprintf(pFile, "%i ", v[i]); 
		fprintf(pFile, "\n"); 
	}

	void print_int_hex_ptr(const int_t * v, int n = -1) 
	{ 
		for (int i = 0; i < (n >= 0 ? n : PROGRAM_COUNT); i++) 
			fprintf(pFile, "0x%X ", v[i]); 
		fprintf(pFile, "\n"); 
	}

	void print_int32_hex_ptr(const int32_t * v, int n = -1) 
	{ 
		for (int i = 0; i < (n >= 0 ? n : PROGRAM_COUNT); i++) 
			fprintf(pFile, "0x%X ", v[i]); 
		fprintf(pFile, "\n"); 
	}

	void print_int(const uint32_t v) 
	{ 
		fprintf(pFile, "%u\n", v); 
	}

	void print_int(const int32_t v) 
	{ 
		fprintf(pFile, "%i\n", v); 
	}

	void print_int(const uint64_t v) 
	{ 
		fprintf(pFile, "0x%llx\n", v); 
	}

	void print_error(const char* pMsg) 
	{ 
		fprintf(pFile, "%s", pMsg); 
		fprintf(stderr, "%s", pMsg); 
	}

	bool _call(FILE * p);

	bool test_array_float();
	void test_array_int();
	void test_sort();
	void test_rand();
	void test_return();
};

// https://burtleburtle.net/bob/rand/smallprng.html
typedef unsigned long int  u4;
typedef struct ranctx { u4 a; u4 b; u4 c; u4 d; } ranctx;

#define rot(x,k) (((x)<<(k))|((x)>>(32-(k))))
static u4 ranval(ranctx* x) 
{
	u4 e = x->a - rot(x->b, 27);
	x->a = x->b ^ rot(x->c, 17);
	x->b = x->c + x->d;
	x->c = x->d + e;
	x->d = e + x->a;
	return x->d;
}

static void raninit(ranctx* x, u4 seed) 
{
	u4 i;
	x->a = 0xf1ea5eed;
	x->b = seed ^ 0xd8487b1f;
	x->c = seed ^ 0xdbadef9a;
	x->d = seed;
	for (i = 0; i < 20; ++i) {
		(void)ranval(x);
	}
}

static uint32_t leading_zero(uint32_t x)
{
	uint32_t n = 0;
	if (x <= 0x0000ffff) n += 16, x <<= 16;
	if (x <= 0x00ffffff) n += 8, x <<= 8;
	if (x <= 0x0fffffff) n += 4, x <<= 4;
	if (x <= 0x3fffffff) n += 2, x <<= 2;
	if (x <= 0x7fffffff) n++;
	return n;
}

static uint32_t trailing_zero(uint32_t v)
{
	unsigned int c = 32; // c will be the number of zero bits on the right
	v &= -signed(v);
	if (v) c--;
	if (v & 0x0000FFFF) c -= 16;
	if (v & 0x00FF00FF) c -= 8;
	if (v & 0x0F0F0F0F) c -= 4;
	if (v & 0x33333333) c -= 2;
	if (v & 0x55555555) c -= 1;
	return c;
}

static int count_bits(uint32_t v)
{
	int total = 0;
	while (v)
	{
		v &= (v - 1);
		total++;
	}
	return total;
}

static float fixed_roundf(float a)
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

bool test_kernel::_call(FILE* p)
{
	SPMD_BEGIN_CALL

	bool succeeded = true;
			
	pFile = p;

	test_rand();

	test_return();

	int_t av[MAX_LANES] = { 0, 1, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15 };
	vint_t a = loadu_linear(av);

	int_t bv[MAX_LANES] = { 0, 0, 2, 4, 7, 2, 0, 100,  8, 6, 12, 13, 15, 14,0, 100 };
	vint_t b = loadu_linear(bv);

	int_t cv[MAX_LANES] = { 1, 1, 6, -14, -100, 2, 5, 100,  1, 1, 6, -14, -100, 2, 5, 100 };
	vint_t c = loadu_linear(cv);

#if !CPPSPMD_INT16
	{
		rand_context z1;
		vint zk1(0xABCD1234);
		for (int i = 0; i < PROGRAM_COUNT; i++)
			insert(zk1, i, 0x1BCD1234 * i + i);
	
		seed_rand(z1, zk1);
		for (int i = 0; i < 65536; i++)
		{
			vint bits(get_randu(z1));
#if CPPSPMD_SSE
			vint shift((get_randu(z1) & 0x7FFFFFFF) % 64);
#else
			vint shift((get_randu(z1) & 0x7FFFFFFF) & 31);
#endif

			vint r = vuint_shift_right(bits, shift);
			
			for (int p = 0; p < PROGRAM_COUNT; p++)
			{
				uint32_t a = extract(bits, p);
				uint32_t s = extract(shift, p);
				uint32_t as = (s >= 32) ? 0 : (a >> s);
				if (as != (uint32_t)extract(r, p))
				{
					printf("!");
					succeeded = false;
				}
			}

			vint rs = bits >> shift;

			for (int p = 0; p < PROGRAM_COUNT; p++)
			{
				int a = extract(bits, p);
				int s = extract(shift, p);
				int as = (s >= 32) ? (a < 0 ? -1 : 0) : (a >> s);
				if (as != extract(rs, p))
				{
					printf("!");
					succeeded = false;
				}
			}
		}

	}
#endif

#if !CPPSPMD_INT16
	fprintf(pFile, "div_epi32:\n");
	rand_context z1;
	vint zk1(0xABCD1234);
	for (int i = 0; i < PROGRAM_COUNT; i++)
		insert(zk1, i, 0x1BCD1234 * i + i);
	seed_rand(z1, zk1);
	for (int i = 0; i < 32; i++)
	{
		vint_t d1 = get_randu(z1), d2 = get_randu(z1);

		SPMD_IF(get_randu(z1) & 1)
		{
			store(d2, d2 & 0xFFFF);
		}
		SPMD_END_IF

		SPMD_WHILE(d1 == 0x80000000)
		{
			store(d2, get_randu(z1));
		}
		SPMD_WEND
						
		SPMD_WHILE((d2 == 0) || (d2 == 0x80000000))
		{
			store(d2, get_randu(z1));
		}
		SPMD_WEND
		
		vint dr0 = d1 / d2;
		vint dr1;
		for (int i = 0; i < PROGRAM_COUNT; i++)
			insert(dr1, i, extract(d1, i) / extract(d2, i));

		vint r0 = d1 % d2;
		vint r1;
		for (int i = 0; i < PROGRAM_COUNT; i++)
			insert(r1, i, extract(d1, i) % extract(d2, i));

		print_vint(dr0);
		print_vint(dr1);
		print_vint(r0);
		print_vint(r1);

		if (spmd_any(dr0 != dr1) || spmd_any(r0 != r1))
		{
			printf("!");
			succeeded = false;
		}
	}
#endif

#if !CPPSPMD_INT16
	int_t ev[MAX_LANES] = { (int_t)0x01020304, (int_t)0x01234567, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304,  (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304, (int_t)0x01020304 };
	vint_t e = loadu_linear(ev);

	fprintf(pFile, "byteswap:\n");
	print_vint_hex(e);
	print_vint_hex(byteswap(e));
	print_vint_hex(byteswap(byteswap(e)));

	vint t1, t2;
	init_reverse_bits(t1, t2);
	
	//for (int i = 0; i < 16; i++)
	//	printf("%u, ", leading_zero(i << 28));
	
	rand_context c1;
	vint k1(0xABCD1234);
	insert(k1, 1, 0xABCD1239);
	insert(k1, 2, 0xAB2D1234);
	insert(k1, 3, 0x1BCD1234);
	seed_rand(c1, k1);
	for (int t = 0; t < 64; t++)
	{
		vint q = get_randu(c1);

		print_vint_hex(q);

		vint lz = count_leading_zeros(q);
		print_vint_hex(lz);

		vint tz = count_trailing_zeros(q);
		print_vint_hex(lz);

		vint sb = count_set_bits(q);

		for (int k = 0; k < PROGRAM_COUNT; k++)
		{
			int e = extract(q, k);
			if (count_bits(e) != extract(sb, k))
			{
				print_error("csb failed!\n");
			}

			int clz = leading_zero(e);
			if (extract(lz, k) != clz)
			{
				print_error("clz failed!\n");
			}

			int ctz = trailing_zero(e);
			if (extract(tz, k) != ctz)
			{
				print_error("ctz failed!\n");
			}
		}

		vint r = 0;
		for (int k = 0; k < PROGRAM_COUNT; k++)
		{
			for (int i = 0; i < 32; i++)
			{
				int byte_ofs = (i + k * 32) / 8;
				int bit_ofs = (i + k * 32) % 8;

				int bit = (((uint8_t*)&q)[byte_ofs] & (1 << bit_ofs)) != 0;

				int r_byte_ofs = (31 - i + k * 32) / 8;
				int r_bit_ofs = (31 - i + k * 32) % 8;

				if (bit)
					((uint8_t*)&r)[r_byte_ofs] |= (1 << r_bit_ofs);
			}
		}

		vint s = reverse_bits(q, t1, t2);
		vint si = reverse_bits(s, t1, t2);

		print_vint(q);
												
		if (spmd_any(r != s) || spmd_any(q != si))
		{
			printf("!");
			succeeded = false;
		}
	}

#endif

	fprintf(pFile, "int math:\n");
	print_vint(a);
	print_vint(-a);
	print_vint(a + b);
	print_vint(a - b);
	print_vint(a / c);
	print_vint(a * c);
	print_vint(a * 2);
	print_vint(2 * a);
	print_vint(2 + a);
	print_vint(2 - a);
	print_vint(a % b);
	print_vint(a % 2);
	print_vint(a / b);
	print_vint(a / 2);

	fprintf(pFile, "int logical:\n");
	print_vint(a & b);
	print_vint(a | b);
	print_vint(a ^ b);

	fprintf(pFile, "bool logical:\n");
	print_vbool(vbool(a));
	print_vbool(vbool(b));
	print_vbool(!vbool(a));
	print_vbool(!vbool(b));
	print_vbool(!vbool(a) || !vbool(b));
	print_vbool(!vbool(a) && !vbool(b));

	fprintf(pFile, "int comps 1:\n");
	print_vbool(a == a);
	print_vbool(a != a);
	print_vbool(a < a);
	print_vbool(a > a);
	print_vbool(a <= a);
	print_vbool(a >= a);
	fprintf(pFile, "\n");

	fprintf(pFile, "int comps 2:\n");
	print_vbool(a == b);
	print_vbool(a != b);
	print_vbool(a < b);
	print_vbool(a > b);
	print_vbool(a <= b);
	print_vbool(a >= b);
	fprintf(pFile, "\n");

	//float fav[MAX_LANES] = { 0, 1, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15 };
	float fav[MAX_LANES] = { 14, 15, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15 };
	vfloat fa = loadu_linear(fav);

	float fbv[MAX_LANES] = { 0, 0, 2, 4, 7, 2, 0, 100,  100, 0, 2, 7, 4, 2, 0, 0 };
	vfloat fb = loadu_linear(fbv);

	//float fcv[MAX_LANES] = { 1, 1, 6, -14, -100, 2, 5, 100,  100, 5, 2, -100, -14, 6, 1, 1 };
	float fcv[MAX_LANES] = { 1, 1, 6, -14, -100, 2, 5, 100,  100, 5, 2, -100, -14, 6, 1, 1 };
	vfloat fc = loadu_linear(fcv);

	float fdv[MAX_LANES] = { -100.5f, -100.4999f, -100.5001f, 0.5f, -.5f, .4999f, 1.5f, 1.49f,  -100.5f, -100.4999f, -100.5001f, 0.5f, -.5f, .4999f, 1.5f, 1.49f, };
	vfloat fd = loadu_linear(fdv);

	float fev[MAX_LANES] = { 0.0f, -.5f, .4999f, 1.5f, 1.49f, -100.5f, -100.4999f, -100.5001f,  0.0f, -.5f, .4999f, 1.5f, 1.49f, -100.5f, -100.4999f, -100.5001f };
	vfloat fe = loadu_linear(fev);

	fprintf(pFile, "abs/floor/ceil/min/max:\n");
	print_vfloat(fd);
	print_vfloat(abs(fd));
	print_vfloat(floor(fd));
	print_vfloat(ceil(fd));
	print_vfloat(min(fd, fe));
	print_vfloat(max(fd, fe));
	print_vint(min(a, b));
	print_vint(min(b, a));
	print_vint(max(b, a));
	print_vint(max(a, b));
	print_vfloat(sqrt(fd));
	print_vfloat(round_nearest(fd));
	print_vfloat(round_truncate(fd));
	print_vfloat(clamp(fa, fd, fc));

	fprintf(pFile, "Extracts:\n");
	print_vfloat(fa);
	fprintf(pFile, "%f %f %f %f\n", VFLOAT_EXTRACT(fa, 0), VFLOAT_EXTRACT(fa, 1), VFLOAT_EXTRACT(fa, 2), VFLOAT_EXTRACT(fa, 3));
	fprintf(pFile, "%f %f %f %f\n", extract(fa, 0), extract(fa, 1), extract(fa, 2), extract(fa, 3));

	vbool vb = a > b;
	print_vbool(vb);
	fprintf(pFile, "%i %i %i %i\n", VBOOL_EXTRACT(vb, 0), VBOOL_EXTRACT(vb, 1), VBOOL_EXTRACT(vb, 2), VBOOL_EXTRACT(vb, 3));
	fprintf(pFile, "%i %i %i %i\n", extract(vb, 0), extract(vb, 1), extract(vb, 2), extract(vb, 3));

	print_vint(a);
	fprintf(pFile, "%i %i %i %i\n", VINT_EXTRACT(a, 0), VINT_EXTRACT(a, 1), VINT_EXTRACT(a, 2), VINT_EXTRACT(a, 3));
	fprintf(pFile, "%i %i %i %i\n", extract(a, 0), extract(a, 1), extract(a, 2), extract(a, 3));

	fprintf(pFile, "int mul:\n");
	print_vint(a * b);
	print_vint(a * -b);
	print_vint(-a * -b);
	print_vint(-a * b);

#if !CPPSPMD_INT16
	fprintf(pFile, "minu/maxu:\n");
	print_vint(minu(vint_t((int)0xFFFFFFFF), vint((int)0x7FFFFFFE)));
	print_vint(maxu(vint_t((int)0x7FFFFFFF), vint((int)0xFFFFFFFE)));

	fprintf(pFile, "table4:\n");
	CPPSPMD_DECL(const uint8_t, table0[16]) = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	CPPSPMD_DECL(const uint8_t, table1[16]) = { 0, 1, 0x80|2, 0x80 | 3, 4, 5, 0x80 | 6, 7, 8, 0x80 | 9, 10, 0x80 | 11, 12, 13, 14, 15 };
	CPPSPMD_DECL(const uint8_t, table2[16]) = { 8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0 };

	vint tt0 = init_lookup4(table0);
	vint tt1 = init_lookup4(table1);
	vint tt2 = init_lookup4(table2);

	print_vint(table_lookup4_8(a, tt0));
	print_vint(table_lookup4_8(b, tt0));
	print_vint(table_lookup4_8(b | 0x80, tt0));
	print_vint(table_lookup4_8(a, tt1));
	print_vint(table_lookup4_8(b, tt1));
	print_vint(table_lookup4_8(b | 0x80, tt1));
	print_vint(table_lookup4_8(a, tt2));
	print_vint(table_lookup4_8(b, tt2));
	print_vint(table_lookup4_8(c | 0x8000, tt2));
	print_vint(table_lookup4_8(c, tt2));
#endif

	fprintf(pFile, "round/truncate/ceil/floor:\n");
	for (int si = -16; si <= 16; si++)
	{
		float sf = (float)si / 4.0f;
		vfloat sfv(sf);

		fprintf(pFile, "--- %3.8f\n", sf);
		fprintf(pFile, "%3.8f %3.8f %3.8f %3.8f\n", fixed_roundf(sf), (float)((int)sf), ceil(sf), floor(sf));
		fprintf(pFile, "%3.8f %3.8f %3.8f %3.8f\n",
			extract(round_nearest(sfv), 0),
			extract(round_truncate(sfv), 0),
			extract(ceil(sfv), 0),
			extract(floor(sfv), 0));
	}
		
	fprintf(pFile, "float math:\n");
	print_vfloat(-0.0f);
	print_vfloat(fa);
	print_vfloat(-fa);
	print_vfloat(fa + fb);
	print_vfloat(fa - fb);

	fprintf(pFile, "fa:\n");
	print_vfloat(fa);
	print_int32_hex_ptr((int*)&fa);

	fprintf(pFile, "fc:\n");
	print_vfloat(fc);
	print_int32_hex_ptr((int*)&fc);

	vfloat dr = fa / fc;

	print_vfloat(dr);
	print_int32_hex_ptr((int*)&dr);

	print_vfloat(fa * fc);
	print_vfloat(fa * 2.0f);
	print_vfloat(2.0f * fa);
	print_vfloat(2.0f + fa);
	print_vfloat(2.0f - fa);

	fprintf(pFile, "\n");
	fprintf(pFile, "Casts:\n");
	print_vfloat(vfloat(a));
	print_vint(vint_t(fa));
	print_vint((vint_t)vbool(fa == fa));
	print_vint((vint_t)vbool(fa != fa));
	print_vbool((vbool)(vint_t)vbool(fa == fa));
	print_vbool((vbool)(vint_t)vbool(fa == fa));
	print_vint((vint_t)vbool(a == a));
	print_vint((vint_t)vbool(a != a));
	print_vbool((vbool)(vint_t)vbool(a == a));
	print_vbool((vbool)(vint_t)vbool(a == a));

	fprintf(pFile, "float comps:\n");
	print_vbool(fa == fa);
	print_vbool(fa != fa);
	print_vbool(fa < fa);
	print_vbool(fa > fa);
	print_vbool(fa <= fa);
	print_vbool(fa >= fa);
	fprintf(pFile, "\n");

	fprintf(pFile, "float comps:\n");
	print_vbool(fa == fb);
	print_vbool(fa != fb);
	print_vbool(fa < fb);
	print_vbool(fa > fb);
	print_vbool(fa <= fb);
	print_vbool(fa >= fb);
	fprintf(pFile, "\n");

	fprintf(pFile, "SPMD_WHILE:\n");
	const int_t rf[MAX_LANES] = { 0, 5, 2, 3, 4, 5, 6, 7,  3, 1, 2, 5, 0, 13, 2, 4 };
	const int_t jf[MAX_LANES] = { 0, 1, 2, 3, 4, 5, 6, 5,  0, 1, 2, 3, 4, 5, 6, 5 };
	vint_t r, j;
	store(r, loadu_linear(rf));
	store(j, loadu_linear(jf));

	const int_t kf[MAX_LANES] = { 0, 1, 2, 3, 4, 5, 6, 7,  3, 2, 1, 0, 5, 6, 7, 3 };
	vint_t k;
	store(k, loadu_linear(kf));

	SPMD_WHILE(k < 10)
	{
		SPMD_IF(k == 5)
		{
			store(k, k + 2);
			spmd_continue();
		}
		SPMD_END_IF

		SPMD_IF(k == 9)
		{
			store(k, k + 100);
			spmd_break();
		}
		SPMD_END_IF

		store(k, k + 1);

		for (uint32_t i = 0; i < PROGRAM_COUNT; i++) fprintf(pFile, "%i ", extract(k, i)); fprintf(pFile, "\n");
	}
	SPMD_WEND

	fprintf(pFile, "SPMD_IF:\n");

	vint_t l = 0;
	SPMD_IF(a < b)
	{
		store(l, 1);

		SPMD_IF((a + 1) < b)
			store(l, 3);
		SPMD_END_IF
	}
	SPMD_ELSE(a < b)
	{
		store(l, 2);
	}
	SPMD_END_IF

	print_vint(l);

	fprintf(pFile, "SPMD_SIF:\n");

	vint_t z = 0;
	SPMD_SIF(a < b)
	{
		store(z, 1);

		SPMD_SIF((a + 1) < b)
			store(z, 3);
		SPMD_SENDIF
	}
	SPMD_SELSE(a < b)
	{
		store(z, 2);
	}
	SPMD_SENDIF

	print_vint(z);

	fprintf(pFile, "\n");

	fprintf(pFile, "Shifts:\n");
	print_vint(a);
	print_vint(a << 1);
	print_vint(a >> 1);
	print_vint(vuint_shift_right(a, 1));

	print_vint(VINT_SHIFT_LEFT(a, 1));
	print_vint(VINT_SHIFT_RIGHT(a, 1));
	print_vint(VUINT_SHIFT_RIGHT(a, 1));

	print_vint(VINT_SHIFT_LEFT(a, 32));
	print_vint(VINT_SHIFT_RIGHT(a, 32));
	print_vint(VUINT_SHIFT_RIGHT(a, 32));
	
	fprintf(pFile, "Shifts2:\n");

	int_t shiftv[MAX_LANES] = { 0, 10, 20, 31, 4, 5, 6, 7,  8, 9, 12, 14, 16, 18, 25, 30 };
	vint_t shift = loadu_linear(shiftv);
	print_vint(a << shift);
	print_vint((~a) >> shift);
	print_vint(vuint_shift_right((~a), shift));

	print_vint(a << 32);
	print_vint((~a) >> 31);
	print_vint((~a) >> 32);
	print_vint(vuint_shift_right((~a), 32));
	print_vint(vuint_shift_right((~a), 31));

	fprintf(pFile, "Shifts3:\n");

#if !CPPSPMD_INT16
	srand(1);
	for (int t = 0; t < 32; t++)
	{
		uint32_t i = rand() | (rand() << 16);
		uint32_t j = rand() & 31;

		vint_t q0 = vint((int)i) << vint((int)j);
		int q0s = extract(q0, 0);
		int qs = i << j;

		vint_t r0 = vint((int)i) >> vint((int)j);
		int r0s = extract(r0, 0);
		int rs = i >> j;

		print_vint(q0);
		print_vint(r0);

		if (qs != q0s)
		{
			print_error("Left shift failed!\n");
		}

		if (rs != r0s)
		{
			print_error("Right shift failed!\n");
		}
	}
#endif
	
	fprintf(pFile, "Ternary:\n");
	print_vint(a);
	print_vint(b);
	print_vint(c);
	print_vint(spmd_ternaryi(a > 3, b, c));
	print_vfloat(spmd_ternaryf(fa > 3.0f, fb, fc));

	fprintf(pFile, "Stores:\n");

	CPPSPMD_ALIGN(64) float t[MAX_LANES];

	memset(t, 0xFF, sizeof(t)); store_linear_all(t, fa); print_float_ptr(t);
	memset(t, 0xFF, sizeof(t)); storeu_linear_all(t, fa); print_float_ptr(t);
	memset(t, 0xFF, sizeof(t)); storeu_linear(t, fa); print_float_ptr(t);
	memset(t, 0xFF, sizeof(t)); store_all_strided(t, 1, fa); print_float_ptr(t);
	memset(t, 0xFF, sizeof(t)); store_strided(t, 1, fa); print_float_ptr(t);

	fprintf(pFile, "Masked stores:\n");
	SPMD_IF(a > 2)
	{
		memset(t, 0xFF, sizeof(t)); store_linear_all(t, fa); print_float_ptr(t);
		memset(t, 0xFF, sizeof(t)); storeu_linear_all(t, fa); print_float_ptr(t);
		memset(t, 0xFF, sizeof(t)); storeu_linear(t, fa); print_float_ptr(t);
		memset(t, 0xFF, sizeof(t)); store_all_strided(t, 1, fa); print_float_ptr(t);
		memset(t, 0xFF, sizeof(t)); store_strided(t, 1, fa); print_float_ptr(t);
	}
	SPMD_END_IF

	fprintf(pFile, "Masked loads:\n");
	memset(t, 0xFF, sizeof(t)); store_linear_all(t, fa); print_vfloat(load_linear_all(t));
	memset(t, 0xFF, sizeof(t)); storeu_linear_all(t, fa); print_vfloat(loadu_linear_all(t));
	memset(t, 0xFF, sizeof(t)); storeu_linear(t, fa); print_vfloat(loadu_linear(t));
	memset(t, 0xFF, sizeof(t)); store_all_strided(t, 1, fa); print_vfloat(load_all_strided(t, 1));
	memset(t, 0xFF, sizeof(t)); store_strided(t, 1, fa); print_vfloat(load_strided(t, 1));

	fprintf(pFile, "Masked loads:\n");
	SPMD_IF(a > 2)
	{
		memset(t, 0xFF, sizeof(t)); store_linear_all(t, fa); print_vfloat(load_linear_all(t));
		memset(t, 0xFF, sizeof(t)); storeu_linear_all(t, fa); print_vfloat(loadu_linear_all(t));
		memset(t, 0xFF, sizeof(t)); storeu_linear(t, fa); print_vfloat(loadu_linear(t));
		memset(t, 0xFF, sizeof(t)); store_all_strided(t, 1, fa); print_vfloat(load_all_strided(t, 1));
		memset(t, 0xFF, sizeof(t)); store_strided(t, 1, fa); print_vfloat(load_strided(t, 1));
	}
	SPMD_END_IF

	fprintf(pFile, "SPMD_IF all opt:\n");
	SPMD_IF(a > 1000)
	{
		fprintf(pFile, "Inside 1\n");
	}
	SPMD_ELSE(a > 1000)
	{
		fprintf(pFile, "Inside 2\n");
	}
	SPMD_END_IF

	fprintf(pFile, "WEND spmd_break:\n");
	vint_t w = a;
	SPMD_WHILE(w < 10)
		SPMD_IF(w > 4)
		spmd_break();
	SPMD_END_IF

		print_vint(w);
	store(w, w + 1);
	SPMD_WEND

	fprintf(pFile, "vint_vref:\n");
	vint_t rr[10];
	vfloat rrf[10];

	int_t nv[MAX_LANES] = { 0, 5, 6, 7,  1, 2, 4, 3,  10, 11, 12, 13, 14, 15, 16, 17 };
	vint_t n = loadu_linear(nv);
	vint_t nn = 0;
	SPMD_WHILE(nn < 10)
	{
		store((n % 10)[rr], nn);
		store((n % 10)[rrf], vfloat(nn));

		store(n, n + 1);
		store(nn, nn + 1);
	}
	SPMD_WEND

	for (uint32_t i = 0; i < 10; i++)
		print_vint(rr[i]);

	fprintf(pFile, "spmd_foreach 4:\n");

	float xxf[32];
	int_t xxi[32];
	memset(xxf, 0xFF, sizeof(xxf));
	memset(xxi, 0xFF, sizeof(xxi));
	spmd_foreach(0, 4, [&](const lint_t& index, int pcount)
		{
			(void)pcount;
			store(index[xxf], vfloat(index));
			store(index[xxi], vint_t(index));

			vfloat k1 = load(index[xxf]);
			vint_t k2 = load(index[xxi]);

			print_vfloat(k1);
			print_vint(k2);
		});

	print_float_ptr(xxf, 32);
	print_int_ptr(xxi, 32);

	fprintf(pFile, "spmd_foreach 2:\n");

	memset(xxf, 0xFF, sizeof(xxf));
	memset(xxi, 0xFF, sizeof(xxi));
	spmd_foreach(0, 2, [&](const lint_t& index, int pcount)
		{
			(void)pcount;

			store(index[xxf], vfloat(index));
			store(index[xxi], vint_t(index));

			vfloat k1 = load(index[xxf]);
			vint_t k2 = load(index[xxi]);

			print_vfloat(k1);
			print_vint(k2);
		});

	print_float_ptr(xxf, 32);
	print_int_ptr(xxi, 32);

	fprintf(pFile, "spmd_foreach 6:\n");

	memset(xxf, 0xFF, sizeof(xxf));
	memset(xxi, 0xFF, sizeof(xxi));
	spmd_foreach(0, 6, [&](const lint_t& index, int pcount)
		{
			(void)pcount;

			store(index[xxf], vfloat(index));
			store(index[xxi], vint_t(index));

			vfloat k1 = load(index[(float*)xxf]);
			vint_t k2 = load(index[(const int_t*)xxi]);

			vfloat k3 = load(index[(float*)xxf]);
			vint_t k4 = load(index[(int_t*)xxi]);

			print_vfloat(k1);
			print_vint(k2);

			print_vfloat(k3);
			print_vint(k4);
		});

	print_float_ptr(xxf, 32);
	print_int_ptr(xxi, 32);

	fprintf(pFile, "spmd_foreach 7:\n");

	memset(xxf, 0xFF, sizeof(xxf));
	memset(xxi, 0xFF, sizeof(xxi));
	spmd_foreach(0, 7, [&](const lint_t& index, int pcount)
		{
			(void)pcount;

			store(index[xxf], vfloat(index));
			store(index[xxi], vint_t(index));

			vfloat k1 = load(index[xxf]);
			vint_t k2 = load(index[xxi]);

			print_vfloat(k1);
			print_vint(k2);
		});

	print_float_ptr(xxf, 32);
	print_int_ptr(xxi, 32);

#if !CPPSPMD_INT16
	fprintf(pFile, "Casts:\n");
	print_vint(cast_vfloat_to_vint(fa));
	print_vfloat(cast_vint_to_vfloat(a));

	print_vfloat(fa);
	print_vfloat(cast_vint_to_vfloat(cast_vfloat_to_vint(fa)));
	print_vint(b);
	print_vint(cast_vfloat_to_vint(cast_vint_to_vfloat(b)));
#endif

	fprintf(pFile, "Load all int:\n");
	//vint_t vii = 0;
	print_vint(load_all(a[(const int_t*)av]));
	print_vint(load_all(a[(int_t*)av]));

	print_vint(load(a[(const int_t*)av]));
	print_vint(load(a[(int_t*)av]));

	fprintf(pFile, "Load all float:\n");
	//vfloat vif = 0;

	// load_all/load from const float * using varying indices isn't supported yet.
	//print_vfloat(load_all(a[(const float *)fav]));
	print_vfloat(load_all(a[(float*)fav]));

	//print_vfloat(load(a[(const float *)fav]));
	print_vfloat(load(a[(float*)fav]));

	fprintf(pFile, "SPMD_FOR:\n");

	//vint_t i = loadu_linear(av);
	//SPMD_FOR( void, i < 15)
	SPMD_FOR(vint_t i = loadu_linear(av), i < 15)
	{
		fprintf(pFile, "i: ");
		print_vint(i);

		SPMD_FOR(vint_t jj = i, jj < 10)
		{
			fprintf(pFile, "  j: ");
			print_vint(jj);
		}
		SPMD_END_FOR(store(jj, jj + 1))
	}
	SPMD_END_FOR(store(i, i + 1))

	fprintf(pFile, "Nested while:\n");

	{
		vint_t i = loadu_linear(av);

		SPMD_WHILE(i < 5)
		{
			fprintf(pFile, "i: ");
			print_vint(i);

			vint_t h = loadu_linear(av) + i;

			SPMD_WHILE(h < 20)
			{
				fprintf(pFile, "h: ");
				print_vint(h);

				SPMD_IF(h >= 18)
				{
					store(h, 20);
					spmd_continue();
				}
				SPMD_END_IF

					store(h, h + 1);
			}
			SPMD_WEND

				store(i, i + 1);
		}
		SPMD_WEND
	}

	if (!test_array_float())
		succeeded = false;

	test_array_int();
	test_sort();

#if !CPPSPMD_INT16
	fprintf(pFile, "approx math:\n");
	float qf[MAX_LANES] = { .125f, 2, 3, 4.5f, -5.1f, 6.75f, 7, 10,  12.0f, 17.5f, 24.0f, 100.0f, 39.1f, 200.0f, 32768, 2939393 };

	vfloat qfv = loadu_linear(qf);
	print_vfloat(qfv);

	fprintf(pFile, "exp2_est:\n");
	print_vfloat(exp2_est(qfv));

	fprintf(pFile, "exp_est:\n");
	print_vfloat(exp_est(qfv));

	fprintf(pFile, "log2_est:\n");
	vfloat qflv = log2_est(qfv);
	print_vfloat(qflv);

	vfloat qflev = log_est(qfv);
	print_vfloat(qflev);

	print_vfloat(pow_est(qfv, qfv));
	print_vfloat(sin_est(qfv));
	print_vfloat(cos_est(qfv));
	print_vfloat(tan_est(qfv));
	print_vfloat(atan2_est(qfv, qfv * .5f));
#endif

#if !CPPSPMD_INT16
	fprintf(pFile, "epi8/epu8:\n");

	CPPSPMD_DECL(int8_t, ab[PROGRAM_COUNT * 4]);
	CPPSPMD_DECL(int8_t, bb[PROGRAM_COUNT * 4]);
	CPPSPMD_DECL(int8_t, cb[PROGRAM_COUNT * 4]);
	CPPSPMD_DECL(int8_t, db[PROGRAM_COUNT * 4]);
	CPPSPMD_DECL(int8_t, eb[PROGRAM_COUNT * 4]);
	CPPSPMD_DECL(int8_t, gb[PROGRAM_COUNT * 4]);
	for (int i = 0; i < PROGRAM_COUNT * 4; i++)
	{
		ab[i] = i;
		bb[i] = i * 2;
		cb[i] = (int8_t)-i;
		db[i] = (int8_t)-i * 3;
		eb[i] = (int8_t)127;
		gb[i] = (int8_t)-128;
	}

	vint abv = load_linear_all((const int *)ab);
	vint bbv = load_linear_all((const int*)bb);
	vint cbv = load_linear_all((const int*)cb);
	vint dbv = load_linear_all((const int*)db);
	vint ebv = load_linear_all((const int*)eb);
	vint gbv = load_linear_all((const int*)gb);

	print_vint_u8_hex(add_epi8(abv, bbv));
	print_vint_u8_hex(add_epi8(abv, dbv));
	print_vint_u8_hex(adds_epi8(abv, bbv));
	print_vint_u8_hex(adds_epi8(abv, ebv));
	print_vint_u8_hex(adds_epi8(abv, gbv));
	print_vint_u8_hex(adds_epi8(gbv, gbv));
	print_vint_u8_hex(adds_epi8(ebv, ebv));
	print_vint_u8_hex(adds_epi8(abv, ebv));

	print_vint_u8_hex(sub_epi8(abv, bbv));
	print_vint_u8_hex(sub_epi8(abv, dbv));
	print_vint_u8_hex(subs_epi8(abv, bbv));
	print_vint_u8_hex(subs_epi8(abv, ebv));
	print_vint_u8_hex(subs_epi8(abv, gbv));
	print_vint_u8_hex(subs_epi8(gbv, gbv));
	print_vint_u8_hex(subs_epi8(ebv, ebv));
	print_vint_u8_hex(subs_epi8(abv, ebv));

	print_vint_u8_hex(adds_epu8(abv, bbv));
	print_vint_u8_hex(adds_epu8(abv, ebv));
	print_vint_u8_hex(adds_epu8(abv, gbv));
	print_vint_u8_hex(adds_epu8(gbv, gbv));
	print_vint_u8_hex(adds_epu8(ebv, ebv));
	print_vint_u8_hex(adds_epu8(abv, ebv));

	print_vint_u8_hex(subs_epu8(abv, bbv));
	print_vint_u8_hex(subs_epu8(abv, ebv));
	print_vint_u8_hex(subs_epu8(abv, gbv));
	print_vint_u8_hex(subs_epu8(gbv, gbv));
	print_vint_u8_hex(subs_epu8(ebv, ebv));
	print_vint_u8_hex(subs_epu8(abv, ebv));

	print_vint_u8_hex(avg_epu8(abv, bbv));
	print_vint_u8_hex(avg_epu8(abv, ebv));
	print_vint_u8_hex(avg_epu8(abv, gbv));
	print_vint_u8_hex(avg_epu8(gbv, gbv));
	print_vint_u8_hex(avg_epu8(ebv, ebv));
	print_vint_u8_hex(avg_epu8(abv, ebv));

	print_vint_u8_hex(max_epu8(abv, bbv));
	print_vint_u8_hex(max_epu8(abv, ebv));
	print_vint_u8_hex(max_epu8(abv, gbv));
	print_vint_u8_hex(max_epu8(gbv, gbv));
	print_vint_u8_hex(max_epu8(ebv, ebv));
	print_vint_u8_hex(max_epu8(abv, ebv));

	print_vint_u8_hex(min_epu8(abv, bbv));
	print_vint_u8_hex(min_epu8(abv, ebv));
	print_vint_u8_hex(min_epu8(abv, gbv));
	print_vint_u8_hex(min_epu8(gbv, gbv));
	print_vint_u8_hex(min_epu8(ebv, ebv));
	print_vint_u8_hex(min_epu8(abv, ebv));

	print_vint_u8_hex(cmpeq_epi8(abv, bbv));
	print_vint_u8_hex(cmpeq_epi8(bbv, bbv));
	print_vint_u8_hex(cmpeq_epi8(abv, ebv));
	print_vint_u8_hex(cmpeq_epi8(abv, gbv));
	print_vint_u8_hex(cmpeq_epi8(gbv, gbv));
	print_vint_u8_hex(cmpeq_epi8(ebv, ebv));
	print_vint_u8_hex(cmpeq_epi8(abv, ebv));

	print_vint_u8_hex(cmplt_epi8(abv, bbv));
	print_vint_u8_hex(cmplt_epi8(bbv, bbv));
	print_vint_u8_hex(cmplt_epi8(abv, ebv));
	print_vint_u8_hex(cmplt_epi8(abv, gbv));
	print_vint_u8_hex(cmplt_epi8(gbv, gbv));
	print_vint_u8_hex(cmplt_epi8(ebv, ebv));
	print_vint_u8_hex(cmplt_epi8(abv, ebv));

	print_vint_u8_hex(cmpgt_epi8(abv, bbv));
	print_vint_u8_hex(cmpgt_epi8(bbv, bbv));
	print_vint_u8_hex(cmpgt_epi8(abv, ebv));
	print_vint_u8_hex(cmpgt_epi8(abv, gbv));
	print_vint_u8_hex(cmpgt_epi8(gbv, gbv));
	print_vint_u8_hex(cmpgt_epi8(ebv, ebv));
	print_vint_u8_hex(cmpgt_epi8(abv, ebv));

	print_vint_u8_hex(sad_epu8(abv, bbv));
	print_vint_u8_hex(sad_epu8(bbv, bbv));
	print_vint_u8_hex(sad_epu8(abv, ebv));
	print_vint_u8_hex(sad_epu8(abv, gbv));
	print_vint_u8_hex(sad_epu8(gbv, gbv));
	print_vint_u8_hex(sad_epu8(ebv, ebv));
	print_vint_u8_hex(sad_epu8(abv, ebv));

	print_vint_u8_hex(unpacklo_epi8(abv, bbv));
	print_vint_u8_hex(unpacklo_epi8(bbv, bbv));
	print_vint_u8_hex(unpacklo_epi8(abv, ebv));
	print_vint_u8_hex(unpacklo_epi8(abv, gbv));
	print_vint_u8_hex(unpacklo_epi8(gbv, gbv));
	print_vint_u8_hex(unpacklo_epi8(ebv, ebv));
	print_vint_u8_hex(unpacklo_epi8(abv, ebv));

	print_vint_u8_hex(unpackhi_epi8(abv, bbv));
	print_vint_u8_hex(unpackhi_epi8(bbv, bbv));
	print_vint_u8_hex(unpackhi_epi8(abv, ebv));
	print_vint_u8_hex(unpackhi_epi8(abv, gbv));
	print_vint_u8_hex(unpackhi_epi8(gbv, gbv));
	print_vint_u8_hex(unpackhi_epi8(ebv, ebv));
	print_vint_u8_hex(unpackhi_epi8(abv, ebv));

	print_vint_u8_hex(unpackhi_epi8(abv, bbv));
	print_vint_u8_hex(unpackhi_epi8(bbv, bbv));
	print_vint_u8_hex(unpackhi_epi8(abv, ebv));
	print_vint_u8_hex(unpackhi_epi8(abv, gbv));
	print_vint_u8_hex(unpackhi_epi8(gbv, gbv));
	print_vint_u8_hex(unpackhi_epi8(ebv, ebv));
	print_vint_u8_hex(unpackhi_epi8(abv, ebv));

	print_int(movemask_epi8(abv));
	print_int(movemask_epi8(bbv));
	print_int(movemask_epi8(cbv));
	print_int(movemask_epi8(dbv));
	print_int(movemask_epi8(ebv));
	print_int(movemask_epi8(gbv));
	print_int(movemask_epi8(gbv ^ abv));

	print_int(movemask_epi32(abv));
	print_int(movemask_epi32(bbv));
	print_int(movemask_epi32(cbv));
	print_int(movemask_epi32(dbv));
	print_int(movemask_epi32(ebv));
	print_int(movemask_epi32(gbv));
	print_int(movemask_epi32(gbv ^ abv));

	print_vint_u8_hex(cmple_epu8(abv, bbv));
	print_vint_u8_hex(cmple_epu8(bbv, bbv));
	print_vint_u8_hex(cmple_epu8(abv, ebv));
	print_vint_u8_hex(cmple_epu8(abv, gbv));
	print_vint_u8_hex(cmple_epu8(gbv, gbv));
	print_vint_u8_hex(cmple_epu8(ebv, ebv));
	print_vint_u8_hex(cmple_epu8(abv, ebv));

	print_vint_u8_hex(cmpge_epu8(abv, bbv));
	print_vint_u8_hex(cmpge_epu8(bbv, bbv));
	print_vint_u8_hex(cmpge_epu8(abv, ebv));
	print_vint_u8_hex(cmpge_epu8(abv, gbv));
	print_vint_u8_hex(cmpge_epu8(gbv, gbv));
	print_vint_u8_hex(cmpge_epu8(ebv, ebv));
	print_vint_u8_hex(cmpge_epu8(abv, ebv));

	print_vint_u8_hex(cmpgt_epu8(abv, bbv));
	print_vint_u8_hex(cmpgt_epu8(bbv, bbv));
	print_vint_u8_hex(cmpgt_epu8(abv, ebv));
	print_vint_u8_hex(cmpgt_epu8(abv, gbv));
	print_vint_u8_hex(cmpgt_epu8(gbv, gbv));
	print_vint_u8_hex(cmpgt_epu8(ebv, ebv));
	print_vint_u8_hex(cmpgt_epu8(abv, ebv));

	print_vint_u8_hex(cmplt_epu8(abv, bbv));
	print_vint_u8_hex(cmplt_epu8(bbv, bbv));
	print_vint_u8_hex(cmplt_epu8(abv, ebv));
	print_vint_u8_hex(cmplt_epu8(abv, gbv));
	print_vint_u8_hex(cmplt_epu8(gbv, gbv));
	print_vint_u8_hex(cmplt_epu8(ebv, ebv));
	print_vint_u8_hex(cmplt_epu8(abv, ebv));

	print_vint_u8_hex(absdiff_epu8(abv, bbv));
	print_vint_u8_hex(absdiff_epu8(bbv, bbv));
	print_vint_u8_hex(absdiff_epu8(abv, ebv));
	print_vint_u8_hex(absdiff_epu8(abv, gbv));
	print_vint_u8_hex(absdiff_epu8(gbv, gbv));
	print_vint_u8_hex(absdiff_epu8(ebv, ebv));
	print_vint_u8_hex(absdiff_epu8(abv, ebv));
#endif

	vint_t ux = undefined_vint();
	vfloat uf = undefined_vfloat();
	(void)ux;
	(void)uf;

	fprintf(pFile, "lane shuffles:\n");
	print_vint(VINT_LANE_SHUFFLE_EPI32(a, VINT_LANE_SHUFFLE_MASK(3, 2, 1, 0)));
	print_vint(VINT_LANE_SHUFFLE_EPI32(a, 0 | (1 << 2) | (3 << 4) | (2 << 6)));
	print_vint(VINT_LANE_SHUFFLE_EPI32(a, 2 | (1 << 2) | (3 << 4) | (0 << 6)));
		
	print_vint(VINT_LANE_SHUFFLELO_EPI16(a, 3 | (2 << 2) | (1 << 4) | (0 << 6)));
	print_vint(VINT_LANE_SHUFFLELO_EPI16(a, 0 | (1 << 2) | (3 << 4) | (2 << 6)));
	print_vint(VINT_LANE_SHUFFLELO_EPI16(a, 2 | (1 << 2) | (3 << 4) | (0 << 6)));
	print_vint(VINT_LANE_SHUFFLEHI_EPI16(a, 3 | (2 << 2) | (1 << 4) | (0 << 6)));
	print_vint(VINT_LANE_SHUFFLEHI_EPI16(a, 0 | (1 << 2) | (3 << 4) | (2 << 6)));
	print_vint(VINT_LANE_SHUFFLEHI_EPI16(a, 2 | (1 << 2) | (3 << 4) | (0 << 6)));

	fprintf(pFile, "lane unpack:\n");
	print_vint(vint_lane_unpacklo_epi8(a, b));
	print_vint(vint_lane_unpacklo_epi8(a, c));
	print_vint(vint_lane_unpacklo_epi8(c, a));

	print_vint(vint_lane_unpacklo_epi16(a, b));
	print_vint(vint_lane_unpacklo_epi16(a, c));
	print_vint(vint_lane_unpacklo_epi16(c, a));

	print_vint(vint_lane_unpacklo_epi32(a, b));
	print_vint(vint_lane_unpacklo_epi32(a, c));
	print_vint(vint_lane_unpacklo_epi32(c, a));

	print_vint(vint_lane_unpacklo_epi64(a, b));
	print_vint(vint_lane_unpacklo_epi64(a, c));
	print_vint(vint_lane_unpacklo_epi64(c, a));

	print_vint(vint_lane_unpackhi_epi8(a, b));
	print_vint(vint_lane_unpackhi_epi8(a, c));
	print_vint(vint_lane_unpackhi_epi8(c, a));

	print_vint(vint_lane_unpackhi_epi16(a, b));
	print_vint(vint_lane_unpackhi_epi16(a, c));
	print_vint(vint_lane_unpackhi_epi16(c, a));

	print_vint(vint_lane_unpackhi_epi32(a, b));
	print_vint(vint_lane_unpackhi_epi32(a, c));
	print_vint(vint_lane_unpackhi_epi32(c, a));

	print_vint(vint_lane_unpackhi_epi64(a, b));
	print_vint(vint_lane_unpackhi_epi64(a, c));
	print_vint(vint_lane_unpackhi_epi64(c, a));
			
	fprintf(pFile, "set1:\n");
	print_vint_hex(vint_set1_epi8(0xDE));
	print_vint_hex(vint_set1_epi16(0xDEAD));
	print_vint_hex(vint_set1_epi32(0xDEADBEEF));
	print_vint_hex(vint_set1_epi64(0xDEADBEEFCCCCCCCCULL));

	fprintf(pFile, "lane shift:\n");
	print_vint_hex(VINT_LANE_SHIFT_LEFT_BYTES(a, 0));
	print_vint_hex(VINT_LANE_SHIFT_LEFT_BYTES(a, 1));
	print_vint_hex(VINT_LANE_SHIFT_LEFT_BYTES(a, 4));
	print_vint_hex(VINT_LANE_SHIFT_LEFT_BYTES(a, 12));
	print_vint_hex(VINT_LANE_SHIFT_LEFT_BYTES(a, 16));

	print_vint_hex(VINT_LANE_SHIFT_RIGHT_BYTES(a, 0));
	print_vint_hex(VINT_LANE_SHIFT_RIGHT_BYTES(a, 1));
	print_vint_hex(VINT_LANE_SHIFT_RIGHT_BYTES(a, 4));
	print_vint_hex(VINT_LANE_SHIFT_RIGHT_BYTES(a, 12));
	print_vint_hex(VINT_LANE_SHIFT_RIGHT_BYTES(a, 16));

#if !CPPSPMD_INT16
	fprintf(pFile, "epi16/epu16:\n");

	int16_t dv[MAX_LANES*2] = { 
		32767, -32768, 32000, 16, 0, -3000, 5, 100,  1, 1, 6, -14, -100, 2, 5, 100,
		32767, -32768, 32000, 16, 0, -3000, 5, 100,  1, 1, 6, -14, -100, 2, 5, 100  };
	vint_t d = loadu_linear((int *)dv);

	print_vint16(add_epi16(a, b)); print_vint16(add_epi16(b, d)); print_vint16(add_epi16(d, d));
	print_vint16(adds_epi16(a, b)); print_vint16(adds_epi16(b, d)); print_vint16(adds_epi16(d, d));
	print_vint16(adds_epu16(a, b)); print_vint16(adds_epu16(b, d)); print_vint16(adds_epu16(d, d));
	print_vint16(sub_epi16(a, b)); print_vint16(sub_epi16(b, d)); print_vint16(sub_epi16(d, d));
	print_vint16(subs_epi16(a, b)); print_vint16(subs_epi16(b, d)); print_vint16(subs_epi16(d, d));
	print_vint16(subs_epu16(a, b)); print_vint16(subs_epu16(b, d)); print_vint16(subs_epu16(d, d));
	print_vint16(avg_epu16(a, b)); print_vint16(avg_epu16(b, d)); print_vint16(avg_epu16(d, d));
	print_vint16(mullo_epi16(a, b)); print_vint16(mullo_epi16(b, d)); print_vint16(mullo_epi16(d, d));
	print_vint16(mulhi_epi16(a, b)); print_vint16(mulhi_epi16(b, d)); print_vint16(mulhi_epi16(d, d));
	print_vint16(mulhi_epu16(a, b)); print_vint16(mulhi_epu16(b, d)); print_vint16(mulhi_epu16(d, d));
	print_vint16(min_epi16(a, b)); print_vint16(min_epi16(b, d)); print_vint16(min_epi16(d, d));
	print_vint16(max_epi16(a, b)); print_vint16(max_epi16(b, d)); print_vint16(max_epi16(d, d));
	print_vint16(madd_epi16(a, b)); print_vint16(madd_epi16(b, d)); print_vint16(madd_epi16(d, d));
	print_vint16(packs_epi16(a, b)); print_vint16(packs_epi16(b, d)); print_vint16(packs_epi16(d, d));
	print_vint16(packus_epi16(a, b)); print_vint16(packus_epi16(b, d)); print_vint16(packus_epi16(d, d));

	fprintf(pFile, "compare epi16:\n");
	print_vint16(cmpeq_epi16(a, b)); print_vint16(cmpeq_epi16(b, d)); print_vint16(cmpeq_epi16(d, d));
	print_vint16(cmpgt_epi16(a, b)); print_vint16(cmpgt_epi16(b, d)); print_vint16(cmpgt_epi16(d, d));
	print_vint16(cmplt_epi16(a, b)); print_vint16(cmplt_epi16(b, d)); print_vint16(cmplt_epi16(d, d));

	fprintf(pFile, "shift epi16:\n");
	print_vint16(uniform_shift_left_epi16(a, b)); print_vint16(uniform_shift_left_epi16(b, d)); print_vint16(uniform_shift_left_epi16(d, d));
	print_vint16(uniform_shift_left_epi16(a, vint_set1_epi64(2))); print_vint16(uniform_shift_left_epi16(b, vint_set1_epi64(5))); print_vint16(uniform_shift_left_epi16(d, vint_set1_epi64(8)));
	print_vint16(uniform_shift_right_epi16(a, vint_set1_epi64(2))); print_vint16(uniform_shift_right_epi16(b, vint_set1_epi64(5))); print_vint16(uniform_shift_right_epi16(d, vint_set1_epi64(8)));
	print_vint16(uniform_arith_shift_right_epi16(a, vint_set1_epi64(2))); print_vint16(uniform_arith_shift_right_epi16(b, vint_set1_epi64(5))); print_vint16(uniform_arith_shift_right_epi16(d, vint_set1_epi64(8)));
	print_vint16(VINT_SHIFT_RIGHT_EPI16(a, 2)); print_vint16(VINT_SHIFT_RIGHT_EPI16(b, 5)); print_vint16(VINT_SHIFT_RIGHT_EPI16(d, 8));
	print_vint16(VUINT_SHIFT_RIGHT_EPI16(a, 2)); print_vint16(VUINT_SHIFT_RIGHT_EPI16(b, 5)); print_vint16(VUINT_SHIFT_RIGHT_EPI16(d, 8));

	fprintf(pFile, "blendv:\n");
	print_vint(blendv_epi8(a, b, c));
	print_vint(blendv_epi8(c, a, b));
	print_vint(blendv_epi8(c, b, a));
	
	print_vint(blendv_epi32(a, b, c));
	print_vint(blendv_epi32(c, a, b));
	print_vint(blendv_epi32(c, b, a));
#endif

	fprintf(pFile, "a=");
	print_vint(a);

	print_active_lanes("active lanes before SIF: ");

	SPMD_SIF(a != 0)
	{
		fprintf(pFile, "SPMD_SIF:\n");
		print_active_lanes("active lanes inside SIF: ");

		fprintf(pFile, "SPMD_SSELECT:\n");
					
		SPMD_SSELECT(a)
		{
			SPMD_SCASE(0)
			{
				print_active_lanes("case 0 active lanes: ");
			}
			SPMD_SCASE_END

			SPMD_SCASE(1)
			{
				print_active_lanes("case 1 active lanes: ");
			}
			SPMD_SCASE_END

			SPMD_SCASE(3)
			{
				print_active_lanes("case 3 active lanes: ");
			}
			SPMD_SCASE_END

			SPMD_SCASE(99)
			{
				print_active_lanes("case 99 active lanes: ");
			}
			SPMD_SCASE_END

			SPMD_SDEFAULT
			{
				print_active_lanes("default active lanes: ");
			}
			SPMD_SDEFAULT_END
		}
		SPMD_SSELECT_END
	}
	SPMD_SENDIF
			
	fprintf(pFile, "SPMD_SSELECT:\n");

	vint_t qa = 3;
					
	SPMD_SSELECT(qa)
	{
		SPMD_SCASE(3)
		{
			print_active_lanes("case 3 active lanes: ");
		}
		SPMD_SCASE_END

		SPMD_SCASE(0)
		{
			print_active_lanes("case 0 active lanes: ");
		}
		SPMD_SCASE_END

		SPMD_SCASE(1)
		{
			print_active_lanes("case 1 active lanes: ");
		}
		SPMD_SCASE_END

		SPMD_SCASE(2)
		{
			print_active_lanes("case 2 active lanes: ");
		}
		SPMD_SCASE_END
					
		SPMD_SCASE(99)
		{
			print_active_lanes("case 99 active lanes: ");
		}
		SPMD_SCASE_END

		SPMD_SDEFAULT
		{
			print_active_lanes("default active lanes: ");
		}
		SPMD_SDEFAULT_END
	}
	SPMD_SSELECT_END

	fprintf(pFile, "SPMD_SASELECT:\n");

	SPMD_SIF(a != 0)
	{
		fprintf(pFile, "SPMD_SIF:\n");
		print_active_lanes("active lanes inside SIF: ");

		SPMD_SASELECT(a)
		{
			SPMD_SACASE(3)
			{
				print_active_lanes("case 3 active lanes: ");
			}
			SPMD_SACASE_END

			SPMD_SACASE(0)
			{
				print_active_lanes("case 0 active lanes: ");
			}
			SPMD_SACASE_END

			SPMD_SACASE(1)
			{
				print_active_lanes("case 1 active lanes: ");
			}
			SPMD_SACASE_END

			SPMD_SACASE(99)
			{
				print_active_lanes("case 99 active lanes: ");
			}
			SPMD_SACASE_END

			SPMD_SADEFAULT
			{
				print_active_lanes("default active lanes: ");
			}
			SPMD_SADEFAULT_END
		}
		SPMD_SASELECT_END
	}
	SPMD_SENDIF

	fprintf(pFile, "SPMD_FOREACH 0-8:\n");
	SPMD_FOREACH(loop_index, 0, 8)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 0-3:\n");
	SPMD_FOREACH(loop_index, 0, 3)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 0-1:\n");
	SPMD_FOREACH(loop_index, 0, 1)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 0-17:\n");
	SPMD_FOREACH(loop_index, 0, 17)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 0-9:\n");
	SPMD_FOREACH(loop_index, 0, 9)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 5-7:\n");
	SPMD_FOREACH(loop_index, 5, 7)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_END(loop_index);

	fprintf(pFile, "SPMD_FOREACH 17-0 with spmd_break():\n");
	SPMD_FOREACH(loop_index, 17, 0)
	{
		print_vint(vint_t(loop_index));
		print_active_lanes("active lanes: ");
		spmd_break();
	}
	SPMD_FOREACH_END(loop_index);

	// We don't support nested SPMD_FOREACH, but let's make sure it compiles anyway.
	fprintf(pFile, "Nested outer SPMD_FOREACH 0-7:\n");
	SPMD_FOREACH(loop_index1, 0, 7)
	{
		print_vint(vint_t(loop_index1));
		print_active_lanes("outer active lanes: ");

		fprintf(pFile, "Inner SPMD_FOREACH 5-16:\n");
			
		SPMD_FOREACH(loop_index2, 5, 16)
		{
			print_vint(vint_t(loop_index2));
			print_active_lanes("inner active lanes: ");
		}
		SPMD_FOREACH_END(loop_index2);
	}
	SPMD_FOREACH_END(loop_index1);

#if !CPPSPMD_INT16
	fprintf(pFile, "mulhiu:\n");
	print_vint_hex(mulhiu(a, b));
	print_vint_hex(mulhiu(a, vint(0xFFFFFFFF)));
	print_vint_hex(mulhiu(a, vint(0x7FFFFFFF)));
	print_vint_hex(mulhiu(a, vint(0x1FFFFFFF)));
	print_vint_hex(mulhiu(b, vint(0xFFFFFFFF)));
	print_vint_hex(mulhiu(b, vint(0x7FFFFFFF)));
	print_vint_hex(mulhiu(b, vint(0x1FFFFFFF)));
	print_vint_hex(mulhiu(vint(0xFFFFFFFF), vint(0xEFFFFFFF)));
	print_vint_hex(mulhiu(vint(0x8FFFFFFF), vint(0xDFFFFFFF)));
	print_vint_hex(mulhiu(vint(0x4FFFFFFF), vint(0xCFFFFFFF)));
	print_vint_hex(mulhiu(vint(0x0FFFFFFF), vint(0xAFFFFFFF)));
#endif

	fprintf(pFile, "SPMD_FOREACH_ACTIVE 1:\n");
	SPMD_FOREACH_ACTIVE(index)
	{
		fprintf(pFile, "index=%u\n", (int32_t)index);
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_ACTIVE_END;

	fprintf(pFile, "SPMD_FOREACH_ACTIVE 2:\n");
	SPMD_SIF(vint_t(program_index) >= 4)
	{
		SPMD_FOREACH_ACTIVE(index)
		{
			fprintf(pFile, "index=%u\n", (int32_t)index);
			print_active_lanes("active lanes: ");
		}
		SPMD_FOREACH_ACTIVE_END;
	}
	SPMD_SENDIF;

	fprintf(pFile, "SPMD_FOREACH_ACTIVE 3:\n");
	SPMD_SIF(vint_t(program_index) == 3 || vint_t(program_index) == 1)
	{
		SPMD_FOREACH_ACTIVE(index)
		{
			fprintf(pFile, "index=%u\n", (int32_t)index);
			print_active_lanes("active lanes: ");
		}
		SPMD_FOREACH_ACTIVE_END;
	}
	SPMD_SENDIF;

	fprintf(pFile, "SPMD_FOREACH_UNIQUE_INT 1:\n");
	SPMD_FOREACH_UNIQUE_INT(v, (vint_t)program_index)
	{
		fprintf(pFile, "index=%u\n", (int32_t)v);
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_UNIQUE_INT_END;

	vint_t kz = 0;
	insert(kz, 0, 1);
	insert(kz, 1, 1);
	insert(kz, 2, 1);
	insert(kz, 3, 3);
	if (PROGRAM_COUNT > 4)
	{
		insert(kz, 4, 4);
		insert(kz, 5, 4);
		insert(kz, 6, 4);
		insert(kz, 7, 1);
	}

	fprintf(pFile, "SPMD_FOREACH_UNIQUE_INT 2:\n");
	SPMD_FOREACH_UNIQUE_INT(v, kz)
	{
		fprintf(pFile, "index=%u\n", (int32_t)v);
		print_active_lanes("active lanes: ");
	}
	SPMD_FOREACH_UNIQUE_INT_END;

	fprintf(pFile, "SPMD_UNMASKED_BEGIN:\n");
	SPMD_SIF((vint_t)program_index == 2)
	{
		print_active_lanes("active lanes in SPMD_SIF: ");

		SPMD_UNMASKED_BEGIN
		{
			print_active_lanes("active lanes in SPMD_UNMASKED_BEGIN: ");
		}
		SPMD_UNMASKED_END
	}
	SPMD_SENDIF;

	fprintf(pFile, "Reduce add:\n");
	print_vfloat(fa);
	print_vfloat(fb);
	
	fprintf(pFile, "Reduce add 1: %f\n", reduce_add(fa));
	fprintf(pFile, "Reduce add 2: %f\n", reduce_add(fb));

	for (int i = 0; i < 16; i++)
	{
		SPMD_SIF((vint_t)program_index == i)
		{
			fprintf(pFile, "Reduce add: %f\n", reduce_add(fa));
			fprintf(pFile, "Reduce add: %f\n", reduce_add(fb));
		}
		SPMD_SENDIF;
	}

	print_active_lanes("active lanes: ");
		
	return succeeded;
}

bool test_kernel::test_array_float()
{
	SPMD_BEGIN_CALL

	bool succeeded = true;

	::srand(1);

	for (int t = 0; t < 5; t++)
	{
		fprintf(pFile, "Float array abs %u:\n", t);

		vfloat ar[16][16];
		vint_t ar_size_x = 0, ar_size_y = 0;

		for (int p = 0; p < PROGRAM_COUNT; p++)
		{
			insert(ar_size_x, p, 1 + (::rand() % 16));
			insert(ar_size_y, p, 1 + (::rand() % 16));
		}

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int p = 0; p < PROGRAM_COUNT; p++)
				{
					float f = ((float)::rand() / RAND_MAX) * 2.0f - 1.0f;

					insert(ar[i][j], p, f);
				}
			}
		}

		float arr_m[PROGRAM_COUNT];
		int arr_best_i[PROGRAM_COUNT];
		int arr_best_j[PROGRAM_COUNT];

		for (int p = 0; p < PROGRAM_COUNT; p++)
		{
			fprintf(pFile, "%u: Size %ix%i\n", p, extract(ar_size_x, p), extract(ar_size_y, p));

			float m = 0.0f;
			int best_i = 0, best_j = 0;
			for (int i = 0; i < extract(ar_size_x, p); i++)
			{
				for (int j = 0; j < extract(ar_size_y, p); j++)
				{
					float f = extract(ar[i][j], p);
					if (fabs(f) > m)
					{
						m = fabs(f);
						best_i = i;
						best_j = j;
					}
					fprintf(pFile, "%f ", f);
				}
				fprintf(pFile, "\n");
			}

			arr_m[p] = m;
			arr_best_i[p] = best_i;
			arr_best_j[p] = best_j;

			fprintf(pFile, "%i max: %f, %i %i\n\n", p, m, best_i, best_j);
		}

		vint_t li = -1, lj = -1;
		vfloat largest_abs = 0.0f;
		SPMD_FOR(vint_t i = 0, i < ar_size_x)
		{
			SPMD_FOR(vint_t j = 0, j < ar_size_y)
			{
				vfloat k = abs(load((i * 16 + j)[(vfloat*)ar]));

				SPMD_IF(k > largest_abs)
				{
					store(largest_abs, k);
					store(li, i);
					store(lj, j);
				}
				SPMD_END_IF
			}
			SPMD_END_FOR(store(j, j + 1));
		}
		SPMD_END_FOR(store(i, i + 1));

		for (int i = 0; i < PROGRAM_COUNT; i++)
			fprintf(pFile, "%i max: %f, %i %i\n", i, extract(largest_abs, i), extract(li, i), extract(lj, i));

		fprintf(pFile, "Non-SPMD check:\n");
		for (int i = 0; i < PROGRAM_COUNT; i++)
		{
			fprintf(pFile, "%i max: %f, %i %i\n", i, arr_m[i], arr_best_i[i], arr_best_j[i]);
			if (extract(largest_abs, i) != arr_m[i])
			{
				printf("!");
				succeeded = false;
			}
			if (extract(li, i) != arr_best_i[i])
			{
				printf("!");
				succeeded = false;
			}
			if (extract(lj, i) != arr_best_j[i])
			{
				printf("!");
				succeeded = false;
			}
		}

		fprintf(pFile, "\n");

		VASSERT(abs(load((li * 16 + lj)[(vfloat*)ar])) == largest_abs);

		if (spmd_any(abs(load((li * 16 + lj)[(vfloat*)ar])) != largest_abs))
		{
			fprintf(pFile, "Check failed!\n");
		}

		store((li * 16 + lj)[(vfloat*)ar], vfloat(program_index));

		for (int i = 0; i < PROGRAM_COUNT; i++)
		{
			if (extract(ar[arr_best_i[i]][arr_best_j[i]], i) != i)
			{
				assert(0);
				fprintf(pFile, "Check failed!\n");
			}
		}
	}

	return succeeded;
}

void test_kernel::test_array_int()
{
	SPMD_BEGIN_CALL

	::srand(1);

	for (int t = 0; t < 5; t++)
	{
		fprintf(pFile, "Int array abs %u:\n", t);

		vint_t ar[16][16];
		vint_t ar_size_x = 0, ar_size_y = 0;

		for (int p = 0; p < PROGRAM_COUNT; p++)
		{
			insert(ar_size_x, p, 1 + (::rand() % 16));
			insert(ar_size_y, p, 1 + (::rand() % 16));
		}

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int p = 0; p < PROGRAM_COUNT; p++)
				{
					int_t f = (int_t)(::rand() - (RAND_MAX / 2));

					insert(ar[i][j], p, f);
				}
			}
		}

		int arr_m[PROGRAM_COUNT];
		int arr_best_i[PROGRAM_COUNT];
		int arr_best_j[PROGRAM_COUNT];

		for (int p = 0; p < PROGRAM_COUNT; p++)
		{
			fprintf(pFile, "%u: Size %ix%i\n", p, extract(ar_size_x, p), extract(ar_size_y, p));

			int m = 0;
			int best_i = 0, best_j = 0;
			for (int i = 0; i < extract(ar_size_x, p); i++)
			{
				for (int j = 0; j < extract(ar_size_y, p); j++)
				{
					int f = extract(ar[i][j], p);
					if (abs(f) > m)
					{
						m = abs(f);
						best_i = i;
						best_j = j;
					}
					fprintf(pFile, "%i ", f);
				}
				fprintf(pFile, "\n");
			}

			arr_m[p] = m;
			arr_best_i[p] = best_i;
			arr_best_j[p] = best_j;

			fprintf(pFile, "%i max: %i, %i %i\n\n", p, m, best_i, best_j);
		}

		vint_t li = -1, lj = -1;
		vint_t largest_abs = 0;
		SPMD_FOR(vint_t i = 0, i < ar_size_x)
		{
			SPMD_FOR(vint_t j = 0, j < ar_size_y)
			{
				vint_t k = abs(load((i * 16 + j)[(vint_t*)ar]));

				SPMD_IF(k > largest_abs)
				{
					store(largest_abs, k);
					store(li, i);
					store(lj, j);
				}
				SPMD_END_IF
			}
			SPMD_END_FOR(store(j, j + 1));
		}
		SPMD_END_FOR(store(i, i + 1));

		for (int i = 0; i < PROGRAM_COUNT; i++)
			fprintf(pFile, "%i max: %i, %i %i\n", i, extract(largest_abs, i), extract(li, i), extract(lj, i));

		fprintf(pFile, "Non-SPMD check:\n");
		for (int i = 0; i < PROGRAM_COUNT; i++)
			fprintf(pFile, "%i max: %i, %i %i\n", i, arr_m[i], arr_best_i[i], arr_best_j[i]);

		fprintf(pFile, "\n");

		VASSERT(abs(load((li * 16 + lj)[(vint_t*)ar])) == largest_abs);

		if (spmd_any(abs(load((li * 16 + lj)[(vint_t*)ar])) != largest_abs))
		{
			fprintf(pFile, "Check failed!\n");
		}

		store((li * 16 + lj)[(vint_t*)ar], vint_t(program_index));

		for (int i = 0; i < PROGRAM_COUNT; i++)
		{
			if (extract(ar[arr_best_i[i]][arr_best_j[i]], i) != i)
			{
				assert(0);
				fprintf(pFile, "Check failed!\n");
			}
		}
	}
}

void test_kernel::test_sort()
{
	SPMD_BEGIN_CALL

	fprintf(pFile, "Int sort:\n");

	::srand(100);

	static const int s_gaps[] = { 701, 301, 132, 57, 23, 10, 4, 1 };
	const int NUM_GAPS = sizeof(s_gaps) / sizeof(s_gaps[0]);

#ifdef _DEBUG
	const int MAX_N = 256;
	const int Q = 1;
#else
	const int MAX_N = 4096;
	const int Q = 4;
#endif
	vint_t values[MAX_N];

	for (int q = 0; q < Q; q++)
	{
		const int MAX_LANES = 1 + (::rand() % MAX_N);

		for (int i = 0; i < MAX_LANES; i++)
		{
			for (int p = 0; p < PROGRAM_COUNT; p++)
			{
				int_t v = (int_t)(::rand());
				insert(values[i], p, v);
			}
		}

		vint_t orig_values[MAX_N];
		memcpy(orig_values, values, sizeof(values));

		for (int gap_index = 0; gap_index < NUM_GAPS; gap_index++)
		{
			const int gap = s_gaps[gap_index];
			for (int i = gap; i < MAX_LANES; i++)
			{
				vint_t temp = values[i];

				vint_t j;
				SPMD_FOR(store(j, i), (j >= gap) && load(max(0, (j - gap))[values]) > temp)
				{
					store(j[values], load((j - gap)[values]));
				}
				SPMD_END_FOR(store(j, j - gap));

				store(j[values], temp);
			}
		}

		for (int p = 0; p < PROGRAM_COUNT; p++)
		{
			for (int i = 0; i < MAX_LANES; i++)
			{
				const int v = extract(orig_values[i], p);

				int j;
				for (j = 0; j < MAX_LANES; j++)
				{
					if (extract(values[j], p) == v)
						break;
				}

				if (j == MAX_LANES)
				{
					assert(0);
					fprintf(pFile, "Sort failed!\n");
				}
			}

			for (int i = 0; i < MAX_LANES; i++)
			{
				if (i)
				{
					if (!(extract(values[i], p) >= extract(values[i - 1], p)))
					{
						assert(0);
						fprintf(pFile, "Sort failed!\n");
					}
				}
				fprintf(pFile, "%i ", extract(values[i], p));
			}
			fprintf(pFile, "\n");
		}
	}
}

void test_kernel::test_return()
{
	SPMD_BEGIN_CALL_ALL_LANES

	fprintf(pFile, "test_return:\n");

	const int_t kf[MAX_LANES] = { 0, 1, 8, 3, 4, 5, 6, 7,  3, 2, 1, 0, 5, 6, 7, 3 };
	vint_t k;
	store(k, loadu_linear(kf));

	SPMD_WHILE(k < 10)
	{
		SPMD_IF(k == 5)
		{
			store(k, k + 2);
			spmd_continue();
		}
		SPMD_END_IF

		SPMD_IF(k == 7)
		{
			store(k, k + 2);
			spmd_continue();
		}
		SPMD_END_IF

		SPMD_IF(k == 8)
		{
			spmd_return();
		}
		SPMD_END_IF

		SPMD_IF(k == 9)
		{
			store(k, k + 100);
			spmd_break();
		}
		SPMD_END_IF

		store(k, k + 1);

		print_vint(k);
	}
	SPMD_WEND
}

// Most use unique struct/class names, otherwise the linker will get confused between the different variants. (At least with MSVC.)
void test_kernel::test_rand()
{
	SPMD_BEGIN_CALL

#if !CPPSPMD_INT16
	fprintf(pFile, "test_rand:\n");

	ranctx c0;
	
	raninit(&c0, 0xABCD1234); fprintf(pFile, "----- 0xABCD1234:\n");
	for (int i = 0; i < 4; i++) fprintf(pFile, "0x%X\n", (uint32_t)ranval(&c0));

	raninit(&c0, 0xABCD1239); fprintf(pFile, "----- 0xABCD1239:\n");
	for (int i = 0; i < 4; i++) fprintf(pFile, "0x%X\n", (uint32_t)ranval(&c0));

	raninit(&c0, 0xAB2D1234); fprintf(pFile, "----- 0xAB2D1234:\n");
	for (int i = 0; i < 4; i++) fprintf(pFile, "0x%X\n", (uint32_t)ranval(&c0));
	
	raninit(&c0, 0x1BCD1234); fprintf(pFile, "----- 0x1BCD1234:\n");
	for (int i = 0; i < 4; i++) fprintf(pFile, "0x%X\n", (uint32_t)ranval(&c0));

	rand_context c1;
	vint k(0xABCD1234);
	insert(k, 1, 0xABCD1239);
	insert(k, 2, 0xAB2D1234);
	insert(k, 3, 0x1BCD1234);
	seed_rand(c1, k);
	fprintf(pFile, "Vectorized:\n");
	
	for (int i = 0; i < 4; i++)
		print_vint_hex(get_randu(c1));

	for (int i = 0; i < 8; i++)
		print_vint_hex(get_randi(c1, vint(0x50), vint(0x70)));

	for (int i = 0; i < 8; i++)
		print_vfloat(get_randf(c1, -1.0f, 1.0f));
#endif
}

} // namespace

using namespace CPPSPMD_NAME(test_kernel_namespace);

bool CPPSPMD_NAME(cppspmd_lowlevel_test)(FILE *pFile)
{
	return spmd_call<test_kernel>(pFile);
}
