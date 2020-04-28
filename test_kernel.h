// test_kernel.h
// This is NOT an example kernel. It's here to help test all the overloads, SPMD conditonals, exec masking, etc.

struct CPPSPMD_MAKE_NAME(test_kernel) : CPPSPMD::spmd_kernel
{
	FILE *pFile;

	void print_vbool(vbool v) { for (uint32_t i = 0; i < CPPSPMD::PROGRAM_COUNT; i++) fprintf(pFile, "%i ", extract(v, i)); fprintf(pFile, "\n"); }
	void print_vint(vint v) { for (uint32_t i = 0; i < CPPSPMD::PROGRAM_COUNT; i++) fprintf(pFile, "%i ", extract(v, i)); fprintf(pFile, "\n"); }
	void print_vfloat(vfloat v) { for (uint32_t i = 0; i < CPPSPMD::PROGRAM_COUNT; i++) fprintf(pFile, "%f ", extract(v, i)); fprintf(pFile, "\n"); }
	void print_float_ptr(const float *v, int n = -1) { for (int i = 0; i < (n >= 0 ? n : CPPSPMD::PROGRAM_COUNT); i++) fprintf(pFile, "%f ", v[i]); fprintf(pFile, "\n"); }
	void print_int_ptr(const int *v, int n = -1) { for (int i = 0; i < (n >= 0 ? n : CPPSPMD::PROGRAM_COUNT); i++) fprintf(pFile, "%i ", v[i]); fprintf(pFile, "\n"); }

	void _call(FILE *p)
	{
		const uint32_t N = 16; // max supported lane widh

		pFile = p;

		int av[N] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		vint a = loadu_linear(av);

		int bv[N] = { 0, 0, 2, 4, 7, 2, 0, 100 };
		vint b = loadu_linear(bv);

		int cv[N] = { 1, 1, 6, -14, -100, 2, 5, 100 };
		vint c = loadu_linear(cv);

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

		float fav[N] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		vfloat fa = loadu_linear(fav);

		float fbv[N] = { 0, 0, 2, 4, 7, 2, 0, 100 };
		vfloat fb = loadu_linear(fbv);

		float fcv[N] = { 1, 1, 6, -14, -100, 2, 5, 100 };
		vfloat fc = loadu_linear(fcv);

		float fdv[N] = { -100.5f, -100.4999f, -100.5001f, 0.5f, -.5f, .4999f, 1.5f, 1.49f };
		vfloat fd = loadu_linear(fdv);

		float fev[N] = { 0.0f, -.5f, .4999f, 1.5f, 1.49f, -100.5f, -100.4999f, -100.5001f };
		vfloat fe = loadu_linear(fev);

		fprintf(pFile, "abs/floor/ceil/min/max:\n");
		print_vfloat(abs(fd));
		print_vfloat(floor(fd));
		print_vfloat(ceil(fd));
		print_vfloat(min(fd, fe));
		print_vfloat(max(fd, fe));
		print_vfloat(sqrt(fd));
		print_vfloat(round_nearest(fd));
		print_vfloat(round_truncate(fd));
		print_vfloat(clamp(fa, fd, fc));

		fprintf(pFile, "float math:\n");
		print_vfloat(-0.0f);
		print_vfloat(fa);
		print_vfloat(-fa);
		print_vfloat(fa + fb);
		print_vfloat(fa - fb);
		print_vfloat(fa / fc);
		print_vfloat(fa * fc);
		print_vfloat(fa * 2.0f);
		print_vfloat(2.0f * fa);
		print_vfloat(2.0f + fa);
		print_vfloat(2.0f - fa);
				
		fprintf(pFile, "\n");
		fprintf(pFile, "Casts:\n");
		print_vfloat(vfloat(a));
		print_vint(vint(fa));
		print_vint((vint)vbool(fa==fa));
		print_vint((vint)vbool(fa!=fa));
		print_vbool((vbool)(vint)vbool(fa==fa));
		print_vbool((vbool)(vint)vbool(fa==fa));
		print_vint((vint)vbool(a==a));
		print_vint((vint)vbool(a!=a));
		print_vbool((vbool)(vint)vbool(a==a));
		print_vbool((vbool)(vint)vbool(a==a));

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
		const int rf[N] = { 0, 5, 2, 3, 4, 5, 6, 7 };
		const int jf[N] = { 0, 1, 2, 3, 4, 5, 6, 5 };
		vint r, j;
		store(r, loadu_linear(rf));
		store(j, loadu_linear(jf));
												
		const int kf[N] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		vint k;
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

			for (uint32_t i = 0; i < CPPSPMD::PROGRAM_COUNT; i++) fprintf(pFile, "%i ", extract(k, i)); fprintf(pFile, "\n");
		}
		SPMD_WEND

		fprintf(pFile, "SPMD_IF:\n");

		vint l;
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

		fprintf(pFile, "SPMD_SIMPLE_IF:\n");

		vint z;
		SPMD_SIMPLE_IF(a < b)
		{
			store(z, 1);

			SPMD_SIMPLE_IF((a + 1) < b)
				store(z, 3);
			SPMD_SIMPLE_END_IF
		}
		SPMD_SIMPLE_ELSE(a < b)
		{
			store(z, 2);
		}
		SPMD_SIMPLE_END_IF
					
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

		fprintf(pFile, "Ternary:\n");
		print_vint(a);
		print_vint(b);
		print_vint(c);
		print_vint( spmd_ternaryi(a > 3, b, c) );
		print_vfloat( spmd_ternaryf(fa > 3.0f, fb, fc) );

		fprintf(pFile, "Stores:\n");

		float t[N];

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
		vint w = a;
		SPMD_WHILE(w < 10)
			SPMD_IF(w > 4)
				spmd_break();
			SPMD_END_IF

			print_vint(w);
			store(w, w + 1);
		SPMD_WEND

		fprintf(pFile, "vint_vref:\n");
		vint rr[10];
		vfloat rrf[10];
		
		int nv[N] = { 0, 5, 6, 7,  1, 2, 4, 3 };
		vint n = loadu_linear(nv);
		vint nn = 0;
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
		int xxi[32];
		memset(xxf, 0xFF, sizeof(xxf));
		memset(xxi, 0xFF, sizeof(xxi));
		spmd_foreach(0, 4, [&](const lint& index, int pcount) 
			{
				store(index[xxf], vfloat(index));
				store(index[xxi], vint(index));

				vfloat k1 = load(index[xxf]);
				vint k2 = load(index[xxi]);

				print_vfloat(k1);
				print_vint(k2);
			});

		print_float_ptr(xxf, 32);
		print_int_ptr(xxi, 32);

		fprintf(pFile, "spmd_foreach 2:\n");
		
		memset(xxf, 0xFF, sizeof(xxf));
		memset(xxi, 0xFF, sizeof(xxi));
		spmd_foreach(0, 2, [&](const lint& index, int pcount) 
			{
				store(index[xxf], vfloat(index));
				store(index[xxi], vint(index));

				vfloat k1 = load(index[xxf]);
				vint k2 = load(index[xxi]);

				print_vfloat(k1);
				print_vint(k2);
			});

		print_float_ptr(xxf, 32);
		print_int_ptr(xxi, 32);

		fprintf(pFile, "spmd_foreach 6:\n");
		
		memset(xxf, 0xFF, sizeof(xxf));
		memset(xxi, 0xFF, sizeof(xxi));
		spmd_foreach(0, 6, [&](const lint& index, int pcount) 
			{
				store(index[xxf], vfloat(index));
				store(index[xxi], vint(index));

				vfloat k1 = load(index[(float *)xxf]);
				vint k2 = load(index[(const int *)xxi]);

				vfloat k3 = load(index[(float *)xxf]);
				vint k4 = load(index[(int *)xxi]);

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
		spmd_foreach(0, 7, [&](const lint& index, int pcount) 
			{
				store(index[xxf], vfloat(index));
				store(index[xxi], vint(index));

				vfloat k1 = load(index[xxf]);
				vint k2 = load(index[xxi]);

				print_vfloat(k1);
				print_vint(k2);
			});

		print_float_ptr(xxf, 32);
		print_int_ptr(xxi, 32);

		fprintf(pFile, "Casts:\n");
		print_vint(cast_vfloat_to_vint(fa));
		print_vfloat(cast_vint_to_vfloat(a));

		print_vfloat(fa);
		print_vfloat(cast_vint_to_vfloat(cast_vfloat_to_vint(fa)));
		print_vint(b);
		print_vint(cast_vfloat_to_vint(cast_vint_to_vfloat(b)));

		fprintf(pFile, "Load all int:\n");
		vint vii = 0;
		print_vint(load_all(a[(const int *)av]));
		print_vint(load_all(a[(int *)av]));

		print_vint(load(a[(const int *)av]));
		print_vint(load(a[(int *)av]));

		fprintf(pFile, "Load all float:\n");
		vfloat vif = 0;
		
		// load_all/load from const float * using varying indices isn't supported yet.
		//print_vfloat(load_all(a[(const float *)fav]));
		print_vfloat(load_all(a[(float *)fav]));

		//print_vfloat(load(a[(const float *)fav]));
		print_vfloat(load(a[(float *)fav]));

	}
};
