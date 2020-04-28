struct CPPSPMD_MAKE_NAME(mandel) : CPPSPMD::spmd_kernel
{
#if 0
	// Old lambda-based control flow, slow
	vint _call(const vfloat& c_re, const vfloat& c_im, const vint& count)
	{
		vfloat z_re = c_re, z_im = c_im;

		vint i;
		spmd_for([&] { store(i, 0);  }, [&] { return i < count; }, [&] { store(i, i + 1); }, [&] {

			spmd_if(z_re * z_re + z_im * z_im > 4.0f, [&] {
				spmd_break();
				});

			vfloat new_re = z_re * z_re - z_im * z_im;
			vfloat new_im = 2.f * z_re * z_im;

			spmd_unmasked([&] {
				store(z_re, c_re + new_re);
				store(z_im, c_im + new_im);
				});
			});

		return i;
	}
#elif 0
	// Macro control flow, int counter
	vint _call(const vfloat& c_re, const vfloat& c_im, const vint& count)
	{
		vfloat z_re = c_re, z_im = c_im;

		vint i = 0;
		SPMD_WHILE(i < count)
		{
			spmd_if_break(z_re * z_re + z_im * z_im > 4.0f);

			vfloat new_re = z_re * z_re - z_im * z_im;
			vfloat new_im = 2.f * z_re * z_im;

			store_all(z_re, c_re + new_re);
			store_all(z_im, c_im + new_im);

			store(i, i + 1);
		}
		SPMD_WEND

		return i;
	}
#else
	// Macro control flow, with float counter, which is significantly faster with AVX1.
	vint _call(const vfloat& c_re, const vfloat& c_im, const vint& count)
	{
		vfloat z_re = c_re, z_im = c_im;

		vfloat i = 0;
		vfloat countf(count);
		SPMD_WHILE(i < countf)
		{
			spmd_if_break(z_re * z_re + z_im * z_im > 4.0f);

			vfloat new_re = z_re * z_re - z_im * z_im;
			vfloat new_im = 2.f * z_re * z_im;

			store_all(z_re, c_re + new_re);
			store_all(z_im, c_im + new_im);

			store(i, i + 1.0f);
		}
		SPMD_WEND

		return vint(i);
	}
#endif
};

struct CPPSPMD_MAKE_NAME(mandelbrot) : CPPSPMD::spmd_kernel
{
	void _call(
		float x0, float y0,
		float x1, float y1,
		int width, int height,
		int maxIterations,
		int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;

		for (int j = 0; j < height; j++) {
			// Note that we'll be doing programCount computations in parallel,
			// so increment i by that much.  This assumes that width evenly
			// divides programCount.
			spmd_foreach(0, width, [&](const lint& orig_index, int pcount) {
				// Figure out the position on the complex plane to compute the
				// number of iterations at.  Note that the x values are
				// different across different program instances, since its
				// initializer incorporates the value of the programIndex
				// variable.
				vfloat x = x0 + orig_index * dx;
				vfloat y = y0 + j * dy;

				lint index = j * width + orig_index;

				store(index[output], spmd_call<CPPSPMD_MAKE_NAME(mandel)>(x, y, maxIterations));
				});
		}
	}
};
