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

using namespace CPPSPMD;

// Must use unique struct/class names, or put them into uniquely named namespaces, otherwise the linker will get confused between the different variants. (At least with MSVC.)
namespace CPPSPMD_NAME(mandel_kernel_namespace)
{

struct mandel : spmd_kernel
{
	vint_t _call(const vfloat & c_re, const vfloat & c_im, const vint_t & count);
};

struct mandelbrot_kernel : spmd_kernel
{
	void _call(
		float x0, float y0,
		float x1, float y1,
		int width, int height,
		int maxIterations,
		int output[]);
};

mandel::vint_t mandel::_call(const vfloat & c_re, const vfloat & c_im, const vint_t & count)
{
	vfloat z_re = c_re, z_im = c_im;

	vint_t i = 0;
	SPMD_WHILE(i < count)
	{
		spmd_if_break(vfma(z_re, z_re, z_im * z_im) > 4.0f);

		vfloat new_re = vfms(z_re, z_re, z_im * z_im);
		vfloat new_im = 2.f * z_re * z_im;

		store_all(z_re, c_re + new_re);
		store_all(z_im, c_im + new_im);

		store(i, i + 1);
	}
	SPMD_WEND

	return i;
}

void mandelbrot_kernel::_call(
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
		spmd_foreach(0, width, [&](const lint_t& orig_index, int pcount) {
			(void)pcount;

			// Figure out the position on the complex plane to compute the
			// number of iterations at.  Note that the x values are
			// different across different program instances, since its
			// initializer incorporates the value of the programIndex
			// variable.
			vfloat x = x0 + orig_index * dx;
			vfloat y = y0 + j * dy;

#if CPPSPMD_INT16
			int* p = output + orig_index.get_first_value() + j * width;
			lint_t index = program_index;

			store((vint_t(index) * 2)[(int16_t*)p], spmd_call<mandel>(x, y, maxIterations));
			store((vint_t(index) * 2 + 1)[(int16_t*)p], vint_t(0));
#else
			lint_t index = j * width + orig_index;
			store(index[output], spmd_call<mandel>(x, y, maxIterations));
#endif
			});
	}
}

} // namespace

using namespace CPPSPMD_NAME(mandel_kernel_namespace);

void CPPSPMD_NAME(mandelbrot)(
	float x0, float y0,
	float x1, float y1,
	int width, int height,
	int maxIterations,
	int output[])
{
	spmd_call< mandelbrot_kernel >(x0, y0, x1, y1, width, height, maxIterations, output);
}
