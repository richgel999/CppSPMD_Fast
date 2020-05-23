#if !defined(_MSC_VER)
#if !__AVX2__ || !__FMA__
#error Please check your compiler options
#endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <cmath>

#include "cppspmd_int16_avx2_fma.h"

#include "cppspmd_type_aliases.h"

#include "mandelbrot_declares.h"
#include "mandelbrot_imp.h"

#include "test_kernel_declares.h"
#include "test_kernel_imp.h"

