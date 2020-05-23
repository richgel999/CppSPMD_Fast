#if !defined(_MSC_VER)
#if !__AVX512F__
#error Please check your compiler options
#endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <cmath>

#include "cppspmd_avx512.h"
#include "cppspmd_type_aliases.h"

#include "mandelbrot_declares.h"
#include "mandelbrot_imp.h"

#include "test_kernel_declares.h"
#include "test_kernel_imp.h"

#include "rt_kernel_declares.h"
#include "rt_kernel_imp.h"

#include "simple_declares.h"
#include "simple_imp.h"

#include "volume_kernel_declares.h"
#include "volume_kernel_imp.h"