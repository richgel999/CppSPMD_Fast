# CppSPMD_Fast
CppSPMD_Fast is a C++ header-only library that implements a subset of [Intel's ispc](https://ispc.github.io/) [SPMD](https://en.wikipedia.org/wiki/SPMD) language in C++, using [SIMD](https://en.wikipedia.org/wiki/SIMD) processor intrinsics. Many ispc kernels can be ported to CppSPMD with few changes. CppSPMD supports math and logical ops using varying ints and floats, varying SPMD flow control constructs (subroutine calls, if/else, while, do, for, foreach, return, break, continue, any, all) implemented with macros or lambdas, and gather/scatter load/store operations. It also contains a simple portable and scalable math and trig approximation library written in CppSPMD itself. 

CppSPMD_Fast currently supports SSE2, SSE4.1, AVX1, AVX2, AVX2 FMA3, and AVX-512. WebAssembly and ARM Neon support is planned. This code is still very closely compatible with the [original CppSPMD project](https://github.com/nlguillemot/CppSPMD).

This is a development repo. It's a lot of brand new code, so there are definitely going to be bugs in here. If you find it useful or interesting, that's wonderful, but please keep in mind this code is actively changing.

IMPORTANT: This code has only been compiled with clang 9.0.0. Earlier versions were compiled with MSVC 2019, but once I added AVX-512 support I had to switch to clang because MSVC wasn't reliable (the compiler was crashing). Everything but AVX-512 should still compile with MSVC 2019, but I haven't verified this yet. gcc should work too, but you may encounter compiler warnings as I haven't tested it yet.

Organization
------------

The different vector instruction sets (SSE2, SSE4.1, AVX1, AVX2, etc.) are different headers that share some common functionality. To use CppSPMD for a specific instruction set, just #include one of these headers:

- cppspmd_float4.h: This is the non-SIMD header, implemented in pure standard C/C++ with no intrinsics usage. It completely emulates the functionality of the other headers, and is therefore a bit slow (to very slow depending on what you do). It's only intended to ease porting to new systems and for testing/validation.
- cppspmd_sse.h: The SSE2 and SSE4.1 header. #define CPPSPMD_SSE2 to 1 before including to get SSE2, otherwise you get SSE4.1. Compile with either "-msse4.1" or "-msse2".
- cppspmd_avx1.h: The "AVX1 alt" header. The "vint" struct contains two __m128i's. Compile with "-mavx".
- cppspmd_avx2.h: This header supports both AVX2 and AVX1, with optional FMA3. #define CPPSPMD_USE_AVX2 to 1 before including to get AVX1, otherwise you get AVX2. The "vint" struct contains a single "__m256i". Compile with either "-mavx" or "-mavx2 -mfma".
SSE operations must be used for most integer ops in AVX1, and there are multiple ways of doing this with different pros and cons, so AVX1 is supported through both headers.
- cppspmd_avx512.h: AVX-512 support. Compile with "-mavx512f -mavx512vbmi -mavx512dq".
- cppspmd_int16_avx2_fma.h: A simplified AVX2 header where vint's are 16-wide int16_t's vs. int32_t's of the other headers. Used to gain more parallelism in purely integer kernels. Doesn't include the built-in math/trig library. Compile with "-mavx2 -mfma".

Common headers (don't include them directly), used by all the headers above except for cppspmd_int16_avx2_fma.h:

- cppspmd_math.h: Math/trig/helper library. Contains approximations for log/log2, exp/exp2, pow, tan, atan/atan2, reciprocal estimate, and reciprocal square root estimate.  Importantly, all vectorized math approximation functions are completely implemented in CppSPMD itself, not using raw intrinsics, so they are portable between vector instruction sets and will generate the same results with all headers (ignoring differences due to FMA3 usage). 
Also contains a simple SPMD random number generator, reverse bits, count leading/trailing zeros, and count set bits helpers. 
- cppspmd_math_declares.h: Declares for the math library.
- cppspmd_sincos.h: Vectorized sin/cos ported from Microsoft's MIT licensed DirectXMath project. (This is currently the only code ported from [DirectXMath](https://github.com/Microsoft/DirectXMath), which falls under Microsoft's MIT license.)
- cppspmd_flow.h: All SPMD flow control functionality is here. Contains the original CppSPMD project's lambda-based flow control (with some bug fixes/improvements), along with the new and more efficient macro-based flow control. Compared to the original CppSPMD project, CppSPMD_fast uses one less execution mask. (The "internal" mask has been removed as it was unnecessary.)

One of the key design goals of CppSPMD_Fast: If you faithfully port the core CppSPMD header to a new vector instruction set, you automatically get the vectorized math/trig operations and SPMD flow control for that instruction set.

References:
-----------

[SPMD Programming In C++: CPPCon 2016](https://github.com/CppCon/CppCon2016/blob/master/Presentations/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC%20-%20Nicolas%20Guillemot%20-%20CppCon%202016.pdf)

[Original CppSPMD project](https://github.com/nlguillemot/CppSPMD)

Here's a CppSPMD BC1 encoding example (written for an earlier version of CppSPMD_Fast - I'll be updating this next):

https://pastebin.com/xaACX3Th

(This is ONLY released on pastebin.com as a CppSPMD example. This BC1 kernel has several quality-released bugs which I am currently fixing.)

Simple Kernel Example
--------------------

This is the [ispc "simple" example](https://github.com/ispc/ispc/blob/master/examples/simple/simple.ispc), ported to CppSPMD. 

It purposely uses the lambda-based flow control implementation of "spmd_foreach", but note that a macro-based implementation (SPMD_FOREACH) is also supported and could have been used. (Whether or not that would lead to more efficient code generation is highly compiler dependent, and also dependent on the kernel itself.)

This example uses "SPMD_SIF", which is a "simple" SPMD varying conditional if statement that doesn't support spmd_break/spmd_continue/spmd_return inside the if or else blocks (for slightly better performance). 

```
using namespace CPPSPMD;

// Must use unique struct/class names, or put them into uniquely named namespaces, otherwise the linker will get confused between the different variants. (At least with MSVC.)
namespace CPPSPMD_NAME(simple_namespace)
{

struct simple : spmd_kernel
{
    void _call(float vin[], float vout[], int count) 
    {
        spmd_foreach(0, count, [&](const lint& index, int pcount)
            {
                // Load the appropriate input value for this program instance.
                vfloat v = load(index[vin]);

                // Do an arbitrary little computation, but at least make the
                // computation dependent on the value being processed
                
                // Important: The CppSPMD macros evaluate the conditional in both the SPMD_SIF/SPMD_SELSE macros, so we cannot change v inside the if block like the ispc sample does.
                // Instead, we write to "result".

                vfloat result;
                SPMD_SIF(v < 3.0f)
                {
                    store(result, v * v);
                }
                SPMD_SELSE(v < 3.0f)
                {
                    store(result, sqrt(v));
                }
                SPMD_SENDIF;

                // And write the result to the output array.
                store(index[vout], result);
            });
    }
};

} // namespace

using namespace CPPSPMD_NAME(simple_namespace);

void CPPSPMD_NAME(simple)(float vin[], float vout[], int count)
{
	spmd_call< simple >(vin, vout, count);
}
```

Macro-based control flow examples:
----------------------------------

The original lambda-based control flow is still available, but in many cases results in less than optimal code generation. New code should prefer the macros:

"Simple" SPMD if or if/else statement:

```
// Simpler/faster spmd_if's for when you know the SPMD control flow won't diverge inside the conditional
// DO NOT use spmd_break(), spmd_continue(), spmd_return(), inside SPMD_SIMPLE_IF's. Nesting SPMD_SIMPLE_IF()'s is OK.
SPMD_SIF(cond)
{
}
// DO NOT invert the conditional.
SPMD_SELSE(cond)
{
}
SPMD_SENDIF
```

SPMD if or if/else statement:

```
// OK to use spmd_break(), spmd_continue(), spmd_return(), SPMD_WHILE, SPMD_SIMPLE_IF, inside SPMD_IF's.
SPMD_IF(cond)
{
}
SPMD_ELSE(cond)
{
}
SPMD_END_IF
```

SPMD while loop (for loops are coming soon):

```
// OK to use spmd_break(), spmd_continue(), spmd_return() inside while loop. OK to use SPMD_IF/SPMD_SIMPLE_IF inside while loop too.
SPMD_WHILE(cond)
{
}
SPMD_WEND
```

Other notes:
------------

- The float4 header is only for testing/debugging/porting use. It takes forever to compile in release and is quite slow (several times slower than just plain C code). Most of the demo's compilation time is spent on float4.

- Each SIMD ISA is a single self-contained header file. 

- If you want to do an SPMD break on a conditional, it's more efficient to use spmd_if_break(cond); than an SPMD_IF and a separate call to spmd_break().

- I just added loads/stores to pointers to vint and vfloat arrays, using vint indices. This isn't super well tested yet. I will be adding int16 and int8/uint8 support as well, through both varying and non-varying indices. (There's already a little bit of int16 load/store support already).

- If you really care about good AVX1 performance, write your code using vfloat's vs. vint's. Even with AVX2, vfloat code seems to perform slightly faster in general. If you don't care about AVX1-only CPU's, then this can be ignored.

- Benchmark your kernel using the AVX1 vs. AVX2 headers, and use the one with the best. perf. The best one to use may be surprising. 

- The AVX2 header supports AVX1-only CPU's too, but it may be less efficient for int32 ops. The way the AVX1 vs. AVX2 headers implement int32 opts is different: The AVX1-only header uses two __m128i's for vint's and a single __m256 for vfloats, and the other uses a single __m256i for vint's).

- SSE 4.1 supports float and int ops equally well in my benchmarking.

- For performance: Don't use vint division or modulus operations. They are implemented in plain scalar code and are quite slow.

- Accessing vint or vfloat arrays (through vint/vfloat pointers) using store()/load()'s with *vint* indices is quite expensive (this is the slowest supported gather/scatter operation). ispc issues automatic warnings about these sorts of operations. Don't do it unless you mean it. 

- Use uniforms as much as possible. Don't use vint or vfloat unless you KNOW and are positive the lanes must have different values. 

- Use plain C uniform control flow as much as possible. For the types of kernels I write (texture encoders), almost all control flow is uniform control flow.

- Use store_all() when you know that you don't need lane masking, for example to stack temporaries or if you know all lanes must be active. Same for load_all().

- If you're careful, you can use store_all() to temps in a SPMD_IF, and then regular store()'s in the SPMD_ELSE. The code in the if block is always evaluated first, before the else.

- Use SPMD_SIMPLE_IF if there is no SPMD control flow of any sort (other than other SPMD_SIMPLE_IF's) inside the conditional. This should lead to better code gen (less exec mask management). There are no checks for this, so you're on your own. If in doubt use SPMD_IF instead.

- There are new helpers for linear and strided loads/stores: store_strided(), load_linear(), etc.

- FMA support is optional for AVX2. I would benchmark with it turned on and off, and only use it if it actually makes a difference. If my kernels, it doesn't.

- Be aware that there are [AVX-VEX transition penalities](https://software.intel.com/sites/default/files/m/d/4/1/d/8/11MC12_Avoiding_2BAVX-SSE_2BTransition_2BPenalties_2Brh_2Bfinal.pdf). To actually ship kernels with multiple SIMD ISA's, you will need to compile them to separate files using the correct MSVC/Intel/etc. compiler options. 

Also see Agner Fog's [dispatch example](https://github.com/tpn/agner/blob/master/vectorclass/dispatch_example.cpp).

License
=======

See [LICENSE](https://github.com/richgel999/CppSPMD_Fast/blob/master/LICENSE).

The CppSPMD_Fast headers fall under the MIT license. Many of the example kernels use Intel's open source license.
