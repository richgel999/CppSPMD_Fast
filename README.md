# CppSPMD_Fast
C++ SPMD test project: macro control flow, SSE2/SSE4.1/AVX1/AVX2/AVX2 FMA/AVX-512 support, lots of optimizations. This code is still closely compatible with the [original CppSPMD project](https://github.com/nlguillemot/CppSPMD).

This a development repo. The implementation is incomplete, it's a lot of brand new code so there are definitely going to be bugs in here, and I am refactoring the code to cut down on the amount of code duplication between the various headers. If you find it useful or interesting, that's wonderful, but please keep in mind this code is actively changing.

IMPORTANT: This code has only been compiled with clang 9.0.0. Earlier versions were compiled with MSVC 2019, but once I added AVX-512 support I had to switch to clang because MSVC wasn't reliable (the compiler was crashing). Everything but AVX-512 should still compile with MSVC 2019, but I haven't verified this yet.

References:
-----------

[SPMD Programming In C++: CPPCon 2016](https://github.com/CppCon/CppCon2016/blob/master/Presentations/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC%20-%20Nicolas%20Guillemot%20-%20CppCon%202016.pdf)

[Original CppSPMD project](https://github.com/nlguillemot/CppSPMD)

Here's a CppSPMD BC1 encoding example:

https://pastebin.com/xaACX3Th

(This is ONLY released on pastebin.com as a CppSPMD example. This BC1 kernel has several quality-released bugs which I am currently fixing.)

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
