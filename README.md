# CppSPMD_Fast
C++ SPMD test project: macro control flow, SSE4.1/AVX1/AVX2/AVX2 FMA support, lots of optimizations

This a development repo. It's incomplete and there are bound to be bugs in here. If you find it useful, wonderful, but please keep in mind this code is actively changing.

IMPORTANT: This code has *ONLY* been compiled with Visual Studio 2019 so far. It should compile with VS 2017 (I tested this earlier, but then I made some simple changes). The original CppSPMD code compiled with clang/gcc, but I've basically rewritten 90% of the code (although I kept its basic structure), so there will need to be fixes/changed for gcc/clang compilation.

References:
CppCon presentation:
https://github.com/CppCon/CppCon2016/blob/master/Presentations/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC/SPMD%20Programming%20Using%20C%2B%2B%20and%20ISPC%20-%20Nicolas%20Guillemot%20-%20CppCon%202016.pdf

The original CppSPMD project:
https://github.com/nlguillemot/CppSPMD

Here's a CppSPMD BC1 encoding example:
https://pastebin.com/xaACX3Th
(This is ONLY released as a CppSPMD example. This kernel has some quality bugs which I am currently fixing.)

Macro control flow examples:

"Simple" SPMD if or if/else statement:

```
// Simpler/faster spmd_if's for when you know the SPMD control flow won't diverge inside the conditional
// DO NOT use spmd_break(), spmd_continue(), spmd_return(), inside SPMD_SIMPLE_IF's. Nesting SPMD_SIMPLE_IF()'s is OK.
SPMD_SIMPLE_IF(cond)
{
}
// DO NOT invert the conditional.
SPMD_SIMPLE_ELSE(cond)
{
}
SPMD_SIMPLE_END_IF
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
SPMD_WEND
```

Other notes:
Use store_all() when you know that you don't need lane masking, for example to stack temporaries if you know all lanes are active. Same for load_all().

There are new helpers for linear and strided loads/stores: store_strided(), load_linear(), etc.

