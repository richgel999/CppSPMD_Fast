# CppSPMD_Fast
C++ SPMD test project: macro control flow, SSE4.1/AVX1/AVX2/AVX2 FMA support, lots of optimizations

This a development repo. It's incomplete and there are bound to be bugs in here. If you find it useful, wonderful, but please keep in mind this code is actively changing.

IMPORTANT: This code has *ONLY* been compiled with Visual Studio 2019 so far. It should compile with VS 2017 (I tested this earlier, but then I made some simple changes). The original CppSPMD code compiled with clang/gcc, but I've basically rewritten 90% of the code (although I kept its basic structure), so there will need to be fixes/changed for gcc/clang compilation.
