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

