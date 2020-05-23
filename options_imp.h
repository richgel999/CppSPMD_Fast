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
namespace CPPSPMD_NAME(options_namespace)
{

#define BINOMIAL_NUM 64

struct black_scholes_kernel : spmd_kernel
{
    inline vfloat CND(const vfloat& X) 
    {
        vfloat L = abs(X);

        vfloat k = 1.0f / (1.0f + 0.2316419f * L);
        vfloat k2 = k*k;
        vfloat k3 = k2*k;
        vfloat k4 = k2*k2;
        vfloat k5 = k3*k2;

        const float invSqrt2Pi = 0.39894228040f;
        vfloat w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
                    -1.821255978f * k4 + 1.330274429f * k5)
            * invSqrt2Pi * exp_est(-L * L * .5f);

        SPMD_SIF(X > 0.f)
        {
            store(w, 1.0f - w);
        }
        SPMD_SENDIF;

        return w;
    }

    void _call(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count) 
    {
        SPMD_FOREACH(i, 0, count) 
        {
            vfloat S = load(i[Sa]), X = load(i[Xa]), T = load(i[Ta]), r = load(i[ra]), v = load(i[va]);

            vfloat d1 = (log_est(S/X) + (r + v * v * .5f) * T) / (v * sqrt(T));
            vfloat d2 = d1 - v * sqrt(T);

            store(i[result], S * CND(d1) - X * exp_est(-r * T) * CND(d2));
        }
        SPMD_FOREACH_END(i);
    }
};

// Cumulative normal distribution function
struct binomial_put_kernel : spmd_kernel
{
    inline vfloat do_binomial_put(const vfloat& S, const vfloat& X, const vfloat& T, const vfloat& r, const vfloat& v) 
    {
        vfloat V[BINOMIAL_NUM];

        vfloat dt = T / BINOMIAL_NUM;
        vfloat u = exp_est(v * sqrt(dt));
        vfloat d = 1.0f / u;
        vfloat disc = exp_est(r * dt);
        vfloat Pu = (disc - d) / (u - d);

        for (int j = 0; j < BINOMIAL_NUM; ++j) 
        {
            vfloat upow = pow_est(u, (vfloat)(2*j-BINOMIAL_NUM));
            store_all(V[j], max(0.0f, X - S * upow));
        }

        for (int j = BINOMIAL_NUM-1; j >= 0; --j)
            for (int k = 0; k < j; ++k)
                store_all(V[k], ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc);
        return V[0];
    }
    
    void _call(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count) 
    {
        SPMD_FOREACH(i, 0, count)
        {
            vfloat S = load(i[Sa]), X = load(i[Xa]), T = load(i[Ta]), r = load(i[ra]), v = load(i[va]);
            store(i[result], do_binomial_put(S, X, T, r, v));
        }
        SPMD_FOREACH_END(i);
    }
};

} // namespace

using namespace CPPSPMD_NAME(options_namespace);

void CPPSPMD_NAME(black_scholes)(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count)
{
    spmd_call< black_scholes_kernel >(Sa, Xa, Ta, ra, va, result, count);
}

void CPPSPMD_NAME(binomial_put)(float Sa[], float Xa[], float Ta[],
               float ra[], float va[],
               float result[], int count)
{
    spmd_call< binomial_put_kernel >(Sa, Xa, Ta, ra, va, result, count);
}
