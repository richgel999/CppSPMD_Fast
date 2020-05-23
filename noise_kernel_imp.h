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
namespace CPPSPMD_NAME(noise_kernel_namespace)
{
    
#define NOISE_PERM_SIZE 256

static const int NoisePerm[2 * NOISE_PERM_SIZE] = 
{
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71,
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,  2, 44,
    154, 163, 70, 221, 153, 101, 155, 167,  43, 172, 9, 129, 22, 39, 253,  19,
    98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228, 251,
    34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
    14, 239, 107, 49, 192, 214,  31, 181, 199, 106, 157, 184, 84, 204, 176, 115,
    121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72,
    243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15,
    131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
    37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
    219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125,
    136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
    231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245,
    40, 244, 102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187,
    208,  89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109,
    198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
    126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
    223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155,
    167,  43, 172, 9, 129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232,
    178, 185,  112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
    12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214,
    31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150,
    254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78,
    66, 215, 61, 156, 180
};

using namespace CPPSPMD;

// Most use unique struct/class names, or put them into uniquely named namespaces, otherwise the linker will get confused between the different variants. (At least with MSVC.)
struct noise_kernel : spmd_kernel
{
    static inline vfloat SmoothStep(vfloat low, vfloat high, vfloat value) 
    {
        vfloat v = clamp((value - low) / (high - low), 0.f, 1.f);
        return v * v * (-2.f * v + 3.f);
    }

    static inline vint Floor2Int(vfloat val)
    {
        return (vint)floor(val);
    }

    inline vfloat Grad(vint x, vint y, vint z, vfloat dx, vfloat dy, vfloat dz)
    {
        vint h = load_all((z + load_all((y + load_all(x[NoisePerm]))[NoisePerm]))[NoisePerm]);

        store_all(h, h & 15);

        vfloat u = spmd_ternaryf((h < 8) || (h == 12) || (h == 13), dx, dy);
        vfloat v = spmd_ternaryf((h < 4) || (h == 12) || (h == 13), dy, dz);

        return spmd_ternaryf((h & 1) != 0, -u, u) + spmd_ternaryf((h & 2) != 0, -v, v);
    }

    static inline vfloat NoiseWeight(vfloat t)
    {
        vfloat t3 = t * t * t;
        vfloat t4 = t3 * t;
        return 6.f * t4 * t - 15.f * t4 + 10.f * t3;
    }
    
    static inline vfloat Lerp(vfloat t, vfloat low, vfloat high)
    {
        return (1.0f - t) * low + t * high;
    }

    vfloat Noise(vfloat x, vfloat y, vfloat z)
    {
        // Compute noise cell coordinates and offsets
        vint ix = Floor2Int(x), iy = Floor2Int(y), iz = Floor2Int(z);
        vfloat dx = x - ix, dy = y - iy, dz = z - iz;

        // Compute gradient weights
        store_all(ix, ix & (NOISE_PERM_SIZE - 1));
        store_all(iy, iy & (NOISE_PERM_SIZE - 1));
        store_all(iz, iz & (NOISE_PERM_SIZE - 1));

        vfloat w000 = Grad(ix, iy, iz, dx, dy, dz);
        vfloat w100 = Grad(ix + 1, iy, iz, dx - 1, dy, dz);
        vfloat w010 = Grad(ix, iy + 1, iz, dx, dy - 1, dz);
        vfloat w110 = Grad(ix + 1, iy + 1, iz, dx - 1, dy - 1, dz);
        vfloat w001 = Grad(ix, iy, iz + 1, dx, dy, dz - 1);
        vfloat w101 = Grad(ix + 1, iy, iz + 1, dx - 1, dy, dz - 1);
        vfloat w011 = Grad(ix, iy + 1, iz + 1, dx, dy - 1, dz - 1);
        vfloat w111 = Grad(ix + 1, iy + 1, iz + 1, dx - 1, dy - 1, dz - 1);

        // Compute trilinear interpolation of weights
        vfloat wx = NoiseWeight(dx), wy = NoiseWeight(dy), wz = NoiseWeight(dz);
        vfloat x00 = Lerp(wx, w000, w100);
        vfloat x10 = Lerp(wx, w010, w110);
        vfloat x01 = Lerp(wx, w001, w101);
        vfloat x11 = Lerp(wx, w011, w111);
        vfloat y0 = Lerp(wy, x00, x10);
        vfloat y1 = Lerp(wy, x01, x11);
        return Lerp(wz, y0, y1);
    }

    vfloat Turbulence(vfloat x, vfloat y, vfloat z, int octaves) 
    {
        vfloat omega = 0.6;

        vfloat sum = 0., lambda = 1.0f, o = 1.0f;
        for (int i = 0; i < octaves; ++i) 
        {
            store_all(sum, sum + abs(o * Noise(lambda * x, lambda * y, lambda * z)));
            store_all(lambda, lambda * 1.99f);
            store_all(o, o * omega);
        }
        return sum * 0.5f;
    }

    void _call(float x0, float y0, float x1,
        float y1, int width, int height,
        float output[])
    {
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;

        for (int j = 0; j < height; j++) 
        {
            for (int i = 0; i < width; i += PROGRAM_COUNT) 
            {
                vfloat x = x0 + (i + (vfloat)program_index) * dx;
                vfloat y = y0 + j * dy;

                lint index = (program_index + j * width + i);
                store(index[output], Turbulence(x, y, 0.6f, 8));
            }
        }
    }
};

} // namespace

using namespace CPPSPMD_NAME(noise_kernel_namespace);

void CPPSPMD_NAME(noise)(float x0, float y0, float x1, float y1, int width, int height, float output[])
{
    spmd_call< noise_kernel >(x0, y0, x1, y1, width, height, output);
}
