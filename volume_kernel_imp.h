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
namespace CPPSPMD_NAME(volume_kernel_namespace)
{

struct volume : spmd_kernel
{
#ifdef _MSC_VER
    __declspec(align(16))
#endif
    struct float3
    {
        inline float3() = default;
        inline float3(float xx, float yy, float zz)
            : x(xx), y(yy), z(zz)
        { }

        inline float3 operator*(float f) const {
            return float3(x * f, y * f, z * f);
        }
        inline float3 operator-(const float3& f2) const {
            return float3(x - f2.x, y - f2.y, z - f2.z);
        }
        inline float3 operator*(const float3& f2) const {
            return float3(x * f2.x, y * f2.y, z * f2.z);
        }
        inline float3 operator+(const float3& f2) const {
            return float3(x + f2.x, y + f2.y, z + f2.z);
        }
        inline float3 operator/(const float3& f2) const {
            return float3(x / f2.x, y / f2.y, z / f2.z);
        }
        inline const float& operator[](int i) const { return (&x)[i]; }
        inline float& operator[](int i) { return (&x)[i]; }

        float x, y, z;
        float pad;  // match padding/alignment of ispc version
    }
#ifndef _MSC_VER
    __attribute__((aligned(16)))
#endif
        ;

    // Just enough of a float3 class to do what we need in this file.
    struct vfloat3
    {
        vfloat x;
        vfloat y;
        vfloat z;

        inline vfloat3() = default;

        inline vfloat3(const vfloat& xx, const vfloat& yy, const vfloat& zz)
            : x(xx), y(yy), z(zz)
        { }

        inline vfloat3(const float3& f)
            : x(f.x), y(f.y), z(f.z)
        { }

        inline vfloat3 operator*(const vfloat& f) const {
            return vfloat3(x * f, y * f, z * f);
        }
        inline vfloat3 operator-(const vfloat3& f2) const {
            return vfloat3(x - f2.x, y - f2.y, z - f2.z);
        }
        inline vfloat3 operator*(const vfloat3& f2) const {
            return vfloat3(x * f2.x, y * f2.y, z * f2.z);
        }
        inline vfloat3 operator+(const vfloat3& f2) const {
            return vfloat3(x + f2.x, y + f2.y, z + f2.z);
        }
        inline vfloat3 operator/(const vfloat3& f2) const {
            return vfloat3(x / f2.x, y / f2.y, z / f2.z);
        }
        inline const vfloat& operator[](int i) const { return (&x)[i]; }
        inline vfloat& operator[](int i) { return (&x)[i]; }
    };

    inline vfloat3& vstore(vfloat3& dst, const vfloat3& src)
    {
        spmd_kernel::store(dst.x, src.x);
        spmd_kernel::store(dst.y, src.y);
        spmd_kernel::store(dst.z, src.z);
        return dst;
    }

    struct vRay
    {
        vfloat3 origin, dir;
    };

    inline void generateRay(const float raster2camera[4][4],
        const float camera2world[4][4],
        const vfloat& x, const vfloat& y, vRay& ray) 
    {
        // transform raster coordinate (x, y, 0) to camera space
        vfloat camw = raster2camera[3][3];
        vfloat camx = (raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3]) / camw;
        vfloat camy = (raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3]) / camw;
        vfloat camz = raster2camera[2][3] / camw;

        store_all(ray.dir.x, camera2world[0][0] * camx + camera2world[0][1] * camy + camera2world[0][2] * camz);
        store_all(ray.dir.y, camera2world[1][0] * camx + camera2world[1][1] * camy + camera2world[1][2] * camz);
        store_all(ray.dir.z, camera2world[2][0] * camx + camera2world[2][1] * camy + camera2world[2][2] * camz);

        store_all(ray.origin.x, camera2world[0][3] / camera2world[3][3]);
        store_all(ray.origin.y, camera2world[1][3] / camera2world[3][3]);
        store_all(ray.origin.z, camera2world[2][3] / camera2world[3][3]);
    }

    inline vbool Inside(const vfloat3& p, const vfloat3& pMin, const vfloat3& pMax)
    {
        return (p.x >= pMin.x && p.x <= pMax.x &&
            p.y >= pMin.y && p.y <= pMax.y &&
            p.z >= pMin.z && p.z <= pMax.z);
    }

    vbool IntersectP(const vRay& ray, const vfloat3& pMin, const vfloat3& pMax, vfloat& hit0, vfloat& hit1)
    {
        vfloat t0 = -1e30f, t1 = 1e30f;

        vfloat3 tNear = (pMin - ray.origin) / ray.dir;
        vfloat3 tFar = (pMax - ray.origin) / ray.dir;

        SPMD_SIF(tNear.x > tFar.x)
        {
            swap(tNear.x, tFar.x);
        }
        SPMD_SENDIF

        store(t0, max(tNear.x, t0));
        store(t1, min(tFar.x, t1));

        SPMD_SIF(tNear.y > tFar.y)
        {
            swap(tNear.y, tFar.y);
        }
        SPMD_SENDIF

        store(t0, max(tNear.y, t0));
        store(t1, min(tFar.y, t1));

        SPMD_SIF(tNear.z > tFar.z)
        {
            swap(tNear.z, tFar.z);
        }
        SPMD_SENDIF

        store(t0, max(tNear.z, t0));
        store(t1, min(tFar.z, t1));

        vbool result = t0 <= t1;
        SPMD_SIF(result)
        {
            store(hit0, t0);
            store(hit1, t1);
        }
        SPMD_SENDIF

        return result;
    }

    inline vfloat Lerp(const vfloat& t, const vfloat& a, const vfloat& b)
    {
        return (1.f - t) * a + t * b;
    }

    inline vfloat D(const vint& x, const vint& y, const vint& z, int nVoxels[3], float density[])
    {
        vint xx = clamp(x, 0, nVoxels[0] - 1);
        vint yy = clamp(y, 0, nVoxels[1] - 1);
        vint zz = clamp(z, 0, nVoxels[2] - 1);

        return load((zz * nVoxels[0] * nVoxels[1] + yy * nVoxels[0] + xx)[density]);
    }

    inline vfloat3 Offset(const vfloat3& p, const vfloat3& pMin, const vfloat3& pMax)
    {
        return (p - pMin) / (pMax - pMin);
    }

    vfloat Density(const vfloat3& Pobj, const vfloat3& pMin, const vfloat3& pMax, float density[], int nVoxels[3])
    {
        SPMD_BEGIN_CALL

        vfloat result;
        SPMD_IF(!Inside(Pobj, pMin, pMax))
        {
            store(result, 0);
            spmd_return();
        }
        SPMD_END_IF
                    
        if (!spmd_any())
            return result;

        // Compute voxel coordinates and offsets for _Pobj_
        vfloat3 vox = Offset(Pobj, pMin, pMax);
        store(vox.x, vox.x * nVoxels[0] - .5f);
        store(vox.y, vox.y * nVoxels[1] - .5f);
        store(vox.z, vox.z * nVoxels[2] - .5f);
        vint vx = (vint)(vox.x), vy = (vint)(vox.y), vz = (vint)(vox.z);
        vfloat dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;

        // Trilinearly interpolate density values to compute local density
        vfloat d00 = Lerp(dx, D(vx, vy, vz, nVoxels, density),
            D(vx + 1, vy, vz, nVoxels, density));

        vfloat d10 = Lerp(dx, D(vx, vy + 1, vz, nVoxels, density),
            D(vx + 1, vy + 1, vz, nVoxels, density));

        vfloat d01 = Lerp(dx, D(vx, vy, vz + 1, nVoxels, density),
            D(vx + 1, vy, vz + 1, nVoxels, density));

        vfloat d11 = Lerp(dx, D(vx, vy + 1, vz + 1, nVoxels, density),
            D(vx + 1, vy + 1, vz + 1, nVoxels, density));

        vfloat d0 = Lerp(dy, d00, d10);
        vfloat d1 = Lerp(dy, d01, d11);

        store(result, Lerp(dz, d0, d1));
        return result;
    }

    vfloat transmittance(const float3& p0, const vfloat3& p1, const float3& pMin,
        const float3& pMax, float sigma_t,
        float density[], int nVoxels[3]) 
    {
        SPMD_BEGIN_CALL

        vfloat rayT0(0.0f), rayT1(0.0f);

        vRay ray{ p1, vfloat3(p0) - p1 };

        vfloat result;

        // Find the parametric t range along the ray that is inside the volume.
        SPMD_IF(!IntersectP(ray, pMin, pMax, rayT0, rayT1))
        {
            store(result, 1.0f);
            spmd_return();
        }
        SPMD_END_IF

        if (!spmd_any())
            return result;

        store(rayT0, max(rayT0, 0.f));

        // Accumulate beam transmittance in tau
        vfloat tau = 0.0f;
        vfloat rayLength = sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
            ray.dir.z * ray.dir.z);
        float stepDist = 0.2f;
        vfloat stepT = stepDist / rayLength;

        vfloat t = rayT0;
        vfloat3 pos = ray.origin + ray.dir * rayT0;
        vfloat3 dirStep = ray.dir * stepT;
        SPMD_WHILE(t < rayT1)
        {
            store(tau, tau + stepDist * sigma_t * Density(pos, pMin, pMax, density, nVoxels));
            vstore(pos, pos + dirStep);
            store(t, t + stepT);
        }
        SPMD_WEND

        store(result, exp_est(-tau));
        return result;
    }

    inline vfloat distanceSquared(const vfloat3& a, const vfloat3& b)
    {
        vfloat3 d = a - b;
        return d.x * d.x + d.y * d.y + d.z * d.z;
    }

    vfloat raymarch(float density[], int nVoxels[3], const vRay& ray)
    {
        // Saves the current execution masks, and begins a new SPMD kernel function. This must be done here becuase this kernel may call spmd_return().
        SPMD_BEGIN_CALL

        vfloat rayT0(0.0f), rayT1(0.0f);
        float3 pMin = { .3, -.2, .3 }, pMax = { 1.8, 2.3, 1.8 };
        float3 lightPos = { -1, 4, 1.5 };

        vfloat result;

        SPMD_IF(!IntersectP(ray, pMin, pMax, rayT0, rayT1))
        {
            store(result, 0.0f);
            spmd_return();
        }
        SPMD_END_IF

        if (!spmd_any())
            return result;
                    
        store(rayT0, max(rayT0, 0.f));

        // Parameters that define the volume scattering characteristics and
        // sampling rate for raymarching
        float Le = 0.25f;             // Emission coefficient
        float sigma_a = 10.0f;        // Absorption coefficient
        float sigma_s = 10.0f;        // Scattering coefficient
        float stepDist = 0.025f;      // Ray step amount
        float lightIntensity = 40.0f; // Light source intensity

        vfloat tau = 0.f;  // accumulated beam transmittance
        vfloat L = 0.0f;   // radiance along the ray
        vfloat rayLength = sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
            ray.dir.z * ray.dir.z);
        vfloat stepT = stepDist / rayLength;

        vfloat t = rayT0;
        vfloat3 pos = ray.origin + ray.dir * rayT0;
        vfloat3 dirStep = ray.dir * stepT;
        SPMD_WHILE(t < rayT1)
        {
            vfloat d = Density(pos, pMin, pMax, density, nVoxels);

            // terminate once attenuation is high
            vfloat atten = exp_est(-tau);
            spmd_if_break(atten < 0.005f);

            // spmd_if_break() will disable the execution masks of any active lanes passing the conditional test, but we need to break out of the loop ourselves if all lanes go dead.
            if (!spmd_any())
                break;

            // direct lighting
            vfloat Li = lightIntensity / distanceSquared(lightPos, pos) *
                transmittance(lightPos, pos, pMin, pMax, sigma_a + sigma_s,
                    density, nVoxels);
            store(L, L + stepDist * atten * d * sigma_s * (Li + Le));

            // update beam transmittance
            store(tau, tau + stepDist * (sigma_a + sigma_s) * d);

            vstore(pos, pos + dirStep);
            store(t, t + stepT);
        }
        SPMD_WEND

        // Gamma correction
        store(result, pow_est(L, 1.f / 2.2f));
        return result;
    }

    /* Utility routine used by both the task-based and the single-core entrypoints.
        Renders a tile of the image, covering [x0,x0) * [y0, y1), storing the
        result into the image[] array.
        */
    void volume_tile(int x0, int y0, int x1,
        int y1, float density[], int nVoxels[3],
        const float raster2camera[4][4],
        const float camera2world[4][4],
        int width, int height, float image[]) 
    {
        (void)height;

        // Work on 4x4=16 pixel big tiles of the image.  This function thus
        // implicitly assumes that both (x1-x0) and (y1-y0) are evenly divisble
        // by 4.
        for (int y = y0; y < y1; y += 4) 
        {
            for (int x = x0; x < x1; x += 4) 
            {
                spmd_foreach(0, 16, [&](const lint& o, int p) 
                {
                    (void)p;
                    // These two arrays encode the mapping from [0,15] to
                    // offsets within the 4x4 pixel block so that we render
                    // each pixel inside the block
                    const int xoffsets[16] = { 0, 1, 0, 1, 2, 3, 2, 3,
                                                0, 1, 0, 1, 2, 3, 2, 3 };
                    const int yoffsets[16] = { 0, 0, 1, 1, 0, 0, 1, 1,
                                                2, 2, 3, 3, 2, 2, 3, 3 };

                    // Figure out the pixel to render for this program instance
                    vint xo = x + load(o[xoffsets]), yo = y + load(o[yoffsets]);

                    // Use viewing parameters to compute the corresponding ray
                    // for the pixel
                    vRay ray;
                    generateRay(raster2camera, camera2world, vfloat(xo), vfloat(yo), ray);

                    // And raymarch through the volume to compute the pixel's
                    // value
                    vint offset = yo * width + xo;

                    store(offset[image], raymarch(density, nVoxels, ray));
                });
            }
        }
    }

    void _call(float density[], int nVoxels[3],
        const float raster2camera[4][4],
        const float camera2world[4][4],
        int width, int height, float image[]) 
    {
        volume_tile(0, 0, width, height, density, nVoxels, raster2camera,
            camera2world, width, height, image);
    }
};

} // namespace

using namespace CPPSPMD_NAME(volume_kernel_namespace);

void CPPSPMD_NAME(cppspmd_volume)(
    float density[], int nVoxels[3],
    const float raster2camera[4][4],
    const float camera2world[4][4],
    int width, int height, float image[])
{
    spmd_call< volume >(density, nVoxels, raster2camera, camera2world, width, height, image);
}
