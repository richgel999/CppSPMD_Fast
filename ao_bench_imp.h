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
/*
  Based on Syoyo Fujita's aobench: http://code.google.com/p/aobench
*/

using namespace CPPSPMD;

#define NAO_SAMPLES		8
#define M_PI 3.1415926535f

// Must use unique struct/class names, or put them into uniquely named namespaces, otherwise the linker will get confused between the different variants. (At least with MSVC.)
namespace CPPSPMD_NAME(ao_bench_namespace)
{

struct ao_bench_kernel : spmd_kernel
{

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
    };

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

        inline vfloat3(const float f)
            : x(f), y(f), z(f)
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

    inline vfloat3& vstore_all(vfloat3& dst, const vfloat3& src)
    {
        spmd_kernel::store_all(dst.x, src.x);
        spmd_kernel::store_all(dst.y, src.y);
        spmd_kernel::store_all(dst.z, src.z);
        return dst;
    }

    inline float3 Cross(const float3 &v1, const float3 &v2)
    {
        float v1x = v1.x, v1y = v1.y, v1z = v1.z;
        float v2x = v2.x, v2y = v2.y, v2z = v2.z;
        float3 ret;
        ret.x = (v1y * v2z) - (v1z * v2y);
        ret.y = (v1z * v2x) - (v1x * v2z);
        ret.z = (v1x * v2y) - (v1y * v2x);
        return ret;
    }

    inline vfloat3 Cross(const vfloat3 v1, const vfloat3 v2) 
    {
        vfloat v1x = v1.x, v1y = v1.y, v1z = v1.z;
        vfloat v2x = v2.x, v2y = v2.y, v2z = v2.z;
        vfloat3 ret;
        store_all(ret.x, (v1y * v2z) - (v1z * v2y));
        store_all(ret.y, (v1z * v2x) - (v1x * v2z));
        store_all(ret.z, (v1x * v2y) - (v1y * v2x));
        return ret;
    }

    inline vfloat3 Cross(const vfloat3 v1, const float3 &v2)
    {
        vfloat v1x = v1.x, v1y = v1.y, v1z = v1.z;
        vfloat v2x = v2.x, v2y = v2.y, v2z = v2.z;
        vfloat3 ret;
        store_all(ret.x, (v1y * v2z) - (v1z * v2y));
        store_all(ret.y, (v1z * v2x) - (v1x * v2z));
        store_all(ret.z, (v1x * v2y) - (v1y * v2x));
        return ret;
    }

    inline vfloat3 Cross(const float3 &v1, const vfloat3 v2)
    {
        vfloat v1x = v1.x, v1y = v1.y, v1z = v1.z;
        vfloat v2x = v2.x, v2y = v2.y, v2z = v2.z;
        vfloat3 ret;
        store_all(ret.x, (v1y * v2z) - (v1z * v2y));
        store_all(ret.y, (v1z * v2x) - (v1x * v2z));
        store_all(ret.z, (v1x * v2y) - (v1y * v2x));
        return ret;
    }

    inline float Dot(const float3 &a, const float3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline vfloat Dot(const vfloat3 a, const vfloat3 b) 
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline vfloat Dot(const vfloat3 a, const float3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline vfloat Dot(const float3 &a, const vfloat3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline void vnormalize(vfloat3 &v) 
    {
        vfloat len2 = Dot(v, v);
        
        vfloat invlen = rsqrt_est2(max(len2, 1e-6f));

        store(v.x, v.x * invlen);
        store(v.y, v.y * invlen);
        store(v.z, v.z * invlen);
    }

    inline void vnormalize_all(vfloat3 &v) 
    {
        vfloat len2 = Dot(v, v);
        
        vfloat invlen = rsqrt_est2(max(len2, 1e-6f));

        store_all(v.x, v.x * invlen);
        store_all(v.y, v.y * invlen);
        store_all(v.z, v.z * invlen);
    }

    // varying
    struct Isect 
    {
        vfloat     t;
        vfloat3    p;
        vfloat3    n;
        vint       hit;
    };

    // uniform
    struct Sphere 
    {
        float3     center;
        float      radius;
    };

    // uniform
    struct Plane 
    {
        float3 p;
        float3 n;
    };

    // varying
    struct Ray 
    {
        vfloat3 org;
        vfloat3 dir;
    };
       
    inline void ray_plane_intersect(Isect &isect, const Ray &ray, const Plane &plane) 
    {
        vfloat d = -Dot(plane.p, plane.n);
        vfloat v = Dot(ray.dir, plane.n);

        if (spmd_all(abs(v) < 1.0e-17f))
            return;
        
        vfloat t = -(Dot(ray.org, plane.n) + d) / v;

        SPMD_SIF((t > 0.0f) && (t < isect.t)) 
        {
            store(isect.t, t);
            store(isect.hit, 1);
            
            vstore(isect.p, ray.org + ray.dir * t);
            
            store(isect.n.x, plane.n.x);
            store(isect.n.y, plane.n.y);
            store(isect.n.z, plane.n.z);
        }
        SPMD_SENDIF
    }

    inline void ray_sphere_intersect(Isect &isect, const Ray &ray, const Sphere &sphere) 
    {
        vfloat3 rs = ray.org - sphere.center;

        vfloat B = Dot(rs, ray.dir);
        vfloat C = Dot(rs, rs) - sphere.radius * sphere.radius;
        vfloat D = B * B - C;

        SPMD_SIF(D > 0.0f) 
        {
            vfloat t = -B - sqrt(D);

            SPMD_SIF((t > 0.0f) && (t < isect.t)) 
            {
                if (spmd_all())
                {
                    store_all(isect.t, t);
                    store_all(isect.hit, 1);
                    vstore_all(isect.p, ray.org + ray.dir * t);
                    vstore_all(isect.n, isect.p - sphere.center);
                    vnormalize_all(isect.n);
                }
                else
                {
                    store(isect.t, t);
                    store(isect.hit, 1);
                    vstore(isect.p, ray.org + ray.dir * t);
                    vstore(isect.n, isect.p - sphere.center);
                    vnormalize(isect.n);
                }
            }
            SPMD_SENDIF
        }
        SPMD_SENDIF
    }

    void orthoBasis(vfloat3 basis[3], vfloat3 n) 
    {
        vstore_all(basis[2], n);
        store_all(basis[1].x, 0.0f); 
        store_all(basis[1].y, 0.0f); 
        store_all(basis[1].z, 0.0f);

        vbool x_cond = (n.x < 0.6f) && (n.x > -0.6f);
        vbool y_cond = (n.y < 0.6f) && (n.y > -0.6f);
        vbool z_cond = (n.z < 0.6f) && (n.z > -0.6f);

        SPMD_SIF(x_cond) 
        {
            store(basis[1].x, 1.0f);
        }
        SPMD_SELSE(x_cond)
        {
            SPMD_SIF(y_cond)
            {
                store(basis[1].y, 1.0f);
            } 
            SPMD_SELSE(y_cond)
            {
                SPMD_SIF(z_cond) 
                {
                    store(basis[1].z, 1.0f);
                } 
                SPMD_SELSE(z_cond)
                {
                    store(basis[1].x, 1.0f);
                }
                SPMD_SENDIF
            }
            SPMD_SENDIF
        }
        SPMD_SENDIF

        vstore_all(basis[0], Cross(basis[1], basis[2]));
        vnormalize_all(basis[0]);

        vstore_all(basis[1], Cross(basis[2], basis[0]));
        vnormalize_all(basis[1]);
    }
   
    vfloat ambient_occlusion(Isect &isect, const Plane &plane, const Sphere spheres[3], rand_context &rngstate) 
    {
        const float eps = 0.0001f;
        vfloat3 n;
        vfloat3 basis[3];
        vfloat occlusion = 0.0f;

        vfloat3 p = isect.p + isect.n * vfloat3(eps);

        orthoBasis(basis, isect.n);
                
        static const int ntheta = NAO_SAMPLES;
        static const int nphi   = NAO_SAMPLES;

        for (int j = 0; j < ntheta; j++) 
        {
            for (int i = 0; i < nphi; i++) 
            {
                Isect occIsect;

                vfloat theta = sqrt(get_randf(rngstate, 0.0f, 1.0f));
                vfloat phi   = 2.0f * M_PI * get_randf(rngstate, 0.0f, 1.0f);

                vfloat x = cos_est(phi) * theta;
                vfloat y = sin_est(phi) * theta;
                vfloat z = sqrt(1.0f - theta * theta);

                // local . global
                vfloat rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
                vfloat ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
                vfloat rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

                Ray ray;
                vstore_all(ray.org, p);
                store_all(ray.dir.x, rx);
                store_all(ray.dir.y, ry);
                store_all(ray.dir.z, rz);

                store_all(occIsect.t, 1.0e+17f);
                store_all(occIsect.hit, 0);
                                
                for (int snum = 0; snum < 3; ++snum)
                    ray_sphere_intersect(occIsect, ray, spheres[snum]);

                ray_plane_intersect(occIsect, ray, plane);

                SPMD_SIF(occIsect.hit) 
                {
                    store(occlusion, occlusion + 1.0f);
                }
                SPMD_SENDIF
            }
        }

        return (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
    }

    /* Compute the image for the scanlines from [y0,y1), for an overall image
       of width w and height h.
     */
    void _call(int y0, int y1, int w, int h,  int nsubsamples, float image[]) 
    {
        static const Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
        
        static const Sphere spheres[3] = {
            { { -2.0f, 0.0f, -3.5f }, 0.5f },
            { { -0.5f, 0.0f, -3.0f }, 0.5f },
            { { 1.0f, 0.0f, -2.2f }, 0.5f } };

        rand_context rngstate;

        seed_rand(rngstate, vint(program_index) + (y0 << (vint(program_index) & 15)));

        const float invSamples = 1.f / nsubsamples;

        const int nsubsamples_sq = nsubsamples * nsubsamples;
        const int height = y1 - y0;
        const int total = height * w;

        const float w_div_h = (float)w / (float)h;
        const vfloat half_w = w / 2.0f;
        const vfloat half_h = h / 2.0f;
        const vfloat one_over_half_w = 1.0f / half_w;
        const vfloat one_over_half_h = 1.0f / half_h;;

        SPMD_FOREACH(iter, 0, total)
        {
            vint k = vint(iter) / w;
            vint x = vint(iter) - (k * w);
            vint y = y0 + k;

            for (int u = 0; u < nsubsamples; u++)
            {
                vfloat du = (float)u * invSamples;
                for (int v = 0; v < nsubsamples; v++)
                {
                    vfloat dv = (float)v * invSamples;

                    // Figure out x,y pixel in NDC
                    vfloat px =  ((vfloat)x + du - half_w) * one_over_half_w;
                    vfloat py = -((vfloat)y + dv - half_h) * one_over_half_h;

                    // Scale NDC based on width/height ratio, supporting non-square image output
                    store_all(px, px * w_div_h);

                    Ray ray;
                    vstore_all(ray.org, 0.0f);

                    // Poor man's perspective projection
                    store_all(ray.dir.x, px);
                    store_all(ray.dir.y, py);
                    store_all(ray.dir.z, -1.0f);
                    vnormalize_all(ray.dir);

                    Isect isect;
                    store_all(isect.t, 1.0e+17f);
                    vstore_all(isect.p, 0.0f);
                    vstore_all(isect.n, 0.0f);
                    store_all(isect.hit, 0);

                    for (int snum = 0; snum < 3; ++snum)
                        ray_sphere_intersect(isect, ray, spheres[snum]);

                    ray_plane_intersect(isect, ray, plane);

                    SPMD_SIF(isect.hit) 
                    {
                        vfloat ret = ambient_occlusion(isect, plane, spheres, rngstate) * invSamples * invSamples;
                        
                        vint offset = 3 * (y * w + x);
                        if (spmd_all())
                        {
                            store_all(offset[image], load_all((offset)[image]) + ret);
                            store_all((offset+1)[image], load_all((offset+1)[image]) + ret);
                            store_all((offset+2)[image], load_all((offset+2)[image]) + ret);
                        }
                        else
                        {
                            store(offset[image], load((offset)[image]) + ret);
                            store((offset+1)[image], load((offset+1)[image]) + ret);
                            store((offset+2)[image], load((offset+2)[image]) + ret);
                        }
                    }
                    SPMD_SENDIF;

                } // v
            } // u

        }
        SPMD_FOREACH_END(iter);

    }

}; // struct ao_bench_kernel

} // namespace

using namespace CPPSPMD_NAME(ao_bench_namespace);

void CPPSPMD_NAME(ao)(int w, int h, int nsubsamples, float image[])
{
	spmd_call< ao_bench_kernel >(0, h, w, h, nsubsamples, image);
}
