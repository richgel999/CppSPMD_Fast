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
namespace CPPSPMD_NAME(rt_kernel_namespace)
{
    
struct rt : spmd_kernel
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

    // varying
    struct Ray 
    {
        vfloat3 origin, dir, invDir;
        int dirIsNeg[3];
        vfloat mint, maxt;
        vint hitId;
    };
       
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

    void generateRay(const float raster2camera[4][4],
        const float camera2world[4][4],
        vfloat x, vfloat y, Ray& ray) 
    {
        store_all(ray.mint, 0.0f);
        store_all(ray.maxt, 1e30f);

        store_all(ray.hitId, 0);

        // transform raster coordinate (x, y, 0) to camera space
        vfloat camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
        vfloat camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
        vfloat camz = raster2camera[2][3];
        vfloat camw = raster2camera[3][3];
        store_all(camx, camx / camw);
        store_all(camy, camy / camw);
        store_all(camz, camz / camw);

        store_all(ray.dir.x, camera2world[0][0] * camx + camera2world[0][1] * camy +
            camera2world[0][2] * camz);

        store_all(ray.dir.y, camera2world[1][0] * camx + camera2world[1][1] * camy +
            camera2world[1][2] * camz);

        store_all(ray.dir.z, camera2world[2][0] * camx + camera2world[2][1] * camy +
            camera2world[2][2] * camz);

        store_all(ray.origin.x, camera2world[0][3] / camera2world[3][3]);
        store_all(ray.origin.y, camera2world[1][3] / camera2world[3][3]);
        store_all(ray.origin.z, camera2world[2][3] / camera2world[3][3]);

        vstore_all(ray.invDir, vfloat3(1.0f) / ray.dir);

        ray.dirIsNeg[0] = spmd_any(ray.invDir.x < 0) ? 1 : 0;
        ray.dirIsNeg[1] = spmd_any(ray.invDir.y < 0) ? 1 : 0;
        ray.dirIsNeg[2] = spmd_any(ray.invDir.z < 0) ? 1 : 0;
    }

    vbool BBoxIntersect(const float bounds[2][3], const Ray& ray) 
    {
        float3 bounds0 = { bounds[0][0], bounds[0][1], bounds[0][2] };
        float3 bounds1 = { bounds[1][0], bounds[1][1], bounds[1][2] };
        vfloat t0 = ray.mint, t1 = ray.maxt;

        // Check all three axis-aligned slabs.  Don't try to early out; it's
        // not worth the trouble
        vfloat3 tNear = (vfloat3(bounds0) - ray.origin) * ray.invDir;
        vfloat3 tFar = (vfloat3(bounds1) - ray.origin) * ray.invDir;
        
        SPMD_SIF(tNear.x > tFar.x) 
        {
            swap(tNear.x, tFar.x);
        }
        SPMD_SENDIF

        store_all(t0, max(tNear.x, t0));
        store_all(t1, min(tFar.x, t1));

        SPMD_SIF(tNear.y > tFar.y) 
        {
            swap(tNear.y, tFar.y);
        }
        SPMD_SENDIF

        store_all(t0, max(tNear.y, t0));
        store_all(t1, min(tFar.y, t1));

        SPMD_SIF(tNear.z > tFar.z) 
        {
            swap(tNear.z, tFar.z);
        }
        SPMD_SENDIF

        store_all(t0, max(tNear.z, t0));
        store_all(t1, min(tFar.z, t1));

        return (t0 <= t1);
    }

    vbool TriIntersect(const Triangle& tri, Ray& ray) 
    {
        float3 p0 = { tri.p[0][0], tri.p[0][1], tri.p[0][2] };
        float3 p1 = { tri.p[1][0], tri.p[1][1], tri.p[1][2] };
        float3 p2 = { tri.p[2][0], tri.p[2][1], tri.p[2][2] };
        float3 e1 = p1 - p0;
        float3 e2 = p2 - p0;

        vfloat3 s1 = Cross(ray.dir, e2);
        vfloat divisor = Dot(s1, e1);
        
        vbool hit = true;

        SPMD_SIF(divisor == 0.0f)
        {
            store(hit, false);
        }
        SPMD_SENDIF

        vfloat invDivisor = safe_div(vfloat(1.0f), divisor);

        // Compute first barycentric coordinate
        vfloat3 d = ray.origin - p0;
        vfloat b1 = Dot(d, s1) * invDivisor;
        SPMD_SIF((b1 < 0.0f) || (b1 > 1.0f))
        {
            store(hit, false);
        }
        SPMD_SENDIF

        // Compute second barycentric coordinate
        vfloat3 s2 = Cross(d, e1);
        vfloat b2 = Dot(ray.dir, s2) * invDivisor;
        SPMD_SIF((b2 < 0.0f) || (b1 + b2 > 1.0f))
        {
            store(hit, false);
        }
        SPMD_SENDIF

        // Compute _t_ to intersection point
        vfloat t = Dot(e2, s2) * invDivisor;
        SPMD_SIF((t < ray.mint) || (t > ray.maxt))
        {
            store(hit, false);
        }
        SPMD_SENDIF

        SPMD_SIF(hit) 
        {
            store(ray.maxt, t);
            store(ray.hitId, tri.id);
        }
        SPMD_SENDIF

        return hit;
    }

    vbool BVHIntersect(const LinearBVHNode nodes[], const Triangle tris[], Ray& r) 
    {
        Ray ray = r;
        vbool hit = false;

        // Follow ray through BVH nodes to find primitive intersections
        int todoOffset = 0, nodeNum = 0, todo[64];

        while(true) 
        {
            // Check ray against BVH node
            LinearBVHNode node = nodes[nodeNum];

            if (spmd_any(BBoxIntersect(node.bounds, ray))) 
            {
                unsigned int nPrimitives = node.nPrimitives;

                if (nPrimitives > 0) 
                {
                    // Intersect ray with primitives in leaf BVH node
                    unsigned int primitivesOffset = node.offset;

                    for (unsigned int i = 0; i < nPrimitives; ++i) 
                    {
                        SPMD_SIF(TriIntersect(tris[primitivesOffset + i], ray))
                        {
                            store(hit, true);
                        }
                        SPMD_SENDIF
                    }

                    if (todoOffset == 0)
                        break;
                    nodeNum = todo[--todoOffset];
                }
                else 
                {
                    // Put far BVH node on _todo_ stack, advance to near node
                    if (r.dirIsNeg[node.splitAxis]) 
                    {
                        todo[todoOffset++] = nodeNum + 1;
                        nodeNum = node.offset;
                    }
                    else 
                    {
                        todo[todoOffset++] = node.offset;
                        nodeNum = nodeNum + 1;
                    }
                }
            }
            else 
            {
                if (todoOffset == 0)
                    break;
                nodeNum = todo[--todoOffset];
            }
        }
        
        store_all(r.maxt, ray.maxt);
        store_all(r.hitId, ray.hitId);

        return hit;
    }

    void _call(
        int x0, int x1,
        int y0, int y1,
        int width, int height,
        const float raster2camera[4][4],
        const float camera2world[4][4],
        float image[], int id[],
        const LinearBVHNode nodes[],
        const Triangle triangles[]) 
    {
        (void)x1;
        (void)height;

        int total = (y1 - y0) << TILE_SHIFT;

        spmd_foreach(0, total, [&](const lint& o, int p)
            {
                (void)p;

                vint oi = (vint)o;
                vint x_ofs = (oi & TILE_AND_MASK);
                vint x = x0 + x_ofs;

                SPMD_IF(x < width)
                {
                    vint y_ofs = VINT_SHIFT_RIGHT(oi, TILE_SHIFT);
                    vint y = y0 + y_ofs;

                    Ray ray;
                    generateRay(raster2camera, camera2world, (vfloat)x, (vfloat)y, ray);
                    BVHIntersect(nodes, triangles, ray);

                    vint offset = y * width + x;
                    VASSERT((offset >= 0) && (offset < width* height));
                                        
                    store(offset[image], ray.maxt);
                    store(offset[id], ray.hitId);
                }
                SPMD_END_IF
            }
        );
    }
};

} // namespace

using namespace CPPSPMD_NAME(rt_kernel_namespace);

void CPPSPMD_NAME(raytrace_tile)(
    int x0, int x1,
    int y0, int y1,
    int width, int height,
    const float raster2camera[4][4],
    const float camera2world[4][4],
    float image[], int id[],
    const LinearBVHNode nodes[],
    const Triangle triangles[])
{
    spmd_call< rt >(x0, x1, y0, y1, width, height, raster2camera, camera2world, image, id, nodes, triangles);
}
