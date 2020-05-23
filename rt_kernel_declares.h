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

#ifndef RT_KERNEL_H
#define RT_KERNEL_H

const int TILE_SHIFT = 4;
const int TILE_SIZE = 1 << TILE_SHIFT;
const int TILE_AND_MASK = TILE_SIZE - 1;

// uniform
struct Triangle
{
    float p[3][4];
    int id;
    int pad[3];
};

// uniform
struct LinearBVHNode
{
    float bounds[2][3];
    uint32_t offset;     // num primitives for leaf, second child for interior

    uint8_t nPrimitives;
    uint8_t splitAxis;
    uint16_t pad;
};

#endif // RT_KERNEL_H

void CPPSPMD_NAME(raytrace_tile)(
    int x0, int x1,
    int y0, int y1,
    int width, int height,
    const float raster2camera[4][4],
    const float camera2world[4][4],
    float image[], int id[],
    const LinearBVHNode nodes[],
    const Triangle triangles[]);
