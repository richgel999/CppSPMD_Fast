// Do not include this file directly.
//
// The functions in this file (and ONLY in this one file) were ported from DirectXMath's 
// XMVectorSin/XMVectorCos() functions:
// https://github.com/microsoft/DirectXMath
//
//	The MIT License(MIT)
//
//	Copyright(c) 2011 - 2020 Microsoft Corp
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy of this
//	softwareand associated documentation files(the "Software"), to deal in the Software
//	without restriction, including without limitation the rights to use, copy, modify,
//	merge, publish, distribute, sublicense, and /or sell copies of the Software, and to
//	permit persons to whom the Software is furnished to do so, subject to the following
//	conditions :
//
//	The above copyright noticeand this permission notice shall be included in all copies
//	or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//	PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//	HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//	CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
//	OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Range reduction of a to Pi <= a < Pi
inline vfloat spmd_kernel::mod_angles(vfloat a)
{
	const float fOneOverTwoPi = 0.159154943f;
	const float fTwoPi = 6.283185307f;
			
	vfloat r = round_nearest(a * fOneOverTwoPi);
	
	vfloat result = vfnma(fTwoPi, r, a);
	
	return result;
}

/*
	clang 9.0.0 for win /fp:precise release
	Total near-zero: 144, output above near-zero tresh: 6
	Total near-zero avg: 0.0000067474567516 max: 0.0000133514404297
	Total near-zero sign diffs: 4
	Total passed near-zero check: 16777072
	Total sign diffs: 4
	max abs err: 0.0000016523068072
	max rel err: 0.0663381186093683
	avg abs err: 0.0000002274915396
	avg rel err: 0.0000024307621040
*/

// Ported from XMVectorSin()'s SSE variant
inline vfloat spmd_kernel::sin_est(vfloat a)
{
	const float fPi = 3.141592654f, fHalfPi = 1.570796327f; // fOneOverPi = 0.318309886f
	const float fSin0_X = -0.16666667f, fSin0_Y = +0.0083333310f, fSin0_Z = -0.00019840874f, fSin0_W = +2.7525562e-06f;
	const float fSin1_X = -2.3889859e-08f; // fSin1_Y = -0.16665852f, fSin1_Z = +0.0083139502f, fSin1_W = -0.00018524670f;

	vfloat x = mod_angles(a);
	vint sign = cast_vfloat_to_vint(x) & 0x80000000;
	vfloat c = cast_vint_to_vfloat(sign | cast_vfloat_to_vint(fPi));
	vfloat absx = cast_vint_to_vfloat(andnot(sign, cast_vfloat_to_vint(x)));
	vfloat rflx = c - x;

	store_all(x, spmd_ternaryf(absx <= fHalfPi, x, rflx));

	vfloat x2 = x * x;

	vfloat result = vfma(fSin1_X, x2, fSin0_W);

	store_all(result, vfma(result, x2, fSin0_Z));
	store_all(result, vfma(result, x2, fSin0_Y));
	store_all(result, vfma(result, x2, fSin0_X));

	store_all(result, vfma(result, x2, 1.0f));

	store_all(result, result * x);

	return result;
}

/*
	clang 9.0.0 for win /fp:precise release
	Total near-zero: 144, output above near-zero tresh: 8
	Total near-zero avg: 0.0000066657861074 max: 0.0000126957893372
	Total near-zero sign diffs: 2
	Total passed near-zero check: 16777072
	Total sign diffs: 2
	max abs err: 0.0000017509699197
	max rel err: 0.0815433491993518
	avg abs err: 0.0000002295096017
	avg rel err: 0.0000025697131638

	std::cosf():
	Near-zero thresholds: .0000125f 1e-6f
	Total near-zero: 144, output above near-zero tresh: 0
	Total near-zero avg: 0.0000066998935358 max: 0.0000127593821162
	Total near-zero sign diffs: 8
	Total passed near-zero check: 16777072
	Total sign diffs: 8
	max abs err: 0.0000009542060670
	max rel err: 0.0628415221432127
	avg abs err: 0.0000001778108189
	avg rel err: 0.0000019515542715
*/

// Ported from XMVectorCos()'s SSE variant
inline vfloat spmd_kernel::cos_est(vfloat a)
{
	const float fPi = 3.141592654f, fHalfPi = 1.570796327f; // fOneOverPi = 0.318309886f

	const float fCos0_X = -0.5f, fCos0_Y = +0.041666638f, fCos0_Z = -0.0013888378f, fCos0_W = +2.4760495e-05f;
	const float fCos1_X = -2.6051615e-07f;// , fCos1_Y = -0.49992746f /*Est1*/, fCos1_Z = +0.041493919f /*Est2*/, fCos1_W = -0.0012712436f /*Est3*/;

	vfloat x = mod_angles(a);
	
	vint signi = cast_vfloat_to_vint(x) & 0x80000000;
		
	vfloat c = cast_vint_to_vfloat(signi | cast_vfloat_to_vint(fPi));

	vfloat absx = cast_vint_to_vfloat(andnot(signi, cast_vfloat_to_vint(x)));

	vfloat rflx = c - x;

	vbool comp = (absx <= fHalfPi);
		
	store_all(x, spmd_ternaryf(comp, x, rflx));

	vfloat sign = spmd_ternaryf(comp, 1.0f, -1.0f);

	vfloat x2 = x * x;

	vfloat result = vfma(fCos1_X, x2, fCos0_W);

	store_all(result, vfma(result, x2, fCos0_Z));
	store_all(result, vfma(result, x2, fCos0_Y));
	store_all(result, vfma(result, x2, fCos0_X));

	store_all(result, vfma(result, x2, 1.0f));

	store_all(result, result * sign);

	return result;
}

