#pragma once

#ifndef CPPSPMD_TYPES
#define CPPSPMD_TYPES

using exec_mask = CPPSPMD::exec_mask;

#if CPPSPMD_INT16
using vint16 = CPPSPMD::vint16;
using int16_lref = CPPSPMD::int16_lref;
using cint16_vref = CPPSPMD::cint16_vref;
using int16_vref = CPPSPMD::int16_vref;
using lint16 = CPPSPMD::lint16;
using vint16_vref = CPPSPMD::vint16_vref;
#else
using vint = CPPSPMD::vint;
using int_lref = CPPSPMD::int_lref;
using cint_vref = CPPSPMD::cint_vref;
using int_vref = CPPSPMD::int_vref;
using lint = CPPSPMD::lint;
using vint_vref = CPPSPMD::vint_vref;
#endif

using vbool = CPPSPMD::vbool;
using vfloat = CPPSPMD::vfloat;
using float_lref = CPPSPMD::float_lref;
using float_vref = CPPSPMD::float_vref;
using vfloat_vref = CPPSPMD::vfloat_vref;

#endif // CPPSPMD_TYPES
