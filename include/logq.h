/* SPDX-License-Identifier: MIT
 * logq -- integer-only binary logarithms with verified error bounds.
 *
 * This umbrella header exposes every catalogued variant:
 *
 *   family A(n, p, s)   logq_A<n>_<p><u|g>_*    absolute error, Q8.23 result
 *   family R(n, p)      logq_R<n>_<p>_*         relative error, binary32 result
 *
 * For each family-A variant V:
 *   int32_t  logq_V_kernel(uint32_t k)      log2(1 + k/2^23), Q23, k in [0, 2^23)
 *   int32_t  logq_V_q23(uint32_t bits)      log2 of a positive normal binary32, Q8.23
 *   int32_t  logq_V_q23_u32(uint32_t v)     log2 of an integer v >= 1, Q8.23
 *   uint32_t logq_V_f32bits(uint32_t bits)  full IEEE 754 binary32 domain, bit patterns
 *   float    logq_V_f32(float x)            same, as float
 *
 * For each family-R variant V:
 *   uint32_t logq_V_f32bits(uint32_t bits)
 *   float    logq_V_f32(float x)
 *
 * A product normally needs one variant only: generate a self-contained header
 * for it with `tools/logqgen.py -n <degree> -p <segments> [--seg g] [--rel]`.
 */
#ifndef LOGQ_H
#define LOGQ_H
#include "logq_tables.h"
#endif
