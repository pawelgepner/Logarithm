/* Arm Optimized Routines log2f -- the binary32 log2 used by musl and glibc.
 * Vendored from https://github.com/ARM-software/optimized-routines, math/log2f.c
 * and math/log2f_data.c (commit 20859d3).  Only the include of math_config.h and
 * the error-reporting helpers were replaced by the minimal definitions below;
 * the computation is unchanged.
 *
 * Copyright (c) 2017-2026, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */
#include <stdint.h>
#define LOG2F_TABLE_BITS 4
#define LOG2F_POLY_ORDER 4
#define WANT_ROUNDING 1
#define unlikely(x) __builtin_expect(!!(x), 0)
static inline uint32_t aor_asuint(float f) { union { float f; uint32_t i; } u = { f }; return u.i; }
static inline float aor_asfloat(uint32_t i) { union { uint32_t i; float f; } u = { i }; return u.f; }
#define asuint aor_asuint
#define asfloat aor_asfloat
#define eval_as_float(x) ((float)(x))
#define __math_divzerof(s) aor_asfloat(0xFF800000u)
#define __math_invalidf(x) aor_asfloat(0x7FC00000u)
static const struct log2f_data {
  struct { double invc, logc; } tab[1 << LOG2F_TABLE_BITS];
  double poly[LOG2F_POLY_ORDER];
} __log2f_data = {
  .tab = {
  { 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2 },
  { 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2 },
  { 0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2 },
  { 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2 },
  { 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2 },
  { 0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3 },
  { 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3 },
  { 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4 },
  { 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5 },
  { 0x1p+0, 0x0p+0 },
  { 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4 },
  { 0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3 },
  { 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3 },
  { 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2 },
  { 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2 },
  { 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2 },
  },
  .poly = {
  -0x1.712b6f70a7e4dp-2, 0x1.ecabf496832ep-2, -0x1.715479ffae3dep-1,
  0x1.715475f35c8b8p0,
  }
};

#define N (1 << LOG2F_TABLE_BITS)
#define T __log2f_data.tab
#define A __log2f_data.poly
#define OFF 0x3f330000

static float
aor_log2f (float x)
{
  double z, r, r2, p, y, y0, invc, logc;
  uint32_t ix, iz, top, tmp;
  int k, i;

  ix = asuint (x);
#if WANT_ROUNDING
  /* Fix sign of zero with downward rounding when x==1.  */
  if (unlikely (ix == 0x3f800000))
    return 0;
#endif
  if (unlikely (ix - 0x00800000 >= 0x7f800000 - 0x00800000))
    {
      /* x < 0x1p-126 or inf or nan.  */
      if (ix * 2 == 0)
	return __math_divzerof (1);
      if (ix == 0x7f800000) /* log2(inf) == inf.  */
	return x;
      if ((ix & 0x80000000) || ix * 2 >= 0xff000000)
	return __math_invalidf (x);
      /* x is subnormal, normalize it.  */
      ix = asuint (x * 0x1p23f);
      ix -= 23 << 23;
    }

  /* x = 2^k z; where z is in range [OFF,2*OFF] and exact.
     The range is split into N subintervals.
     The ith subinterval contains z and c is near its center.  */
  tmp = ix - OFF;
  i = (tmp >> (23 - LOG2F_TABLE_BITS)) % N;
  top = tmp & 0xff800000;
  iz = ix - top;
  k = (int32_t) tmp >> 23; /* Arithmetic shift.  */
  invc = T[i].invc;
  logc = T[i].logc;
  z = asfloat (iz);

  /* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k.  */
  r = z * invc - 1;
  y0 = logc + (double) k;

  /* Pipelined polynomial evaluation to approximate log1p(r)/ln2.  */
  r2 = r * r;
  y = A[1] * r + A[2];
  y = A[0] * r2 + y;
  p = A[3] * r + y0;
  y = y * r2 + p;
  return eval_as_float (y);
}
