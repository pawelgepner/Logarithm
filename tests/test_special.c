/* SPDX-License-Identifier: MIT
 * IEEE 754 special values, exact cases and the integer-input entry point. */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "logq.h"

static int fails;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

#define SPECIAL(V) do {                                                          \
    CHECK(logq_##V##_f32bits(0x00000000u) == 0xFF800000u);   /* +0   -> -inf */  \
    CHECK(logq_##V##_f32bits(0x80000000u) == 0xFF800000u);   /* -0   -> -inf */  \
    CHECK(logq_##V##_f32bits(0x7F800000u) == 0x7F800000u);   /* +inf -> +inf */  \
    CHECK(isnan(logq_##V##_f32(-1.0f)));                                         \
    CHECK(isnan(logq_##V##_f32(-INFINITY)));                                     \
    CHECK(isnan(logq_##V##_f32(NAN)));                                           \
    CHECK(isnan(logq_bits_f32(logq_##V##_f32bits(0x7F800001u))));  /* sNaN */    \
    CHECK(fabs(logq_##V##_f32(FLT_TRUE_MIN) + 149.0) < 0.05);                    \
    CHECK(fabs(logq_##V##_f32(FLT_MAX) - 128.0) < 0.05);                         \
    CHECK(fabsf(logq_##V##_f32(nextafterf(2.0f, 0.0f)) - 1.0f) < 0.05f); /* carry */ \
    CHECK(logq_##V##_f32(0x1.fffffep126f) > 126.9f);                             \
} while (0);

#define EXACT_R(V) do {                                                          \
    int e;                                                                       \
    for (e = -149; e <= 127; e++)                                                \
        CHECK(logq_##V##_f32(ldexpf(1.0f, e)) == (float)e);                      \
} while (0);

int main(void)
{
    LOGQ_FOR_EACH_A(SPECIAL)
    LOGQ_FOR_EACH_R(SPECIAL)
    LOGQ_FOR_EACH_R(EXACT_R)          /* family R is exact at every power of two */

    /* integer input: exhaustive over all 32-bit integers for one variant */
    {
        double emin = 0, emax = 0; uint32_t v = 1;
        do {
            double e = logq_A2_16u_q23_u32(v) / 8388608.0 - log2((double)v);
            if (e < emin) emin = e;
            if (e > emax) emax = e;
        } while (++v != 0);
        /* bound: kernel error + truncation of the bits below the 24 leading ones */
        printf("A2_16u_q23_u32 over all v in [1, 2^32): error in [%.3e, %.3e]\n", emin, emax);
        CHECK(emin > -(3.41e-6 + 1.72e-7 + 6e-8) && emax < 3.41e-6 + 6e-8);
    }
    printf(fails ? "%d FAILURES\n" : "all special-value tests passed\n", fails);
    return fails != 0;
}
