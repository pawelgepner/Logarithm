/* SPDX-License-Identifier: MIT
 * Exhaustive accuracy of the comparison functions (baselines/) over every
 * positive finite binary32 input, with the same measures as test_float.c.
 * Normal and subnormal inputs are reported separately because the 2023
 * algorithms do not handle subnormals.  CSV on stdout. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../baselines/aor_log2f.c"
#undef N
#undef T
#undef A
#undef OFF
#include "../baselines/fbsd_log2f.c"
#include "../baselines/legacy_algos.h"

typedef float (*fn_t)(float);
static const struct { const char *n; fn_t f; } F[] = {
    {"aor_log2f", aor_log2f}, {"fbsd_log2f", fbsd_log2f},
    {"legacy_log2_Mitchell", log2_Mitchell}, {"legacy_log2_P2_1", log2_P2_1},
    {"legacy_log2_P2_2", log2_P2_2}, {"legacy_log2_P2_int_2", log2_P2_int_2},
    {"legacy_log2_P1_16_eq", log2_P1_16_eq}, {"legacy_log2_P1_16_neq", log2_P1_16_neq},
    {"legacy_log2_P2_16", log2_P2_16}, {"legacy_log2_P2_16_neq", log2_P2_16_neq},
    {"legacy_log2_P3_16", log2_P3_16} };
#define NF ((int)(sizeof F / sizeof F[0]))

static double ulp32(double r)
{
    int e;
    r = fabs(r);
    if (r < 0x1p-126) return 0x1p-149;
    frexp(r, &e);
    return ldexp(1.0, e - 24);
}

int main(void)
{
    int j;
    printf("name,inputs,abs_err_min,abs_err_max,excess_over_half_ulp,max_ulp_err\n");
    for (j = 0; j < NF; j++) {
        int part;
        for (part = 0; part < 2; part++) {            /* 0: normal, 1: subnormal */
            int64_t b, lo = part ? 1 : 0x00800000, hi = part ? 0x00800000 : 0x7F800000;
            double amin = 0, amax = 0, exc = 0, ulp = 0;
            #pragma omp parallel for reduction(min:amin) reduction(max:amax,exc,ulp) schedule(static, 1 << 20)
            for (b = lo; b < hi; b++) {
                uint32_t u = (uint32_t)b;
                float x, yf;
                double r, y, d;
                memcpy(&x, &u, 4);
                yf = F[j].f(x);
                r = log2((double)x);
                y = (double)yf;
                d = y - r;
                if (d != d) d = INFINITY;
                if (d < amin) amin = d;
                if (d > amax) amax = d;
                if (fabs(d) - 0.5 * ulp32(y) > exc) exc = fabs(d) - 0.5 * ulp32(y);
                if (fabs(d) / ulp32(r) > ulp) ulp = fabs(d) / ulp32(r);
            }
            printf("%s,%s,%.6e,%.6e,%.6e,%.4g\n", F[j].n, part ? "subnormal" : "normal",
                   amin, amax, exc, ulp);
            fflush(stdout);
        }
    }
    return 0;
}
