/* SPDX-License-Identifier: MIT
 * Exhaustive test of every family-A kernel over all 2^23 significands.
 * Prints, per variant: min/max absolute error of log2(1+m), whether the kernel
 * is monotonic across the whole binary32 range (including the step from one
 * binade to the next), and whether it is exact at m = 0 (powers of two).
 * Output is CSV on stdout. */
#include <stdio.h>
#include <math.h>
#include "logq.h"

#define TEST(V) do {                                                             \
    double emin = 0, emax = 0; int mono = 1; int32_t prev = -1; uint32_t k;     \
    for (k = 0; k < (1u << 23); k++) {                                           \
        int32_t f = logq_##V##_kernel(k);                                        \
        double e = (double)f / 8388608.0 - log2(1.0 + (double)k / 8388608.0);    \
        if (e < emin) emin = e;                                                  \
        if (e > emax) emax = e;                                                  \
        if (f < prev) mono = 0;                                                  \
        prev = f;                                                                \
    }                                                                            \
    if (prev > (1 << 23) + logq_##V##_kernel(0)) mono = 0; /* next binade */    \
    printf("%s,%.6e,%.6e,%d,%d\n", #V, emin, emax, mono, logq_##V##_kernel(0) == 0); \
} while (0);

int main(void)
{
    printf("name,err_min,err_max,monotonic,exact_at_pow2\n");
    LOGQ_FOR_EACH_A(TEST)
    return 0;
}
