/* SPDX-License-Identifier: MIT
 * Exhaustive test of the binary32 entry points over every positive finite
 * binary32 number: 2,130,706,432 normal and 8,388,607 subnormal inputs.
 * The reference is log2() in binary64.  For every variant the program reports
 *   - the extreme absolute errors of the binary32 result,
 *   - excess = max(|error| - ulp(y)/2), y the returned binary32 value: the
 *     error left after removing the rounding of the result to binary32.  For family A it must not
 *     exceed the kernel error bound; it shows that the conversion adds nothing,
 *   - the largest relative error and the largest error in ulps (meaningful for
 *     family R; family A has unbounded relative error at x = 1).  CSV on stdout.
 * Build with -fopenmp to use all cores (about 10 minutes on two cores). */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "logq.h"

typedef uint32_t (*fn_t)(uint32_t);
#define ENT_A(V) { #V, logq_##V##_f32bits },
#define ENT_R(V) { #V, logq_##V##_f32bits },
static const struct { const char *name; fn_t f; } FN[] = { LOGQ_FOR_EACH_A(ENT_A) LOGQ_FOR_EACH_R(ENT_R) };
#define NF ((int)(sizeof FN / sizeof FN[0]))

typedef struct { double amin, amax, exc, rel, ulp; uint32_t x_ulp; } stat_t;

static double ulp32(double r)                 /* spacing of binary32 at |r| */
{
    int e;
    r = fabs(r);
    if (r < 0x1p-126) return 0x1p-149;
    frexp(r, &e);
    return ldexp(1.0, e - 24);
}

int main(int argc, char **argv)
{
    static stat_t tot[sizeof FN / sizeof FN[0]];
    int only = -1, i;
    if (argc > 1) for (i = 0; i < NF; i++) if (!strcmp(argv[1], FN[i].name)) only = i;
    memset(tot, 0, sizeof tot);

    #pragma omp parallel
    {
        stat_t loc[sizeof FN / sizeof FN[0]];
        int j;
        int64_t b;
        memset(loc, 0, sizeof loc);
        #pragma omp for schedule(dynamic, 1 << 20) nowait
        for (b = 1; b < 0x7F800000; b++) {
            double r = log2((double)logq_bits_f32((uint32_t)b));
            double u = ulp32(r);
            for (j = 0; j < NF; j++) {
                double y, d, a;
                if (only >= 0 && j != only) continue;
                y = (double)logq_bits_f32(FN[j].f((uint32_t)b));
                d = y - r; a = fabs(d);
                if (d < loc[j].amin) loc[j].amin = d;
                if (d > loc[j].amax) loc[j].amax = d;
                if (a - 0.5 * ulp32(y) > loc[j].exc) loc[j].exc = a - 0.5 * ulp32(y);
                if (r != 0.0 && a > loc[j].rel * fabs(r)) loc[j].rel = a / fabs(r);
                else if (r == 0.0 && a != 0.0) loc[j].rel = INFINITY;
                if (a > loc[j].ulp * u) { loc[j].ulp = a / u; loc[j].x_ulp = (uint32_t)b; }
            }
        }
        #pragma omp critical
        for (j = 0; j < NF; j++) {
            if (loc[j].amin < tot[j].amin) tot[j].amin = loc[j].amin;
            if (loc[j].amax > tot[j].amax) tot[j].amax = loc[j].amax;
            if (loc[j].exc > tot[j].exc) tot[j].exc = loc[j].exc;
            if (loc[j].rel > tot[j].rel) tot[j].rel = loc[j].rel;
            if (loc[j].ulp > tot[j].ulp) { tot[j].ulp = loc[j].ulp; tot[j].x_ulp = loc[j].x_ulp; }
        }
    }
    printf("name,abs_err_min,abs_err_max,excess_over_half_ulp,max_rel_err,max_ulp_err,x_at_max_ulp\n");
    for (i = 0; i < NF; i++) {
        if (only >= 0 && i != only) continue;
        printf("%s,%.6e,%.6e,%.6e,%.6e,%.4g,0x%08X\n", FN[i].name, tot[i].amin, tot[i].amax,
               tot[i].exc, tot[i].rel, tot[i].ulp, tot[i].x_ulp);
    }
    return 0;
}
