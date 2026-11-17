/* SPDX-License-Identifier: MIT
 * Exhaustive accuracy comparison with the fixed-point natural logarithm of
 * CMSIS-DSP (arm_scalar_log_q31, the kernel of arm_vlog_q31) over every Q31
 * input in (0, 1): all 2^31 - 1 values.  Both produce ln(x) in q5.26.
 * The logq versions use the integer entry point of a family-A variant
 * (see ln_* in bench/mcu_funcs.c).  CSV on stdout. */
#include <stdio.h>
#include <math.h>
#include "logq.h"
#include "../baselines/cmsis_log_q31.c"

#define LN2_Q31 0x58B90BFCu
#define LN(V) static int32_t ln_##V(uint32_t src) { \
    int32_t l2 = logq_##V##_q23_u32(src) - (31 << 23);      \
    return (int32_t)(((int64_t)l2 * LN2_Q31) >> 28); }
LN(A2_16u) LN(A2_32u) LN(A3_16u)

typedef int32_t (*fn_t)(uint32_t);
static int32_t cmsis(uint32_t s) { return (int32_t)arm_scalar_log_q31(s); }
static const struct { const char *n; fn_t f; } F[] = {
    {"cmsis_arm_vlog_q31", cmsis}, {"logq_A2_16u", ln_A2_16u}, {"logq_A2_32u", ln_A2_32u},
    {"logq_A3_16u", ln_A3_16u} };
#define NF 4

int main(void)
{
    int j;
    printf("function,err_min,err_max,rms_err,err_min_lsb_q26,err_max_lsb_q26\n");
    for (j = 0; j < NF; j++) {
        double l = 0, h = 0, s = 0;
        int64_t v;
        #pragma omp parallel for reduction(min:l) reduction(max:h) reduction(+:s) schedule(static, 1 << 20)
        for (v = 1; v < (1LL << 31); v++) {
            double ref = log((double)v) - 31.0 * 0.69314718055994530942;
            double e = F[j].f((uint32_t)v) / 67108864.0 - ref;
            if (e < l) l = e;
            if (e > h) h = e;
            s += e * e;
        }
        printf("%s,%.6e,%.6e,%.6e,%.2f,%.2f\n", F[j].n, l, h, sqrt(s / 2147483647.0),
               l * 67108864.0, h * 67108864.0);
        fflush(stdout);
    }
    return 0;
}
