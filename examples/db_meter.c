/* SPDX-License-Identifier: MIT
 * Example: sound level in dBFS of 16-bit PCM frames on a core without an FPU.
 *
 *   level = 10 log10(mean(s^2) / 32768^2) = (10 log10 2) * (log2(sum s^2) - log2(N) - 30)
 *
 * Everything is integer: the energy of a frame is a 64-bit sum, its logarithm
 * comes from logq_A2_16u_q23_u32 applied to the 32 leading bits, and the result
 * is a Q8.23 number of decibels.  With A2_16u the logarithm contributes at most
 * 3.6e-6 to the error of log2, i.e. 1.1e-5 dB.
 * The program checks the integer result against double-precision arithmetic.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "logq.h"

#define N 256                                /* samples per frame (power of two) */
#define LOG2_N 8
#define TEN_LOG10_2_Q28 808071242            /* 10 log10(2) = 3.0103 in Q28 */

static int32_t frame_dbfs_q23(const int16_t *s)
{
    uint64_t e = 0;
    int i, sh = 0;
    for (i = 0; i < N; i++) e += (uint64_t)((int32_t)s[i] * s[i]);
    if (e == 0) return INT32_MIN;                          /* digital silence */
    while (e >> 32) { e >>= 1; sh++; }                     /* keep 32 leading bits */
    {
        int32_t l2 = logq_A2_16u_q23_u32((uint32_t)e) + ((sh - LOG2_N - 30) << 23);
        return (int32_t)(((int64_t)l2 * TEN_LOG10_2_Q28) >> 28);   /* dB, Q8.23 */
    }
}

int main(void)
{
    static int16_t s[N];
    double worst = 0;
    int f, i;
    srand(7);
    for (f = 0; f < 10000; f++) {
        double amp = pow(10.0, -(double)(rand() % 9000) / 100.0 / 20.0) * 32767.0, e = 0;
        double ref, got;
        for (i = 0; i < N; i++) {
            s[i] = (int16_t)lrint(amp * sin(0.05 * i * (1 + f % 7)) + (rand() % 3 - 1));
            e += (double)s[i] * s[i];
        }
        if (e == 0) continue;
        ref = 10.0 * log10(e / N / (32768.0 * 32768.0));
        got = frame_dbfs_q23(s) / 8388608.0;
        if (fabs(got - ref) > worst) worst = fabs(got - ref);
        if (f < 5) printf("frame %d: %9.4f dBFS (double: %9.4f)\n", f, got, ref);
    }
    printf("10000 frames, largest difference from double precision: %.2e dB\n", worst);
    return worst > 1e-4;
}
