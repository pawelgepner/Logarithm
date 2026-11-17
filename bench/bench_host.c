/* SPDX-License-Identifier: MIT
 * Host throughput benchmark (for reference only; logq targets cores WITHOUT
 * an FPU, where bench/emu.py is the relevant measurement).
 * Each function is called through a pointer on 4096 distinct positive normal
 * inputs, log-uniform in [2^-20, 2^20); the results are accumulated so that no
 * call can be removed.  The best of 31 repetitions is reported in ns per call.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "logq.h"
#include "../baselines/aor_log2f.c"
#undef N
#undef T
#undef A
#undef OFF
#include "../baselines/fbsd_log2f.c"
#include "../baselines/legacy_algos.h"

typedef float (*fn_t)(float);
#define W(V) static __attribute__((noinline)) float w_##V(float x) { return logq_##V##_f32(x); }
W(A1_16u) W(A2_16u) W(A2_16g) W(A3_16u) W(R2_16) W(R3_16)
static __attribute__((noinline)) float w_libm(float x) { return log2f(x); }
static __attribute__((noinline)) float w_aor(float x) { return aor_log2f(x); }
static __attribute__((noinline)) float w_fbsd(float x) { return fbsd_log2f(x); }
static __attribute__((noinline)) float w_P3_16(float x) { return log2_P3_16(x); }
static __attribute__((noinline)) float w_empty(float x) { return x; }

static const struct { const char *n; fn_t f; } F[] = {
    {"empty_call", w_empty}, {"logq_A1_16u", w_A1_16u}, {"logq_A2_16u", w_A2_16u},
    {"logq_A2_16g", w_A2_16g}, {"logq_A3_16u", w_A3_16u}, {"logq_R2_16", w_R2_16},
    {"logq_R3_16", w_R3_16}, {"legacy_log2_P3_16", w_P3_16}, {"aor_log2f", w_aor},
    {"fbsd_log2f", w_fbsd}, {"libm_log2f", w_libm} };

int main(void)
{
    enum { NX = 4096, REP = 200, TRIALS = 31 };
    static float x[NX];
    volatile float sink;
    unsigned i;
    srand(1);
    for (i = 0; i < NX; i++) {
        uint32_t b = ((uint32_t)(107 + rand() % 40) << 23) | ((uint32_t)rand() & 0x7FFFFF);
        memcpy(&x[i], &b, 4);
    }
    printf("function,ns_per_call\n");
    for (i = 0; i < sizeof F / sizeof F[0]; i++) {
        double best = 1e30;
        int t, r, k;
        for (t = 0; t < TRIALS; t++) {
            struct timespec a, b;
            float s = 0;
            clock_gettime(CLOCK_MONOTONIC, &a);
            for (r = 0; r < REP; r++)
                for (k = 0; k < NX; k++) s += F[i].f(x[k]);
            clock_gettime(CLOCK_MONOTONIC, &b);
            sink = s;
            {
                double ns = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / ((double)NX * REP);
                if (ns < best) best = ns;
            }
        }
        printf("%s,%.2f\n", F[i].n, best);
    }
    (void)sink;
    return 0;
}
