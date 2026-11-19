/* SPDX-License-Identifier: MIT
 * Cycle-count benchmark of logq and comparison functions on a real MCU.
 * Platform-independent core: the platform provides bench_cycles(),
 * bench_irq_off(), bench_irq_on() and printf (see main.c and host_main.c).
 *
 * For each function: mean / min / max cycles per call over the 1000 inputs of
 * the emulation (bench/emu.py, seed 2026), minus the cost of an empty call,
 * and an FNV-1a checksum of all 1000 results compared with the value computed
 * on a PC (expected.h).  Output: lines starting with "CSV," on the console.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "bench_core.h"
#include "logq.h"
#include "aor_log2f.c"
#undef N
#undef T
#undef A
#undef OFF
#include "fbsd_log2f.c"
#include "cmsis_log_q31.c"
#include "legacy_algos.h"
#include "inputs.h"
#include "expected.h"

#define NOINL __attribute__((noinline))
static inline float u2f(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

/* ---- functions under test: uint32_t f(uint32_t), bit patterns in and out ---- */
NOINL static uint32_t empty_call(uint32_t b) { return b; }

/* logq, binary32 in -> binary32 out, full IEEE 754 domain, integer only */
NOINL static uint32_t f_A1_16u(uint32_t b) { return logq_A1_16u_f32bits(b); }
NOINL static uint32_t f_A2_16u(uint32_t b) { return logq_A2_16u_f32bits(b); }
NOINL static uint32_t f_A2_16g(uint32_t b) { return logq_A2_16g_f32bits(b); }
NOINL static uint32_t f_A3_16u(uint32_t b) { return logq_A3_16u_f32bits(b); }
NOINL static uint32_t f_R3_16(uint32_t b)  { return logq_R3_16_f32bits(b); }
NOINL static uint32_t f_R3_32(uint32_t b)  { return logq_R3_32_f32bits(b); }

/* logq A3_16u: Q8.23 result, then two ways of turning it into a float */
NOINL static uint32_t q_A3_16u(uint32_t b) { return (uint32_t)logq_A3_16u_q23(b); }
NOINL static uint32_t q_A3_16u_softfloat(uint32_t b)
{ return f2u((float)logq_A3_16u_q23(b) / 8388608.0f); }       /* via soft-float */
NOINL static uint32_t q_A3_16u_intfloat(uint32_t b)
{ return logq_q23_to_f32bits(logq_A3_16u_q23(b)); }          /* integer only   */

/* logq A2_16u: kernel alone, Q8.23 from binary32, Q8.23 from an integer */
NOINL static uint32_t k_A2_16u(uint32_t k) { return (uint32_t)logq_A2_16u_kernel(k & 0x7FFFFFu); }
NOINL static uint32_t q_A2_16u(uint32_t b) { return (uint32_t)logq_A2_16u_q23(b); }
NOINL static uint32_t u_A2_16u(uint32_t v) { return (uint32_t)logq_A2_16u_q23_u32(v); }

/* natural log of a Q31 input, q5.26 output: logq vs CMSIS-DSP */
NOINL static uint32_t ln_A3_16u(uint32_t s)
{
    int32_t l2 = logq_A3_16u_q23_u32(s) - (31 << 23);
    return (uint32_t)(int32_t)(((int64_t)l2 * 0x58B90BFC) >> 28);
}
NOINL static uint32_t b_cmsis_log_q31(uint32_t s) { return arm_scalar_log_q31(s); }

/* binary32 library functions */
NOINL static uint32_t b_libm_log2f(uint32_t b) { return f2u(log2f(u2f(b))); }   /* toolchain libm */
NOINL static uint32_t b_aor_log2f(uint32_t b)  { return f2u(aor_log2f(u2f(b))); }
NOINL static uint32_t b_fbsd_log2f(uint32_t b) { return f2u(fbsd_log2f(u2f(b))); }

/* algorithms of the 2023 preprint */
NOINL static uint32_t l_log2_Mitchell(uint32_t b) { return f2u(log2_Mitchell(u2f(b))); }
NOINL static uint32_t l_log2_P2_16(uint32_t b)    { return f2u(log2_P2_16(u2f(b))); }
NOINL static uint32_t l_log2_P3_16(uint32_t b)    { return f2u(log2_P3_16(u2f(b))); }

typedef uint32_t (*fn_t)(uint32_t);
enum { IN_FLOAT, IN_FRAC, IN_U32, IN_Q31 };
static const char *const in_name[] = { "float", "frac", "u32", "q31" };
static const uint32_t *const in_data[] = { IN_FLOAT_DATA, IN_FRAC_DATA, IN_U32_DATA, IN_Q31_DATA };

#define E(fn, in) { #fn, fn, in }
static const struct { const char *name; fn_t f; int in; } FN[] = {
    E(empty_call, IN_FLOAT),
    E(f_A1_16u, IN_FLOAT), E(f_A2_16u, IN_FLOAT), E(f_A2_16g, IN_FLOAT), E(f_A3_16u, IN_FLOAT),
    E(f_R3_16, IN_FLOAT), E(f_R3_32, IN_FLOAT),
    E(q_A3_16u, IN_FLOAT), E(q_A3_16u_softfloat, IN_FLOAT), E(q_A3_16u_intfloat, IN_FLOAT),
    E(k_A2_16u, IN_FRAC), E(q_A2_16u, IN_FLOAT), E(u_A2_16u, IN_U32),
    E(ln_A3_16u, IN_Q31), E(b_cmsis_log_q31, IN_Q31),
    E(b_libm_log2f, IN_FLOAT), E(b_aor_log2f, IN_FLOAT), E(b_fbsd_log2f, IN_FLOAT),
    E(l_log2_Mitchell, IN_FLOAT), E(l_log2_P2_16, IN_FLOAT), E(l_log2_P3_16, IN_FLOAT),
};
#define NFN ((int)(sizeof FN / sizeof FN[0]))

static uint32_t fnv(uint32_t h, uint32_t v)
{
    int i;
    for (i = 0; i < 4; i++) { h ^= (v >> (8 * i)) & 0xFFu; h *= 16777619u; }
    return h;
}

/* cycles of each call, measured one call at a time */
static uint32_t cyc[BENCH_N];

void bench_run(const char *board, unsigned mhz)
{
    double empty_mean = 0;
    int j, i, rep;
    unsigned bad = 0;

    printf("CSV,board,mhz,function,input,mean_cycles_net,min_cycles_net,max_cycles_net,checksum,expected,match\n");
    for (j = 0; j < NFN; j++) {
        const uint32_t *x = in_data[FN[j].in];
        uint32_t h = 2166136261u, mn = 0xFFFFFFFFu, mx = 0;
        double sum = 0, net;
        for (rep = 0; rep < 2; rep++) {            /* rep 0 warms caches and branch predictors */
            bench_irq_off();
            for (i = 0; i < BENCH_N; i++) {
                uint32_t c0 = bench_cycles();
                uint32_t y = FN[j].f(x[i]);
                uint32_t c1 = bench_cycles();
                cyc[i] = c1 - c0;
                if (rep) h = fnv(h, y);
            }
            bench_irq_on();
        }
        for (i = 0; i < BENCH_N; i++) {
            sum += cyc[i];
            if (cyc[i] < mn) mn = cyc[i];
            if (cyc[i] > mx) mx = cyc[i];
        }
        if (j == 0) empty_mean = sum / BENCH_N;
        net = sum / BENCH_N - empty_mean;
        {
            const char *m = EXPECTED[j] == 0 ? "n/a" : (EXPECTED[j] == h ? "yes" : "NO");
            if (m[0] == 'N') bad++;
            printf("CSV,%s,%u,%s,%s,%.1f,%.0f,%.0f,0x%08lX,0x%08lX,%s\n", board, mhz, FN[j].name,
                   in_name[FN[j].in], net, mn - empty_mean, mx - empty_mean,
                   (unsigned long)h, (unsigned long)EXPECTED[j], m);
        }
        bench_pause();
    }
    printf("DONE: %d functions x %d inputs, checksum mismatches: %u\n", NFN, BENCH_N, bad);
}

/* host side: print the checksums that become expected.h */
void bench_print_expected(void)
{
    int j, i;
    printf("/* Generated by host_main.c: FNV-1a of the 1000 results of each function.\n"
           " * 0 = not checked (the toolchain's own log2f differs between libraries). */\n");
    printf("static const uint32_t EXPECTED[] = {\n");
    for (j = 0; j < NFN; j++) {
        const uint32_t *x = in_data[FN[j].in];
        uint32_t h = 2166136261u;
        for (i = 0; i < BENCH_N; i++) h = fnv(h, FN[j].f(x[i]));
        if (FN[j].f == b_libm_log2f) h = 0;
        printf("    0x%08lXu, /* %s */\n", (unsigned long)h, FN[j].name);
    }
    printf("};\n");
}
