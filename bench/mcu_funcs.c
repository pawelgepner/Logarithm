/* SPDX-License-Identifier: MIT
 * Functions measured in the instruction-count emulation (bench/emu.py).
 * Every entry point has the signature uint32_t f(uint32_t) so that binary32
 * arguments and results travel as bit patterns in an integer register; the
 * soft-float ABI of the targets then plays no role in the measurement.
 * The same file is compiled for the host to produce reference outputs.
 */
#include <stdint.h>
#include <string.h>
#include "logq.h"
#include "../baselines/legacy_algos.h"
#include "../baselines/aor_log2f.c"
#undef N
#undef T
#undef A
#undef OFF
#include "../baselines/fbsd_log2f.c"
#include "../baselines/cmsis_log_q31.c"

#define NOINL __attribute__((noinline, used))
static inline float u2f(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

/* logq family A: kernel only, Q8.23 from a normal binary32, Q8.23 from an
 * integer, binary32 over the full domain */
#define MEASURE_A(V) \
  NOINL uint32_t k_##V(uint32_t k) { return (uint32_t)logq_##V##_kernel(k & 0x7FFFFFu); } \
  NOINL uint32_t q_##V(uint32_t b) { return (uint32_t)logq_##V##_q23(b); }                \
  NOINL uint32_t u_##V(uint32_t v) { return (uint32_t)logq_##V##_q23_u32(v); }            \
  NOINL uint32_t f_##V(uint32_t b) { return logq_##V##_f32bits(b); }
MEASURE_A(A1_16u) MEASURE_A(A2_2g) MEASURE_A(A2_8u) MEASURE_A(A2_16u) MEASURE_A(A2_16g)
MEASURE_A(A2_32u) MEASURE_A(A3_16u)

/* logq family R */
#define MEASURE_R(V) NOINL uint32_t f_##V(uint32_t b) { return logq_##V##_f32bits(b); }
MEASURE_R(R2_16) MEASURE_R(R2_64) MEASURE_R(R3_16) MEASURE_R(R3_32)

/* CMSIS-compatible natural logarithm: Q31 input in (0, 1), q5.26 output,
 * computed with a logq kernel (drop-in replacement for arm_scalar_log_q31) */
#define LN2_Q31 0x58B90BFCu
#define MEASURE_LN(V) NOINL uint32_t ln_##V(uint32_t src) { \
    int32_t l2 = logq_##V##_q23_u32(src) - (31 << 23);                   /* log2(src/2^31), Q23 */ \
    return (uint32_t)(int32_t)(((int64_t)l2 * LN2_Q31) >> 28); }      /* x ln 2, Q26 */
MEASURE_LN(A2_16u) MEASURE_LN(A3_16u)

/* baselines */
NOINL uint32_t b_cmsis_log_q31(uint32_t s) { return arm_scalar_log_q31(s); }
NOINL uint32_t b_aor_log2f(uint32_t b)   { return f2u(aor_log2f(u2f(b))); }
NOINL uint32_t b_fbsd_log2f(uint32_t b)  { return f2u(fbsd_log2f(u2f(b))); }
#define LEGACY(F) NOINL uint32_t l_##F(uint32_t b) { return f2u(F(u2f(b))); }
LEGACY(log2_Mitchell) LEGACY(log2_P2_1) LEGACY(log2_P2_2) LEGACY(log2_P2_int_2)
LEGACY(log2_P1_16_eq) LEGACY(log2_P1_16_neq) LEGACY(log2_P2_16) LEGACY(log2_P2_16_neq) LEGACY(log2_P3_16)

NOINL uint32_t empty_call(uint32_t b) { return b; }

#ifdef HOST_REFERENCE
/* Host build: read "<name> <hex input>" lines, print the result. */
#include <stdio.h>
typedef uint32_t (*fn_t)(uint32_t);
#define E_A(V) {"k_" #V, k_##V}, {"q_" #V, q_##V}, {"u_" #V, u_##V}, {"f_" #V, f_##V},
#define E_R(V) {"f_" #V, f_##V},
#define E_L(F) {"l_" #F, l_##F},
static const struct { const char *n; fn_t f; } TAB[] = {
  E_A(A1_16u) E_A(A2_2g) E_A(A2_8u) E_A(A2_16u) E_A(A2_16g) E_A(A2_32u) E_A(A3_16u)
  E_R(R2_16) E_R(R2_64) E_R(R3_16) E_R(R3_32)
  {"ln_A2_16u", ln_A2_16u}, {"ln_A3_16u", ln_A3_16u},
  {"b_cmsis_log_q31", b_cmsis_log_q31}, {"b_aor_log2f", b_aor_log2f}, {"b_fbsd_log2f", b_fbsd_log2f},
  E_L(log2_Mitchell) E_L(log2_P2_1) E_L(log2_P2_2) E_L(log2_P2_int_2) E_L(log2_P1_16_eq)
  E_L(log2_P1_16_neq) E_L(log2_P2_16) E_L(log2_P2_16_neq) E_L(log2_P3_16)
  {"empty_call", empty_call}};
int main(void)
{
    char n[64]; unsigned x; size_t i;
    while (scanf("%63s %x", n, &x) == 2)
        for (i = 0; i < sizeof TAB / sizeof TAB[0]; i++)
            if (!strcmp(n, TAB[i].n)) { printf("%08X\n", TAB[i].f(x)); break; }
    return 0;
}
#else
/* bare-metal entry point, never called: keeps the linker happy */
void _start(void) { for (;;) ; }
#endif
