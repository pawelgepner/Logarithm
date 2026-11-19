/* SPDX-License-Identifier: MIT */
#ifndef BENCH_CORE_H
#define BENCH_CORE_H
#include <stdint.h>
#define BENCH_N 1000
/* provided by the platform */
uint32_t bench_cycles(void);
void bench_irq_off(void);
void bench_irq_on(void);
void bench_pause(void);
/* provided by bench_core.c */
void bench_run(const char *board, unsigned mhz);
void bench_print_expected(void);
#endif
