/* SPDX-License-Identifier: MIT
 * Host build of the benchmark core: prints expected.h (checksums) or, with
 * "run", the benchmark itself with a dummy cycle counter (sanity check). */
#include <stdio.h>
#include <string.h>
#include "bench_core.h"
static uint32_t t;
uint32_t bench_cycles(void) { return t++; }
void bench_irq_off(void) {}
void bench_irq_on(void) {}
void bench_pause(void) {}
int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "run")) bench_run("host", 0);
    else bench_print_expected();
    return 0;
}
