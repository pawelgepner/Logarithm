/* Minimal <string.h> for the bare-metal benchmark build. */
#ifndef BENCH_STRING_H
#define BENCH_STRING_H
#include <stddef.h>
#define memcpy(d, s, n) __builtin_memcpy((d), (s), (n))
#endif
