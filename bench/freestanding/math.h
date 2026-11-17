/* Minimal <math.h> for the bare-metal benchmark build: logbf for binary32
 * (used by one of the 2023 algorithms), integer-only as in fdlibm. */
#ifndef BENCH_MATH_H
#define BENCH_MATH_H
#include <stdint.h>
static float logbf(float x)
{
    uint32_t b; int e;
    __builtin_memcpy(&b, &x, 4);
    b &= 0x7FFFFFFFu;
    if (b == 0) return -__builtin_inff();
    if (b >= 0x7F800000u) return x * x;
    e = (int)(b >> 23);
    if (e == 0) return (float)(-127 - (__builtin_clz(b) - 9));   /* subnormal */
    return (float)(e - 127);
}
#endif
