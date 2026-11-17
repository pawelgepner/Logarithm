/* SPDX-License-Identifier: MIT
 * Example: a product that needs a single logarithm generates its own header,
 * with its own prefix, instead of including the full catalogue:
 *
 *   python3 tools/logqgen.py -n 2 -p 4 --seg g --prefix mylog_ -o mylog.h
 *
 * which emits mylog_A2_4g_{kernel,q23,q23_u32,f32bits,f32} with 64 bytes of
 * tables and prints the verified kernel error (7.8e-5).  This file is built by
 * CMake after that command has run.
 */
#include <stdio.h>
#include <math.h>
#include "mylog.h"

int main(void)
{
    const float xs[] = { 1e-3f, 0.5f, 1.0f, 3.0f, 1000.0f, 1e30f };
    unsigned i;
    for (i = 0; i < sizeof xs / sizeof xs[0]; i++)
        printf("log2(%g) = %.6f   (libm: %.6f)\n", xs[i], mylog_A2_4g_f32(xs[i]), log2(xs[i]));
    return 0;
}
