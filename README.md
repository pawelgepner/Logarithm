# logq — integer-only binary logarithms with verified error bounds

`logq` computes log₂ on processors **without a floating-point unit** (RV32I/M,
Cortex-M0/M0+/M3, small DSPs) using integer operations only. It is a
header-only C99 library plus a generator, `tools/logqgen.py`, that produces a
logarithm of any accuracy between about 4·10⁻² and 6·10⁻⁸ and **verifies its
worst-case error exhaustively** before emitting the code.

* **Family A(n, p, σ)** — piecewise minimax polynomial of degree *n* = 1–3 on
  *p* = 1–64 uniform (σ = u) or geometric (σ = g) segments of the significand.
  Result in Q8.23 fixed point with a guaranteed **absolute** error bound. The
  natural choice for dB levels, log-mel features, entropies and LNS conversion.
* **Family R(n, p)** — same construction applied to log₂(1+z)/z after
  re-centring the argument, giving a guaranteed **relative** error (down to
  0.52 ulp of binary32 for R3_64) with no cancellation near x = 1.
* Entry points for a **binary32** argument (full IEEE 754 domain: zeros,
  subnormals, infinities, NaN), for a **32-bit integer** argument, and the bare
  kernel. No floating-point instruction is executed in any of them.

## Quick start

```c
#include "logq.h"                     /* all 54 catalogued variants */

int32_t q = logq_A2_16u_q23_u32(12345);      /* log2(12345) in Q8.23 */
float   y = logq_A2_16u_f32(3.7f);           /* full binary32 domain */
float   r = logq_R3_16_f32(1.0001f);         /* bounded relative error */
```

A product normally needs one variant. Generate it with its own prefix:

```sh
python3 tools/logqgen.py -n 2 -p 4 --seg g --prefix mylog_ -o mylog.h
# A2_4g  tables 64 B  minimax 7.813e-05  verified [-7.819e-05, +7.819e-05]
```

Requirements of the generator: Python 3.8+ and NumPy.

## Choosing a variant

The kernel error follows E ≈ C·p^-(n+1) until the Q23 output resolution
(1.2·10⁻⁷) is reached. Selected rows of `results/kernel_errors.csv` and
`results/mcu_instructions.csv` (binary32 → binary32, full domain, executed
instructions, mean over 1000 inputs):

| Variant | Max abs. error of log₂ | Tables | RV32IM | Cortex-M0 |
|---|---|---|---|---|
| A1_16u | 3.3·10⁻⁴ | 128 B | 76 | 132 |
| A2_16u | 3.4·10⁻⁶ | 192 B | 85 | 174 |
| A2_16g | 1.3·10⁻⁶ | 256 B | 109 | 197 |
| A3_16u | 9.9·10⁻⁸ | 256 B | 90 | 209 |
| R3_16 (0.85 ulp) | — | 256 B | 114 | 287 |
| Arm Optimized Routines `log2f` (musl, glibc) | 0.75 ulp | 288 B | 1548 | 3154 |
| FreeBSD / fdlibm `log2f` | 0.86 ulp | — | 2131 | 3501 |

The numbers above are copied from the result files of release v1.0.0; rerun
the commands below to reproduce them.

On a desktop CPU with hardware floating point, logq is **not** faster than the
system `log2f`; its purpose is FPU-less targets.

## Reproducing the results

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
./build/test_float  > results/float_exhaustive.csv   # all 2^31 positive binary32, ~15 min on 2 cores
./build/test_q31_ln > results/q31_ln_vs_cmsis.csv    # all 2^31-1 Q31 inputs vs CMSIS-DSP
./build/test_baselines > results/baselines_exhaustive.csv
./build/bench_host > results/host_x86_64.csv
python3 tools/make_tables.py paper/gen              # tables and numbers of the paper
# instruction counts on RV32IM and Cortex-M0 (Unicorn emulator, soft-float);
# clock cycles on an ESP32-C3 board: see bench/hw/esp32c3/README.md
BUILTINS=/path/to/llvm-project/compiler-rt/lib/builtins sh bench/build_mcu.sh
cc -O2 -DHOST_REFERENCE -Iinclude -o bench/build/host_ref bench/mcu_funcs.c -lm
pip install unicorn && python3 bench/emu.py
```

| Program | What it verifies |
|---|---|
| `tools/logqgen.py` | worst-case kernel error of each emitted variant over all 2²³ significands |
| `tests/test_kernel.c` | same, independently in C; monotonicity; exactness at powers of two |
| `tests/test_float.c` | every positive finite binary32 input: error ≤ kernel bound + ½ ulp of the result; relative and ulp errors |
| `tests/test_special.c` | zeros, infinities, NaNs, subnormals, extreme values; family R exact at all powers of two; integer entry point over all 2³² inputs |
| `tests/test_q31_ln.c` | natural log of every Q31 input against CMSIS-DSP `arm_vlog_q31` |
| `tests/test_baselines.c` | the same measures for the comparison functions in `baselines/` |
| `bench/emu.py` | executed instructions and code footprint; every emulated result checked bit-for-bit against the host |
| `bench/hw/esp32c3/` | clock cycles on a real ESP32-C3 (ESP-IDF or Arduino IDE); every result checked against the PC by checksums |

## Repository layout

| Path | Content |
|---|---|
| `include/` | `logq.h` and the generated `logq_tables.h` |
| `tools/` | generator `logqgen.py`, reproducibility check |
| `tests/` | exhaustive tests |
| `bench/` | MCU emulation, cross-compilation script, host benchmark, ESP32-C3 hardware benchmark (`bench/hw/esp32c3`) |
| `examples/` | integer dB meter, generated single-variant header |
| `baselines/` | third-party `log2f`/`log` implementations and the 2023 algorithms, each under its own licence |
| `results/` | result files cited in the paper |

## History

The first version of this code accompanied the 2023 preprint *Innovative
simple yet efficient algorithms of logarithms with adjustable accuracy*
(Research Square); it is preserved unchanged under the tag
https://github.com/pawelgepner/Logarithm/releases/tag/v0.1-preprint
and in `baselines/legacy/`. Version 1.0 replaces it with a generator, integer
kernels, full-domain handling and exhaustive verification. The speed-ups of
3–20× over C libraries reported in the preprint came from a benchmark that
evaluated a single constant (subnormal) argument and are not reproduced by
version 1.0 on hosts with hardware floating point.

## Licence and citation

MIT (see `LICENSE`); third-party files in `baselines/` keep their own licences.
See `CITATION.cff` for how to cite.


