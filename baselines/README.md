# Baselines (third-party and historical code)

These files are used only by the tests and benchmarks. They are not part of
the logq library and keep their own licences.

| File | Origin | Licence |
|---|---|---|
| `aor_log2f.c` | Arm Optimized Routines `math/log2f.c`, `log2f_data.c` (commit 20859d3); the `log2f` of musl and glibc | MIT OR Apache-2.0 WITH LLVM-exception |
| `fbsd_log2f.c` | FreeBSD `lib/msun/src/e_log2f.c`, `k_logf.h`; derived from Sun fdlibm | Sun Microsystems permissive notice (in the file) |
| `cmsis_log_q31.c` | CMSIS-DSP `Source/FastMathFunctions/arm_vlog_q31.c` (commit 5a28089), scalar kernel | Apache-2.0 |
| `legacy_algos.h` | Algorithms of the 2023 preprint, extracted from `legacy/main.cpp` | MIT (this project) |
| `legacy/` | The code of the 2023 preprint, unchanged (also tag `v0.1-preprint`) | mixed: own code MIT; `log2f_musl` MIT; `log2f_Sun` Sun notice; `log2f_fsf*` **GPL-3.0-or-later** (gnulib) |

Only the function bodies were copied; the library-internal headers they
depended on were replaced by minimal definitions, as stated at the top of each
file.
