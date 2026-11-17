# Result files

Produced by the commands in the top-level README (release v1.0.0, host: x86-64 KVM
guest, 2 cores; emulation: clang 18.1.3, compiler-rt 18.1.3, Unicorn 2.1.4).

| File | Produced by | Content |
|---|---|---|
| `kernel_errors.csv` | `tools/logqgen.py --all --report` | minimax error and verified error of every variant |
| `coefficients.json` | `tools/logqgen.py --all --json` | segment boundaries and Q30 coefficients |
| `kernel_exhaustive.csv` | `tests/test_kernel.c` | family A: kernel error over all 2^23 significands, monotonicity, exactness at 2^e |
| `float_exhaustive.csv` | `tests/test_float.c` | all 54 variants over all positive finite binary32 inputs |
| `special_values.txt` | `tests/test_special.c` | special values, integer entry point over all 2^32-1 inputs |
| `baselines_exhaustive.csv` | `tests/test_baselines.c` | Arm Optimized Routines, FreeBSD and 2023 algorithms over all positive binary32 |
| `q31_ln_vs_cmsis.csv` | `tests/test_q31_ln.c` | natural log of all Q31 inputs, logq vs CMSIS-DSP |
| `mcu_instructions.csv` | `bench/emu.py --n 1000` | executed instructions and footprint on RV32IM and Cortex-M0 |
| `host_x86_64.csv` | `bench/bench_host.c` | ns per call on the host (reference only) |
