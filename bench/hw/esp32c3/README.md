# logq benchmark on ESP32-C3 (ESP-IDF)

Measures cycles per call of logq and of the comparison functions on the same
1000 inputs as the emulation (`bench/emu.py`), and checks on the board that
every function returns exactly the results computed on a PC (checksums in
`main/expected.h`).

Measured functions: six logq variants (binary32 in/out, full domain); the
A3_16u fixed-point result alone and converted to float in two ways (soft-float
arithmetic vs integer-only packing); A2_16u kernel, fixed-point and integer
entry points; natural log of Q31 (logq vs CMSIS-DSP); `log2f` of the
toolchain's libm, of Arm Optimized Routines and of FreeBSD; three algorithms of
the 2023 preprint.

## Run

Requires an installed ESP-IDF 5.x (tested: 5.5.5). The `idf.py` command comes
with the ESP-IDF installation, not with this project; see the ESP-IDF
*Get Started* guide for ESP32-C3. To build with the Arduino IDE instead, see
[`arduino/`](arduino/README.md).

```sh
cd bench/hw/esp32c3
idf.py set-target esp32c3
idf.py build flash monitor | tee log.txt       # Ctrl+] ends the monitor after "DONE"
grep '^CSV,' log.txt | cut -c5- > hw_esp32c3.csv
```

The last line on the console must read `checksum mismatches: 0`. A mismatch
means that a function returns different bits on the board than on a PC; send
the log in that case.

Code size of each measured function (inlined logq code included) and of the
tables:

```sh
riscv32-esp-elf-nm -S --size-sort -t d build/logq_bench.elf | grep -E ' [tTrR] (f_|q_|k_|u_|ln_|b_|l_|logq_|__log2f_data)' > sizes_esp32c3.txt
```

Please send back `hw_esp32c3.csv`, `sizes_esp32c3.txt`, the ESP-IDF version
(`idf.py --version`) and the board name.

## Notes

- Columns: `mean/min/max_cycles_net` are cycles per call minus the mean cost of
  an empty call (`empty_call`, first row). Each function runs twice over the
  inputs with interrupts disabled; the second pass is reported.
- The project is compiled with `-Os` (`main/CMakeLists.txt`,
  `CONFIG_COMPILER_OPTIMIZATION_SIZE`), the level of the precompiled
  toolchain `log2f` and soft-float routines; the emulation uses `-O2`.
  Results of the paper: `results/hw_esp32c3.csv`.
- `q_A3_16u_softfloat` divides by 2^23 in the source; the compiler replaces
  this by an exact multiplication, so the measured overhead is `__floatsisf`
  plus `__mulsf3`, not a division.
- `b_libm_log2f` is the toolchain's own `log2f` (newlib); its checksum is not
  compared because different libraries round differently.
- The files `logq.h`, `logq_tables.h` and those from `baselines/` are copies
  from the repository, so the project builds on its own. `make_inputs.py` and
  `host_main.c` regenerate `inputs.h` and `expected.h`:
  `python3 make_inputs.py && cc -O2 -ffp-contract=off -Imain host_main.c main/bench_core.c -lm -o hb && ./hb > main/expected.h`
- Porting to another board: replace `main.c` (four functions: cycle counter,
  interrupts off/on, pause); `arduino/` is an example of such a port.
