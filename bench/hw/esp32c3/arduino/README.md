# logq benchmark on ESP32-C3 with the Arduino IDE

Port of the ESP-IDF benchmark (`../main`) to the Arduino IDE by Paweł
Poczekajło. It runs the same code and prints the same CSV lines; only the entry
point (`log2_v3_esp32c3.ino`) and the platform functions (`bench_port.c`)
differ. Tested with Arduino-ESP32 core 3.3.12 (ESP-IDF 5.5.5, GCC 14.2.0) on a
Waveshare ESP32-C3-Zero.

## Run

1. Copy the shared sources into the sketch folder:
   `python3 make_sketch.py` (or copy by hand the files listed in it from
   `../main`, everything except `main.c`).
2. Open `log2_v3_esp32c3/log2_v3_esp32c3.ino`, select an ESP32-C3 board, set
   `BOARD_NAME` at the top of the sketch, upload, and open the serial monitor
   at 115200 baud.
3. Copy the lines starting with `CSV,` into `hw_esp32c3.csv`. The last line
   must read `checksum mismatches: 0`.

## Optimisation level

With **Sketch → Optimize for Debugging** enabled the Arduino IDE compiles with
`-Og -g3`. `bench_core.c` therefore forces `-O2` with
`#pragma GCC optimize ("O2")` when it is built by the Arduino IDE (define
`LOGQ_BENCH_KEEP_FLAGS` to disable this), so that the results are comparable
with the ESP-IDF project and the emulation. The toolchain's `log2f` and the
soft-float routines are precompiled and are not affected.

## Code sizes

The Arduino IDE keeps the `.elf` in its build folder (shown in the output
window with verbose compilation enabled). Then:

```sh
riscv32-esp-elf-nm -S --size-sort -t d log2_v3_esp32c3.ino.elf | grep -E ' [tTrR] (f_|q_|k_|u_|ln_|b_|l_|logq_|__log2f_data)' > sizes_esp32c3.txt
riscv32-esp-elf-readelf -p .comment log2_v3_esp32c3.ino.elf      # compiler version
```
