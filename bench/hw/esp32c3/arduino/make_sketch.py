#!/usr/bin/env python3
"""Copy the benchmark sources from ../main into the Arduino sketch folder.

The Arduino IDE compiles only the files inside the sketch folder, so the
shared sources (bench_core.c and the headers) are copied there.  main.c is
not copied: the sketch provides setup() and bench_port.c instead.
Run again after any change in ../main.
"""
import shutil
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "main"
dst = here / "log2_v3_esp32c3"
files = ["bench_core.c", "bench_core.h", "logq.h", "logq_tables.h",
         "aor_log2f.c", "fbsd_log2f.c", "cmsis_log_q31.c",
         "legacy_algos.h", "inputs.h", "expected.h"]
for name in files:
    shutil.copy2(src / name, dst / name)
    print("copied", name)
print("Open", dst / "log2_v3_esp32c3.ino", "in the Arduino IDE.")
