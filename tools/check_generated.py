#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate include/logq_tables.h into a temporary file and fail if it differs
from the committed one (the tables in the repository are reproducible)."""
import subprocess, sys, tempfile, filecmp
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "logq_tables.h"
    subprocess.run([sys.executable, str(root / "tools" / "logqgen.py"), "--all", "-o", str(out)],
                   check=True, stderr=subprocess.DEVNULL)
    same = filecmp.cmp(out, root / "include" / "logq_tables.h", shallow=False)
print("include/logq_tables.h is reproducible" if same else "include/logq_tables.h differs from the generator output")
sys.exit(0 if same else 1)
