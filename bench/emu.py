#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Count executed instructions of the functions in mcu_funcs.c on RV32IM and
ARMv6-M (Cortex-M0) by running the cross-compiled images in the Unicorn
emulator.  Every result is also checked bit for bit against the same function
compiled for the host.  The code footprint of each function is the size of a
bare-metal image that contains only that function and everything it needs
(tables and soft-float helpers included): bench/build_mcu.sh links every
function on its own with --gc-sections.

Usage: python3 bench/emu.py [--n 1000] [--out results/mcu_instructions.csv]
Needs: pip install unicorn; bench/build/mcu_{rv32im,m0}.elf (bench/build_mcu.sh)
and a host build: cc -O2 -DHOST_REFERENCE -Iinclude -o bench/build/host_ref bench/mcu_funcs.c
"""
import argparse
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

import unicorn as U
from unicorn import riscv_const as RV
from unicorn import arm_const as AR

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
SENTINEL = 0x0000F000
STACK_TOP = 0x00200000


# ----------------------------------------------------------------------------- ELF
def load_elf(path):
    data = path.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 1, "ELF32 expected"
    (e_phoff, e_shoff) = struct.unpack_from("<II", data, 28)
    (e_phentsize, e_phnum) = struct.unpack_from("<HH", data, 42)
    segs = []
    for i in range(e_phnum):
        p_type, p_off, p_vaddr, _, p_filesz, p_memsz, _, _ = struct.unpack_from("<8I", data, e_phoff + i * e_phentsize)
        if p_type == 1:
            segs.append((p_vaddr, data[p_off:p_off + p_filesz], p_memsz))
    return segs


def symbols(path):
    out = subprocess.run(["llvm-nm", "-S", "--defined-only", str(path)], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4:
            syms[f[3]] = (int(f[0], 16) & ~1, int(f[1], 16), f[2])
    return syms


def footprint(target):
    out = {}
    for line in (BUILD / f"footprint_{target}.txt").read_text().splitlines():
        fn, size = line.split()
        out[fn] = int(size)
    return out


# ----------------------------------------------------------------------------- targets
class Target:
    def __init__(self, name):
        self.name = name
        self.elf = BUILD / f"mcu_{name}.elf"
        self.syms = symbols(self.elf)
        self.foot = footprint(name)
        if name == "rv32im":
            self.uc = U.Uc(U.UC_ARCH_RISCV, U.UC_MODE_RISCV32)
        else:
            self.uc = U.Uc(U.UC_ARCH_ARM, U.UC_MODE_THUMB | U.UC_MODE_MCLASS)
            self.uc.ctl_set_cpu_model(U.arm_const.UC_CPU_ARM_CORTEX_M0)
        self.uc.mem_map(0, STACK_TOP)
        for vaddr, blob, memsz in load_elf(self.elf):
            self.uc.mem_write(vaddr, blob + b"\0" * (memsz - len(blob)))
        self.uc.mem_write(SENTINEL, b"\x00" * 16)
        self.count = 0
        self.cache = {}
        self.uc.hook_add(U.UC_HOOK_BLOCK, self._block)

    def _block(self, uc, addr, size, _):
        n = self.cache.get((addr, size))
        if n is None:
            if self.name == "rv32im":
                n = size // 4                                  # RV32IM: no compressed instructions
            else:
                code, n, i = bytes(uc.mem_read(addr, size)), 0, 0
                while i < size:                                # Thumb: 16- or 32-bit encodings
                    h = code[i] | code[i + 1] << 8
                    i += 4 if (h >> 11) in (0b11101, 0b11110, 0b11111) else 2
                    n += 1
            self.cache[(addr, size)] = n
        self.count += n

    def call(self, fn, arg):
        addr = self.syms[fn][0]
        uc = self.uc
        if self.name == "rv32im":
            uc.reg_write(RV.UC_RISCV_REG_A0, arg)
            uc.reg_write(RV.UC_RISCV_REG_RA, SENTINEL)
            uc.reg_write(RV.UC_RISCV_REG_SP, STACK_TOP - 16)
            self.count = 0
            uc.emu_start(addr, SENTINEL, count=200000)
            res = uc.reg_read(RV.UC_RISCV_REG_A0)
        else:
            uc.reg_write(AR.UC_ARM_REG_R0, arg)
            uc.reg_write(AR.UC_ARM_REG_LR, SENTINEL | 1)
            uc.reg_write(AR.UC_ARM_REG_SP, STACK_TOP - 16)
            self.count = 0
            uc.emu_start(addr | 1, SENTINEL, count=200000)
            res = uc.reg_read(AR.UC_ARM_REG_R0)
        return res & 0xFFFFFFFF, self.count


# ----------------------------------------------------------------------------- inputs
def inputs(kind, n, rng):
    if kind == "float":        # positive normal binary32, log-uniform in [2^-20, 2^20)
        return [((127 + rng.randrange(-20, 20)) << 23) | rng.getrandbits(23) for _ in range(n)]
    if kind == "frac":         # 23-bit significand fraction
        return [rng.getrandbits(23) for _ in range(n)]
    if kind == "u32":          # integer >= 1, uniformly distributed bit length 1..32
        return [(1 << (b - 1)) | rng.getrandbits(b - 1) if b > 1 else 1
                for b in (rng.randrange(1, 33) for _ in range(n))]
    if kind == "q31":          # Q31 in (0, 1): integer >= 1 with bit length 1..31
        return [(1 << (b - 1)) | rng.getrandbits(b - 1) if b > 1 else 1
                for b in (rng.randrange(1, 32) for _ in range(n))]
    raise ValueError(kind)


def kind_of(fn):
    if fn.startswith("k_"):
        return "frac"
    if fn.startswith("u_"):
        return "u32"
    if fn.startswith("ln_") or fn == "b_cmsis_log_q31":
        return "q31"
    return "float"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "mcu_instructions.csv"))
    a = ap.parse_args()

    tgts = [Target("rv32im"), Target("m0")]
    fns = sorted(n for n in tgts[0].syms if re.match(r"^(k|q|u|f|ln|b|l)_", n) or n == "empty_call")
    rng = random.Random(2026)
    sets = {k: inputs(k, a.n, rng) for k in ("float", "frac", "u32", "q31")}

    # host reference
    host = BUILD / "host_ref"
    feed = "".join(f"{fn} {x:08X}\n" for fn in fns for x in sets[kind_of(fn)])
    ref = subprocess.run([str(host)], input=feed, capture_output=True, text=True, check=True).stdout.split()
    ref = iter(int(r, 16) for r in ref)

    rows, mism = [], 0
    for fn in fns:
        xs = sets[kind_of(fn)]
        expected = [next(ref) for _ in xs]
        res = {}
        for t in tgts:
            cnt = []
            for x, e in zip(xs, expected):
                y, c = t.call(fn, x)
                cnt.append(c)
                if y != e:
                    mism += 1
                    if mism <= 10:
                        print(f"MISMATCH {t.name} {fn}({x:08X}) = {y:08X}, host {e:08X}", file=sys.stderr)
            res[t.name] = (sum(cnt) / len(cnt), min(cnt), max(cnt), t.foot[fn])
        rows.append((fn, kind_of(fn), res))
        r, m = res["rv32im"], res["m0"]
        print(f"{fn:22s} rv32im {r[0]:8.1f} [{r[1]:5d},{r[2]:5d}] {r[3]:6d} B   "
              f"m0 {m[0]:8.1f} [{m[1]:5d},{m[2]:5d}] {m[3]:6d} B", file=sys.stderr)

    with open(a.out, "w") as f:
        f.write("function,input,rv32im_mean,rv32im_min,rv32im_max,rv32im_footprint_bytes,"
                "m0_mean,m0_min,m0_max,m0_footprint_bytes\n")
        for fn, kind, res in rows:
            r, m = res["rv32im"], res["m0"]
            f.write(f"{fn},{kind},{r[0]:.1f},{r[1]},{r[2]},{r[3]},{m[0]:.1f},{m[1]},{m[2]},{m[3]}\n")
    print(f"{len(fns)} functions x {a.n} inputs x 2 targets; results differing from host: {mism}", file=sys.stderr)
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
