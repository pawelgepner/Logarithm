#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce the LaTeX tables and plot data of the SoftwareX paper from the
files in results/.  Usage: tools/make_tables.py <output directory>"""
import csv, sys, math
from pathlib import Path

root = Path(__file__).resolve().parent.parent
R = root / "results"
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

def rd(name, key):
    return {r[key]: r for r in csv.DictReader(open(R / name))}

kern = rd("kernel_exhaustive.csv", "name")
gen = rd("kernel_errors.csv", "name")
flt = rd("float_exhaustive.csv", "name")
mcu = rd("mcu_instructions.csv", "function")
base = {(r["name"], r["inputs"]): r for r in csv.DictReader(open(R / "baselines_exhaustive.csv"))}
q31 = rd("q31_ln_vs_cmsis.csv", "function")

def sci(x, d=1):
    if x == 0: return "0"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10: m /= 10; e += 1
    return f"${m:.{d}f}\\cdot 10^{{{e}}}$"

def E(n): return max(-float(kern[n]["err_min"]), float(kern[n]["err_max"]))
def inst(f, t): return f"{float(mcu[f][t + '_mean']):.0f}"
def foot(f): return mcu[f]["m0_footprint_bytes"]

# ---- plot data: kernel error vs segments
for seg in "ug":
    for n in (1, 2, 3):
        rows = [(int(gen[k]["segments"]), E(k)) for k in kern if gen[k]["family"] == "A"
                and gen[k]["degree"] == str(n) and gen[k]["segmentation"] == seg]
        with open(out / f"err_n{n}{seg}.dat", "w") as f:
            f.write("p E\n" + "".join(f"{p} {e:.6e}\n" for p, e in sorted(rows)))

# ---- Table: accuracy and cost, binary32 -> binary32
L = []
def row(label, abs_err, other, f, extra=""):
    L.append(f"{label} & {abs_err} & {other} & {inst(f,'rv32im')} & {inst(f,'m0')} & {foot(f)}{extra} \\\\")
for v in ("A1_16u", "A2_16u", "A2_16g", "A3_16u"):
    mono = "" if kern[v]["monotonic"] == "1" else "$^{\\dagger}$"
    row(f"\\texttt{{{v.replace('_', '\\_')}}}{mono}", sci(E(v)), "unbounded$^{a}$", f"f_{v}")
for v in ("R3_16", "R3_32"):
    row(f"\\texttt{{{v.replace('_', '\\_')}}}", "---", f"{float(flt[v]['max_ulp_err']):.2f} ulp", f"f_{v}")
L.append("\\midrule")
b = base[("aor_log2f", "normal")]
row("Arm Opt.\\ Routines \\texttt{log2f}$^{b}$", "---", f"{float(b['max_ulp_err']):.2f} ulp", "b_aor_log2f")
b = base[("fbsd_log2f", "normal")]
row("FreeBSD/fdlibm \\texttt{log2f}", "---", f"{float(b['max_ulp_err']):.2f} ulp", "b_fbsd_log2f")
b = base[("legacy_log2_P3_16", "normal")]
row("2023 version, \\texttt{log2\\_P3\\_16}$^{c}$", sci(float(b["excess_over_half_ulp"])), "unbounded$^{a}$", "l_log2_P3_16")
(out / "tab_cost.tex").write_text("\n".join(L) + "\n")

# ---- Table: Q31 natural log vs CMSIS-DSP
L = []
for name, fn in (("CMSIS-DSP \\texttt{arm\\_vlog\\_q31}", "cmsis_arm_vlog_q31"),
                 ("logq \\texttt{A2\\_16u}", "logq_A2_16u"), ("logq \\texttt{A3\\_16u}", "logq_A3_16u")):
    r = q31[fn]
    e = max(-float(r["err_min"]), float(r["err_max"]))
    m = "b_cmsis_log_q31" if fn.startswith("cmsis") else "ln_" + fn.split("_", 1)[1]
    L.append(f"{name} & {sci(e)} & {sci(float(r['rms_err']))} & {inst(m,'rv32im')} & {inst(m,'m0')} & {foot(m)} \\\\")
(out / "tab_cmsis.tex").write_text("\n".join(L) + "\n")

# ---- numbers used in the text
vals = {
    "nvariants": len(kern),
    "nmono": sum(r["monotonic"] == "1" for r in kern.values()),
    "Emax": sci(max(E(k) for k in kern)), "Emin": sci(min(E(k) for k in kern)),
    "kernrv": inst("k_A2_16u", "rv32im"), "kernmz": inst("k_A2_16u", "m0"),
    "intrv": inst("u_A2_16u", "rv32im"), "intmz": inst("u_A2_16u", "m0"),
    "fixrv": inst("q_A2_16u", "rv32im"), "fixmz": inst("q_A2_16u", "m0"),
}
# ranges quoted in the abstract and conclusions
fast = ["f_A1_16u", "f_A2_16u", "f_A2_16g", "f_A3_16u", "f_R3_16", "f_R3_32"]
libs = ["b_aor_log2f", "b_fbsd_log2f"]
ins = [float(mcu[f][t + "_mean"]) for f in fast for t in ("rv32im", "m0")]
rat = [float(mcu[l][t + "_mean"]) / float(mcu[f][t + "_mean"]) for f in fast for l in libs for t in ("rv32im", "m0")]
vals.update(insmin=f"{min(ins):.0f}", insmax=f"{max(ins):.0f}", ratmin=f"{min(rat):.0f}", ratmax=f"{max(rat):.0f}")
vals["cmsisrat"] = f"{float(mcu['b_cmsis_log_q31']['m0_mean']) / float(mcu['ln_A3_16u']['m0_mean']):.1f}"
vals["cmsisratrv"] = f"{float(mcu['b_cmsis_log_q31']['rv32im_mean']) / float(mcu['ln_A3_16u']['rv32im_mean']):.1f}"
vals.update(eAtwou=sci(E("A2_32u")), eAsixg=sci(E("A2_16g")),
            iAtwou=inst("f_A2_32u", "rv32im"), iAsixg=inst("f_A2_16g", "rv32im"),
            mAtwou=inst("f_A2_32u", "m0"), mAsixg=inst("f_A2_16g", "m0"))
host = {r["function"]: float(r["ns_per_call"]) for r in csv.DictReader(open(R / "host_x86_64.csv"))}
hr = [v / host["libm_log2f"] for k, v in host.items() if k.startswith("logq_")]
vals.update(hostmin=f"{min(hr):.1f}", hostmax=f"{max(hr):.1f}")
(out / "numbers.tex").write_text("".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in vals.items()))
print("written:", ", ".join(p.name for p in sorted(out.iterdir())))
