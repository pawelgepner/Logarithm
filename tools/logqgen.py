#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""logqgen -- generator of integer-only binary logarithm kernels.

Two families are produced.

  A(n, p, s)  piecewise polynomial of degree n on p segments of the significand,
              s = u (uniform segments) or g (geometric segments, 2^(j/p)).
              Result: log2(x) in Q8.23 fixed point, bounded ABSOLUTE error.

  R(n, p)     same segmentation (uniform), but the polynomial approximates
              q(z) = log2(1+z)/z after re-centring the significand around 1.
              Result: binary32, bounded RELATIVE error (no cancellation at x ~ 1).

For every variant the generator
  1. computes the minimax polynomial of each segment (Remez exchange),
  2. rounds the coefficients to Q30 integers,
  3. evaluates the *integer* kernel on all 2^23 significands and re-centres
     the constant term so that the worst-case error is balanced,
  4. reports the exact worst-case error of the emitted integer code.

Usage:
  logqgen.py --all -o include/logq_tables.h --report results/kernel_errors.csv
  logqgen.py -n 2 -p 16 --seg u -o mylog.h          # one self-contained header
  logqgen.py --rel -n 3 -p 16 -o mylog_rel.h
"""
import argparse
import json
import sys
import numpy as np

FRAC = 23                 # fractional bits of binary32 significand / of the Q8.23 result
QC = 30                   # fractional bits of the coefficients
ONE = 1 << FRAC
DEGREES = (1, 2, 3)
SEGMENTS = (1, 2, 4, 8, 16, 32, 64)
REL_SEGMENTS = (4, 8, 16, 32, 64)


# --------------------------------------------------------------------------- Remez
def remez(f, a, b, n, iters=30, grid=4001):
    """Minimax polynomial of degree n for f on [a, b]; returns (coeffs low->high, E)."""
    j = np.arange(n + 2)
    x = 0.5 * (a + b) - 0.5 * (b - a) * np.cos(np.pi * j / (n + 1))
    xs = np.linspace(a, b, grid)
    fs = f(xs)
    c, E = None, 0.0
    for _ in range(iters):
        A = np.empty((n + 2, n + 2))
        for i in range(n + 1):
            A[:, i] = x ** i
        A[:, n + 1] = (-1.0) ** j
        sol = np.linalg.solve(A, f(x))
        c, E = sol[:n + 1], abs(sol[n + 1])
        err = np.polyval(c[::-1], xs) - fs
        # one extremum per interval between sign changes
        sign = np.sign(err)
        sign[sign == 0] = 1
        cuts = np.flatnonzero(np.diff(sign)) + 1
        parts = np.split(np.arange(grid), cuts)
        ext = np.array([p[np.argmax(np.abs(err[p]))] for p in parts])
        if len(ext) < n + 2:
            break
        while len(ext) > n + 2:           # drop the weaker end point
            if abs(err[ext[0]]) < abs(err[ext[-1]]):
                ext = ext[1:]
            else:
                ext = ext[:-1]
        xn = xs[ext]
        emax = np.max(np.abs(err))
        if np.allclose(xn, x, rtol=0, atol=(b - a) * 1e-12) or emax - E < 1e-3 * E:
            x = xn
            break
        x = xn
    return c, E


# --------------------------------------------------------------------------- variants
class Variant:
    def __init__(self, n, p, seg, rel=False):
        self.n, self.p, self.seg, self.rel = n, p, seg, rel
        self.s = p.bit_length() - 1
        assert p == 1 << self.s, "p must be a power of two"
        if rel:
            assert seg == "u" and p >= 4
            self.name = f"R{n}_{p}"
        else:
            self.name = f"A{n}_{p}{seg}"
        if seg == "u":
            self.sh = FRAC - self.s
            self.kb = np.arange(p + 1, dtype=np.int64) << self.sh
        else:
            self.sh = FRAC + 1 - self.s
            self.kb = np.rint((2.0 ** (np.arange(p + 1) / p) - 1.0) * ONE).astype(np.int64)
            self.kb[0], self.kb[-1] = 0, ONE
        self.jt = int(round((np.sqrt(2.0) - 1.0) * p)) if rel else None
        self.C = np.zeros((p, n + 1), dtype=np.int64)
        self.E_theory = 0.0
        self.build()

    # exact target as a function of the integer significand fraction k (float64 array)
    def target(self, k, j):
        m = k / ONE
        if not self.rel:
            return np.log2(1.0 + m)
        if j < self.jt:
            z = m
        else:
            z = (m - 1.0) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.log1p(z) / z / np.log(2.0)
        return np.where(z == 0.0, 1.0 / np.log(2.0), q)

    def horner(self, j, t):
        """Integer Horner evaluation, Q30 result (int64 array)."""
        c = self.C[j]
        a = np.full(t.shape, c[self.n], dtype=np.int64)
        for i in range(self.n - 1, -1, -1):
            a = c[i] + ((a * t) >> self.sh)
        return a

    def build(self):
        scale = float(1 << self.sh)
        for j in range(self.p):
            k0, k1 = int(self.kb[j]), int(self.kb[j + 1])
            f = lambda u, k0=k0, j=j: self.target(k0 + u * scale, j)
            c, E = remez(f, 0.0, (k1 - k0 - 1) / scale, self.n)
            self.E_theory = max(self.E_theory, E)
            self.C[j] = np.rint(c * (1 << QC)).astype(np.int64)
            # re-centre the constant term on the error of the integer evaluation
            t = np.arange(k1 - k0, dtype=np.int64)
            for _ in range(2):
                e = self.horner(j, t) / float(1 << QC) - self.target((k0 + t).astype(np.float64), j)
                self.C[j, 0] -= int(round(0.5 * (e.max() + e.min()) * (1 << QC)))
        assert np.all(np.abs(self.C) < (1 << 31)), "coefficient overflow"

    def kernel_all(self):
        """Emitted kernel on all 2^23 significands: Q23 for family A, Q30 for family R."""
        out = np.empty(ONE, dtype=np.int64)
        for j in range(self.p):
            k0, k1 = int(self.kb[j]), int(self.kb[j + 1])
            a = self.horner(j, np.arange(k1 - k0, dtype=np.int64))
            out[k0:k1] = a if self.rel else (a + (1 << (QC - FRAC - 1))) >> (QC - FRAC)
        return out

    def verify(self):
        k = np.arange(ONE, dtype=np.float64)
        got = self.kernel_all()
        if self.rel:
            ref = np.empty(ONE)
            for j in range(self.p):
                k0, k1 = int(self.kb[j]), int(self.kb[j + 1])
                ref[k0:k1] = self.target(k[k0:k1], j)
            e = (got / float(1 << QC) - ref) / ref           # relative error of q
        else:
            e = got / float(ONE) - np.log2(1.0 + k / ONE)    # absolute error of log2
        return float(e.min()), float(e.max())

    def table_bytes(self):
        b = 4 * self.p * (self.n + 1)
        if self.seg == "g":
            b += 4 * self.p
        return b


# --------------------------------------------------------------------------- C emission
COMMON = r'''
#ifndef LOGQ_COMMON_H
#define LOGQ_COMMON_H
/* SPDX-License-Identifier: MIT
 * logq -- integer-only binary logarithms with verified error bounds.
 * Generated by tools/logqgen.py; do not edit by hand.
 *
 * No floating-point operation is executed by any function in this header:
 * binary32 arguments and results are handled as bit patterns.  Right shifts of
 * negative integers are assumed to be arithmetic (true for GCC, Clang, MSVC,
 * IAR, armcc).
 */
#include <stdint.h>

#define LOGQ_FRAC 23                    /* Q8.23: value = q / 2^23 */
#define LOGQ_ONE  (1 << LOGQ_FRAC)

static inline int logq_clz32(uint32_t v)            /* v != 0 */
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(v);
#else
    int n = 0;
    if (!(v & 0xFFFF0000u)) { n += 16; v <<= 16; }
    if (!(v & 0xFF000000u)) { n += 8;  v <<= 8; }
    if (!(v & 0xF0000000u)) { n += 4;  v <<= 4; }
    if (!(v & 0xC0000000u)) { n += 2;  v <<= 2; }
    if (!(v & 0x80000000u)) { n += 1; }
    return n;
#endif
}

static inline uint32_t logq_f32_bits(float x)
{
    union { float f; uint32_t u; } c; c.f = x; return c.u;
}

static inline float logq_bits_f32(uint32_t b)
{
    union { float f; uint32_t u; } c; c.u = b; return c.f;
}

/* Round-to-nearest-even conversion of the fixed-point value mag * 2^-fb
 * (mag != 0, with sign bit `sign`) to a binary32 bit pattern. */
static inline uint32_t logq_pack32(uint32_t sign, uint32_t mag, int fb)
{
    int lz = logq_clz32(mag);
    uint32_t m = mag << lz;                         /* bit 31 set */
    uint32_t e = (uint32_t)(127 + 31 - fb - lz);
    uint32_t r = (m >> 8) + ((m >> 7) & 1u);        /* round half up ...       */
    if ((m & 0xFFu) == 0x80u) r &= ~1u;             /* ... and ties to even    */
    return sign | ((e << 23) + r - 0x00800000u);    /* r == 2^24 carries into e */
}

static inline uint32_t logq_q23_to_f32bits(int32_t q)
{
    if (q == 0) return 0;
    return q < 0 ? logq_pack32(0x80000000u, (uint32_t)(-q), LOGQ_FRAC)
                 : logq_pack32(0, (uint32_t)q, LOGQ_FRAC);
}

/* Same for a 64-bit fixed-point value s * 2^-54. */
static inline uint32_t logq_fix54_to_f32bits(int64_t s)
{
    uint32_t sign = 0, hi, lo, m, sticky;
    uint64_t a;
    int lz;
    if (s == 0) return 0;
    if (s < 0) { sign = 0x80000000u; a = (uint64_t)(-s); } else a = (uint64_t)s;
    hi = (uint32_t)(a >> 32); lo = (uint32_t)a;
    if (hi) { lz = logq_clz32(hi); } else { lz = 32 + logq_clz32(lo); }
    a <<= lz;                                        /* bit 63 set */
    m = (uint32_t)(a >> 32);
    sticky = ((uint32_t)a != 0);
    {
        uint32_t e = (uint32_t)(127 + 63 - 54 - lz);
        uint32_t r = (m >> 8) + ((m >> 7) & 1u);
        if ((m & 0xFFu) == 0x80u && !sticky) r &= ~1u;
        return sign | ((e << 23) + r - 0x00800000u);
    }
}

/* Classification of a binary32 bit pattern.  Returns 0 and sets (*e, *k) such
 * that x = 2^e * (1 + k/2^23) for positive finite non-zero x (subnormals are
 * normalised); otherwise returns 1 and sets *special to the IEEE 754 result of
 * log2: -inf for +-0, NaN for x < 0 and NaN, +inf for +inf. */
static inline int logq_decompose(uint32_t b, int32_t *e, uint32_t *k, uint32_t *special)
{
    if (b - 0x00800000u >= 0x7F000000u) {
        int sh;
        if ((b << 1) == 0) { *special = 0xFF800000u; return 1; }
        if (b >> 31)       { *special = 0x7FC00000u; return 1; }
        if (b >= 0x7F800000u) { *special = (b == 0x7F800000u) ? b : (b | 0x00400000u); return 1; }
        sh = logq_clz32(b) - 8;
        b <<= sh;
        *e = -126 - sh;
    } else {
        *e = (int32_t)(b >> 23) - 127;
    }
    *k = b & 0x007FFFFFu;
    return 0;
}

/* Wrappers shared by all family-A kernels. */
#define LOGQ_DEFINE_A(NAME)                                                        \
/* log2 of a positive NORMAL binary32 given by its bits; Q8.23 result. */          \
static inline int32_t NAME##_q23(uint32_t bits)                                    \
{                                                                                  \
    return ((int32_t)(bits >> 23) - 127) * LOGQ_ONE + NAME##_kernel(bits & 0x007FFFFFu); \
}                                                                                  \
/* log2 of an unsigned integer v >= 1; Q8.23 result in [0, 32).  Bits below the   \
 * 24 most significant ones of v are truncated. */                                 \
static inline int32_t NAME##_q23_u32(uint32_t v)                                   \
{                                                                                  \
    int lz = logq_clz32(v);                                                        \
    return (31 - lz) * LOGQ_ONE + NAME##_kernel(((v << lz) >> 8) & 0x007FFFFFu);   \
}                                                                                  \
/* log2 over the full binary32 domain, bit patterns in and out. */                 \
static inline uint32_t NAME##_f32bits(uint32_t bits)                               \
{                                                                                  \
    int32_t e; uint32_t k, sp;                                                     \
    if (logq_decompose(bits, &e, &k, &sp)) return sp;                              \
    return logq_q23_to_f32bits(e * LOGQ_ONE + NAME##_kernel(k));                   \
}                                                                                  \
static inline float NAME##_f32(float x)                                            \
{                                                                                  \
    return logq_bits_f32(NAME##_f32bits(logq_f32_bits(x)));                        \
}

/* Wrappers shared by all family-R kernels.  KT is the significand fraction at    \
 * which the argument is re-centred (x/2 and exponent + 1). */
#define LOGQ_DEFINE_R(NAME, KT)                                                    \
static inline uint32_t NAME##_f32bits(uint32_t bits)                               \
{                                                                                  \
    int32_t e; uint32_t k, sp; int64_t p;                                          \
    if (logq_decompose(bits, &e, &k, &sp)) return sp;                              \
    p = (int64_t)NAME##_kernel(k);                         /* q(z) in Q30 */       \
    if (k < (KT)) p = (p * (int32_t)k) * 2;                /* z =  k / 2^23      */ \
    else { p = -(p * (int32_t)(0x00800000u - k)); e += 1; }/* z = (k-2^23)/2^24 */ \
    return logq_fix54_to_f32bits((int64_t)e * ((int64_t)1 << 54) + p);             \
}                                                                                  \
static inline float NAME##_f32(float x)                                            \
{                                                                                  \
    return logq_bits_f32(NAME##_f32bits(logq_f32_bits(x)));                        \
}
#endif /* LOGQ_COMMON_H */
'''


def emit_variant(v, prefix="logq_"):
    name = prefix + v.name
    L = []
    kind = ("relative-error family R" if v.rel else "absolute-error family A")
    L.append(f"/* {v.name}: {kind}, degree {v.n}, {v.p} "
             f"{'geometric' if v.seg == 'g' else 'uniform'} segment(s), "
             f"{v.table_bytes()} bytes of tables. */")
    if v.seg == "g":
        kb = ", ".join(f"0x{int(x):06X}u" for x in v.kb[:-1])
        L.append(f"static const uint32_t {name}_KB[{v.p}] = {{ {kb} }};")
    rows = ",\n".join("    { " + ", ".join(str(int(c)) for c in row) + " }" for row in v.C)
    L.append(f"static const int32_t {name}_C[{v.p}][{v.n + 1}] = {{\n{rows}\n}};")
    L.append(f"static inline int32_t {name}_kernel(uint32_t k)   /* k = 23-bit significand fraction */")
    L.append("{")
    if v.seg == "u":
        if v.p == 1:
            L.append(f"    const int32_t *c = {name}_C[0];")
            L.append("    int32_t t = (int32_t)k, a;")
        else:
            L.append(f"    const int32_t *c = {name}_C[k >> {v.sh}];")
            L.append(f"    int32_t t = (int32_t)(k & 0x{(1 << v.sh) - 1:X}u), a;")
    else:
        L.append("    uint32_t j = 0;")
        L.append("    const int32_t *c;")
        L.append("    int32_t t, a;")
        for b in range(v.s - 1, -1, -1):
            L.append(f"    if (k >= {name}_KB[j + {1 << b}]) j += {1 << b};")
        L.append(f"    c = {name}_C[j];")
        L.append(f"    t = (int32_t)(k - {name}_KB[j]);")
    L.append(f"    a = c[{v.n}];")
    for i in range(v.n - 1, -1, -1):
        L.append(f"    a = c[{i}] + (int32_t)(((int64_t)a * t) >> {v.sh});")
    if v.rel:
        L.append("    return a;                         /* Q30 */")
    else:
        L.append(f"    return (a + {1 << (QC - FRAC - 1)}) >> {QC - FRAC};          /* Q23, rounded */")
    L.append("}")
    if v.rel:
        L.append(f"LOGQ_DEFINE_R({name}, 0x{v.jt << v.sh:06X}u)")
    else:
        L.append(f"LOGQ_DEFINE_A({name})")
    return "\n".join(L) + "\n"


def all_variants():
    vs = []
    for n in DEGREES:
        for p in SEGMENTS:
            vs.append(("A", n, p, "u"))
    for n in DEGREES:
        for p in SEGMENTS[1:]:
            vs.append(("A", n, p, "g"))
    for n in DEGREES:
        for p in REL_SEGMENTS:
            vs.append(("R", n, p, "u"))
    return vs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--degree", type=int, choices=DEGREES)
    ap.add_argument("-p", "--segments", type=int)
    ap.add_argument("--seg", choices=("u", "g"), default="u", help="uniform or geometric segments")
    ap.add_argument("--rel", action="store_true", help="relative-error family R")
    ap.add_argument("--all", action="store_true", help="emit every catalogued variant")
    ap.add_argument("--prefix", default="logq_", help="prefix of the emitted C identifiers")
    ap.add_argument("-o", "--output", default="-", help="C header to write")
    ap.add_argument("--report", help="CSV file with the verified kernel errors")
    ap.add_argument("--json", help="JSON file with coefficients and errors")
    a = ap.parse_args()

    if a.all:
        specs = all_variants()
    else:
        if a.degree is None or a.segments is None:
            ap.error("give -n and -p, or --all")
        specs = [("R" if a.rel else "A", a.degree, a.segments, a.seg)]

    guard = "LOGQ_TABLES_H" if a.all else f"LOGQ_{a.prefix.upper()}GENERATED_H"
    out = [COMMON, f"#ifndef {guard}\n#define {guard}\n"]
    rows, js = [], []
    for fam, n, p, seg in specs:
        v = Variant(n, p, seg, rel=(fam == "R"))
        emin, emax = v.verify()
        out.append(emit_variant(v, a.prefix))
        rows.append((v.name, fam, n, p, seg, v.table_bytes(), v.E_theory, emin, emax))
        js.append(dict(name=v.name, family=fam, degree=n, segments=p, seg=seg, shift=v.sh,
                       table_bytes=v.table_bytes(), E_theory=v.E_theory, err_min=emin, err_max=emax,
                       boundaries=[int(x) for x in v.kb], coeffs=[[int(c) for c in r] for r in v.C]))
        print(f"{v.name:8s} tables {v.table_bytes():5d} B   minimax {v.E_theory:.3e}   "
              f"verified [{emin:+.3e}, {emax:+.3e}] ({'relative, q' if v.rel else 'absolute, log2'})",
              file=sys.stderr)
    if a.all:
        names_a = [r[0] for r in rows if r[1] == "A"]
        names_r = [r[0] for r in rows if r[1] == "R"]
        out.append("#define LOGQ_FOR_EACH_A(X) " + " ".join(f"X({n})" for n in names_a) + "\n")
        out.append("#define LOGQ_FOR_EACH_R(X) " + " ".join(f"X({n})" for n in names_r) + "\n")
    out.append(f"#endif /* {guard} */\n")
    text = "\n".join(out)
    if a.output == "-":
        sys.stdout.write(text)
    else:
        with open(a.output, "w") as f:
            f.write(text)
    if a.report:
        with open(a.report, "w") as f:
            f.write("name,family,degree,segments,segmentation,table_bytes,minimax_error,verified_min,verified_max\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]:.6e},{r[7]:.6e},{r[8]:.6e}\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(js, f, indent=1)


if __name__ == "__main__":
    main()
