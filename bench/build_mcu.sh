#!/bin/sh
# Cross-compile bench/mcu_funcs.c for RV32IM and ARMv6-M (Cortex-M0), both
# without FPU, against the soft-float routines of LLVM compiler-rt.
# Usage: BUILTINS=/path/to/llvm-project/compiler-rt/lib/builtins sh bench/build_mcu.sh
set -e
cd "$(dirname "$0")"
: "${BUILTINS:?set BUILTINS to compiler-rt/lib/builtins}"
OPT="-O2 -ffreestanding -fno-common -ffunction-sections -fdata-sections -I../include -Ifreestanding"
mkdir -p build
for T in rv32im m0; do
  case $T in
    rv32im) TGT="--target=riscv32-unknown-elf -march=rv32im -mabi=ilp32" ;;
    m0)     TGT="--target=thumbv6m-none-eabi -mcpu=cortex-m0 -mfloat-abi=soft" ;;
  esac
  rm -rf build/rt_$T && mkdir -p build/rt_$T
  for f in addsf3 subsf3 mulsf3 divsf3 comparesf2 floatsisf floatunsisf fixsfsi fixunssfsi \
           extendsfdf2 truncdfsf2 adddf3 subdf3 muldf3 divdf3 comparedf2 floatsidf floatunsidf \
           fixdfsi fixunsdfsi muldi3 ashldi3 lshrdi3 ashrdi3 udivdi3 divdi3 umoddi3 moddi3 \
           udivmoddi4 udivsi3 divsi3 umodsi3 modsi3 clzsi2 clzdi2 negdi2 fp_mode; do
    clang $TGT -O2 -ffreestanding -ffunction-sections -fdata-sections -w -c "$BUILTINS/$f.c" -o build/rt_$T/$f.o
  done
  if [ $T = m0 ]; then   # ARM EABI helpers that exist only as assembly or ARM-specific C
    for f in aeabi_uldivmod aeabi_ldivmod aeabi_uidivmod aeabi_idivmod; do
      clang $TGT -c "$BUILTINS/arm/$f.S" -o build/rt_$T/$f.o 2>/dev/null || true
    done
    for f in aeabi_fcmp aeabi_dcmp aeabi_cfcmp aeabi_cdcmp; do
      clang $TGT -c "$BUILTINS/arm/$f.S" -o build/rt_$T/$f.o 2>/dev/null || true
    done
    clang $TGT -O2 -ffreestanding -w -c "$BUILTINS/arm/aeabi_fcmp.S" -o build/rt_$T/aeabi_fcmp.o 2>/dev/null || true
  fi
  llvm-ar rcs build/librt_$T.a build/rt_$T/*.o 2>/dev/null || ar rcs build/librt_$T.a build/rt_$T/*.o
  clang $TGT $OPT -c mcu_funcs.c -o build/mcu_funcs_$T.o
  ld.lld -e _start -Ttext=0x10000 build/mcu_funcs_$T.o build/librt_$T.a \
      -o build/mcu_$T.elf
  # exact footprint: link each measured function alone, keep only what it reaches
  : > build/footprint_$T.txt
  for fn in $(llvm-nm --defined-only build/mcu_funcs_$T.o | awk '$2=="T"{print $3}' | grep -E '^(k|q|u|f|ln|b|l)_|^empty_call$'); do
    ld.lld --gc-sections -e "$fn" -Ttext=0x10000 build/mcu_funcs_$T.o build/librt_$T.a -o build/one.elf
    llvm-size -A build/one.elf | awk -v fn="$fn" '$1 ~ /^\.(text|rodata|srodata|sdata|data)/ {s+=$2} END {print fn, s}' >> build/footprint_$T.txt
  done
  rm -f build/one.elf
  echo "built build/mcu_$T.elf"
done
