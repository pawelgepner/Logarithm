/* SPDX-License-Identifier: MIT
 * Bit-for-bit comparison of logq R3_32 with the Arm Optimized Routines log2f
 * (musl, glibc) over every positive finite binary32 input. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "logq.h"
#include "../baselines/aor_log2f.c"
int main(void){
  int64_t b; long long diff=0, diff1=0, n=0;
  #pragma omp parallel for reduction(+:diff,diff1,n) schedule(static,1<<20)
  for(b=1;b<0x7F800000;b++){ float x; uint32_t u=(uint32_t)b; memcpy(&x,&u,4);
    float a=aor_log2f(x); uint32_t ab; memcpy(&ab,&a,4);
    uint32_t r=logq_R3_32_f32bits(u); n++;
    if(r!=ab){ diff++; int32_t d=(int32_t)(r-ab); if(d==1||d==-1) diff1++; } }
  printf("inputs %lld, differ %lld (%.4f%%), of which by 1 ulp %lld\n", n, diff, 100.0*diff/n, diff1);
}
