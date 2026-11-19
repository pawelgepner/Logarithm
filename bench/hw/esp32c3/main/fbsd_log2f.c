/* FreeBSD msun log2f (derived from Sun fdlibm), the binary32 log2 used by
 * FreeBSD, Bionic and other fdlibm-derived libraries.  Vendored from
 * https://github.com/freebsd/freebsd-src, lib/msun/src/e_log2f.c and
 * lib/msun/src/k_logf.h.  The function bodies are unchanged; the libm internal
 * headers were replaced by the minimal definitions below, the function was
 * renamed fbsd_log2f, and the float_t branch is fixed to the binary32 case.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
#include <stdint.h>
typedef union { float f; uint32_t w; } fbsd_shape;
#define GET_FLOAT_WORD(i,d) do { fbsd_shape gf_u; gf_u.f = (d); (i) = (int32_t)gf_u.w; } while (0)
#define SET_FLOAT_WORD(d,i) do { fbsd_shape sf_u; sf_u.w = (uint32_t)(i); (d) = sf_u.f; } while (0)
static const float
/* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
Lg1 =      0xaaaaaa.0p-24,	/* 0.66666662693 */
Lg2 =      0xccce13.0p-25,	/* 0.40000972152 */
Lg3 =      0x91e9ee.0p-25,	/* 0.28498786688 */
Lg4 =      0xf89e26.0p-26;	/* 0.24279078841 */

static inline float
k_log1pf(float f)
{
	float hfsq,s,z,R,w,t1,t2;

 	s = f/((float)2.0+f);
	z = s*s;
	w = z*z;
	t1= w*(Lg2+w*Lg4);
	t2= z*(Lg1+w*Lg3);
	R = t2+t1;
	hfsq=(float)0.5*f*f;
	return s*(hfsq+R);
}

static const float
two25      =  3.3554432000e+07, /* 0x4c000000 */
ivln2hi    =  1.4428710938e+00, /* 0x3fb8b000 */
ivln2lo    = -1.7605285393e-04; /* 0xb9389ad4 */

static const float zero   =  0.0;
static volatile float vzero = 0.0;

static float
fbsd_log2f(float x)
{
	float f,hfsq,hi,lo,r,y;
	int32_t i,k,hx;

	GET_FLOAT_WORD(hx,x);

	k=0;
	if (hx < 0x00800000) {			/* x < 2**-126  */
	    if ((hx&0x7fffffff)==0)
		return -two25/vzero;		/* log(+-0)=-inf */
	    if (hx<0) return (x-x)/zero;	/* log(-#) = NaN */
	    k -= 25; x *= two25; /* subnormal number, scale up x */
	    GET_FLOAT_WORD(hx,x);
	}
	if (hx >= 0x7f800000) return x+x;
	if (hx == 0x3f800000)
	    return zero;			/* log(1) = +0 */
	k += (hx>>23)-127;
	hx &= 0x007fffff;
	i = (hx+(0x4afb0d))&0x800000;
	SET_FLOAT_WORD(x,hx|(i^0x3f800000));	/* normalize x or x/2 */
	k += (i>>23);
	y = (float)k;
	f = x - (float)1.0;
	hfsq = (float)0.5*f*f;
	r = k_log1pf(f);

	hi = f - hfsq;
	GET_FLOAT_WORD(hx,hi);
	SET_FLOAT_WORD(hi,hx&0xfffff000);
	lo = (f - hi) - hfsq + r;
	return (lo+hi)*ivln2lo + lo*ivln2hi + hi*ivln2hi + y;
}
