// C++ program to find out execution time of
// Logarithm of functions

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "half.hpp"
#include <cstdint>

using namespace std;

using half = half_float::half;

#if defined(__aarch64__) || defined(__arm__)
static __inline__ unsigned long long readcyclecounter()
{
    unsigned long long result;
    __asm __volatile("mrs       %0, CNTVCT_EL0" : "=&r" (result));
    return result;
}
#else
static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
static __inline__ unsigned long long readcyclecounter()
{
#if __clang__s
    return __builtin_readcyclecounter();
#else
    return rdtsc();
#endif
}
#endif



/************ log2f_musl (ARM LOG2) wraz z zaleźnościami *************************************************************************************/

#ifdef __GNUC__
#define predict_true(x) __builtin_expect(!!(x), 1)
#define predict_false(x) __builtin_expect(x, 0)
#else
#define predict_true(x) (x)
#define predict_false(x) (x)
#endif

typedef union{float _f; uint32_t _i; } conversion_union_asuint;
typedef union{uint32_t _i; float _f; } conversion_union_asfloat;

#define asuint(f) ((conversion_union_asuint{f})._i)
#define asfloat(i) ((conversion_union_asfloat{i})._f)

#define WANT_ROUNDING 1

static inline float eval_as_float(float x)
{
	float y = x;
	return y;
}

#define LOG2N (1 << LOG2F_TABLE_BITS)
#define LOG2T __log2f_data.tab
#define LOG2A __log2f_data.poly
#define LOG2OFF 0x3f330000

#define LOG2F_TABLE_BITS 4
#define LOG2F_POLY_ORDER 4
static const struct log2f_data {
	struct {
		double invc, logc;
	} tab[1 << LOG2F_TABLE_BITS] = {};
	double poly[LOG2F_POLY_ORDER];
} __log2f_data = {
  .tab = {
  { 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2 },
  { 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2 },
  { 0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2 },
  { 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2 },
  { 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2 },
  { 0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3 },
  { 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3 },
  { 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4 },
  { 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5 },
  { 0x1p+0, 0x0p+0 },
  { 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4 },
  { 0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3 },
  { 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3 },
  { 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2 },
  { 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2 },
  { 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2 },
  },
  .poly = {
  -0x1.712b6f70a7e4dp-2, 0x1.ecabf496832ep-2, -0x1.715479ffae3dep-1,
  0x1.715475f35c8b8p0,
  }
};

float log2f_musl(float x)
{
	double_t z, r, r2, p, y, y0, invc, logc;
	uint32_t ix, iz, top, tmp;
	int k, i;

	ix = asuint(x);
	/* Fix sign of zero with downward rounding when x==1.  */
	if (WANT_ROUNDING && predict_false(ix == 0x3f800000))
		return 0;
	if (predict_false(ix - 0x00800000 >= 0x7f800000 - 0x00800000)) {
		/* x < 0x1p-126 or inf or nan.  */
		if (ix * 2 == 0)
			return 1;
		if (ix == 0x7f800000) /* log2(inf) == inf.  */
			return x;
		if ((ix & 0x80000000) || ix * 2 >= 0xff000000)
			return x;
		/* x is subnormal, normalize it.  */
		ix = asuint(x * 0x1p23f);
		ix -= 23 << 23;
	}

	/* x = 2^k z; where z is in range [OFF,2*OFF] and exact.
	   The range is split into N subintervals.
	   The ith subinterval contains z and c is near its center.  */
	tmp = ix - LOG2OFF;
	i = (tmp >> (23 - LOG2F_TABLE_BITS)) % LOG2N;
	top = tmp & 0xff800000;
	iz = ix - top;
	k = (int32_t)tmp >> 23; /* arithmetic shift */
	invc = LOG2T[i].invc;
	logc = LOG2T[i].logc;
	z = (double_t)asfloat(iz);

	/* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k */
	r = z * invc - 1;
	y0 = logc + (double_t)k;

	/* Pipelined polynomial evaluation to approximate log1p(r)/ln2.  */
	r2 = r * r;
	y = LOG2A[1] * r + LOG2A[2];
	y = LOG2A[0] * r2 + y;
	p = LOG2A[3] * r + y0;
	y = y * r2 + p;
	return eval_as_float(y);
}

/************ log2f_fsf (Free Software Foundation,) wraz z zaleźnościami *************************************************************************************/
/* Base 2 logarithm.
   Copyright (C) 2012-2016 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */


/* Best possible approximation of log(2) as a 'float'.  */
#define LOG2 0.693147180559945309417232121458176568075f

/* Best possible approximation of 1/log(2) as a 'float'.  */
#define LOG2_INVERSE 1.44269504088896340735992468100189213743f

/* sqrt(0.5).  */
#define SQRT_HALF 0.707106781186547524400844362104849039284f

float log2f_fsf (float x)
{

    if (x <= 0.0f)
    {
        if (x == 0.0f)
            /* Return -Infinity.  */
            return - HUGE_VALF;
        else
        {
            /* Return NaN.  */
#if defined _MSC_VER
            static float zero;
          return zero / zero;
#else
            return 0.0f / 0.0f;
#endif
        }
    }

    /* Decompose x into
         x = 2^e * y
       where
         e is an integer,
         1/2 < y < 2.
       Then log2(x) = e + log2(y) = e + log(y)/log(2).  */
    {
        int e;
        float y;

        y = frexpf (x, &e);
        if (y < SQRT_HALF)
        {
            y = 2.0f * y;
            e = e - 1;
        }
        return (float) e + logf (y) * LOG2_INVERSE;
    }
}
/************ log2f_Sun (Copyright (C) 1993 by Sun Microsystems, Inc)*************************************************************************************/
static const float
        ivln2hi =  1.4428710938e+00, /* 0x3fb8b000 */
ivln2lo = -1.7605285393e-04, /* 0xb9389ad4 */
/* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
Lg1 = 0xaaaaaa.0p-24, /* 0.66666662693 */
Lg2 = 0xccce13.0p-25, /* 0.40000972152 */
Lg3 = 0x91e9ee.0p-25, /* 0.28498786688 */
Lg4 = 0xf89e26.0p-26; /* 0.24279078841 */
float log2f_Sun(float x)
{
    union {float f; uint32_t i;} u = {x};
    float_t hfsq,f,s,z,R,w,t1,t2,hi,lo;
    uint32_t ix;
    int k;
    ix = u.i;
    k = 0;
    if (ix < 0x00800000 || ix>>31) {  /* x < 2**-126  */
        if (ix<<1 == 0)
            return -1/(x*x);  /* log(+-0)=-inf */
        if (ix>>31)
            return (x-x)/0.0f; /* log(-#) = NaN */
        /* subnormal number, scale up x */
        k -= 25;
        x *= 0x1p25f;
        u.f = x;
        ix = u.i;
    } else if (ix >= 0x7f800000) {
        return x;
    } else if (ix == 0x3f800000)
        return 0;
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix += 0x3f800000 - 0x3f3504f3;
    k += (int)(ix>>23) - 0x7f;
    ix = (ix&0x007fffff) + 0x3f3504f3;
    u.i = ix;
    x = u.f;
    f = x - 1.0f;
    s = f/(2.0f + f);
    z = s*s;
    w = z*z;
    t1= w*(Lg2+w*Lg4);
    t2= z*(Lg1+w*Lg3);
    R = t2 + t1;
    hfsq = 0.5f*f*f;
    hi = f - hfsq;
    u.f = hi;
    u.i &= 0xfffff000;
    hi = u.f;
    lo = f - hi - hfsq + s*(hfsq+R);
    return (lo+hi)*ivln2lo + lo*ivln2hi + hi*ivln2hi + k;
}




/***************  Mitchell’s approximation *******************/

float log2_Mitchell (float x)
       {
       int i = *(int*) & x;
       int m = i & 0x007fffff;
       i = i >> 23;
       i = i-128;
       m = 0x3f800000|m;
       float y = *(float*)&m;
       y = y + i;
      return y;
      }

/*** Porównać czas wykonania kodów i dokładność obliczeń kastomizowalnych funkcji log2 z funkcją  biblioteczną log2(x).****/

using namespace std::chrono;
const long int iterationsCount = 1000000;

/************ Moja  Sekcja Kastomizacja *************************************************************************************/


const float c0[16]={ 28.08643958496275310, 2481.79918407803395466, 10738.26263631644198492 , 26559.90277794395061309, 50703.04540598535745020 , 83314.30394082373718523, 124177.80596893438398803, 172870.44063962133995202, 228859.56877529498024772, 291564.32130631803213663, 360393.65036781315964937, 434769.44895098114900368, 514140.05339695906126626, 597987.55750327042410695, 685831.16725886037100471, 777228.053181107675861950 };

const float c1[16]={ 1.441715908186118885748512, 1.432422668534872227213828, 1.416654217103422772782177, 1.396499665972209649817729, 1.373437359304065211603052, 1.348522031692466705603616, 1.322510538593386612530772, 1.295947748637332714715400,  1.269226104436044368311995, 1.242627479499152257736017, 1.216352947046256264802108, 1.190544179057978977097526, 1.165298976176873338151648, 1.140682634215640921525177,
                     1.116736326088209081268105, 1.093483323624940245577465 };

const float c2[16]={ 0.8090513123507357188709764e-7, 0.7191842463818404206094718e-7, 0.6435013821808266151970200e-7, 0.5791671119786000916030592e-7,
                     0.5240206461407253345606026e-7, 0.4763920685160663517714986e-7,
                     0.4349743435772322947105054e-7, 0.3987326357553934860761498e-7, 0.3668390075023117877562460e-7,
                     0.3386246911013949403724182e-7, 0.3135447295611903177961426e-7,
                     0.2911514540536952366492312e-7, 0.2710743611796251136673654e-7, 0.2530046840579073458796554e-7, 0.2366834463384333916579162e-7,
                     0.2218921286814984187176792e-7};


	float log2_P2_1(float x)
	{
   int i =  *(int*) & x;
	    int k = i & 0x007fffff;
	    i = i >> 23;
	    i = i - 128;
	    k = 41437.706985f + k*(1.334968914f - 4.11091368e-8f*k);
        k = 0x3f800000|k;
	    float y = *(float*) & k;
	    y = y + i;
	    return y;
	}



	float log2_P2_2(float x)
	{
	    float k1=364861.f;
	    float k2=1.2080635f;
	    float k3=0.30062421e-7f;
	    float k4=5230.6446f;
	    float k5=1.4130103f;
	    float k6=0.60124843e-7f;
	    int i = *(int*) & x;
	    int k = i & 0x007fffff;
	    i = i >> 23;
	    i = i - 128;
	    if (k > 0x003504f3)
	    k = k1 + k*(k2 -k3*k);
	    else
	    k = k4 + k*(k5 - k6*k);
	    k = 0x3f800000 | k;
	    float y = *(float*) & k;
	    y = y + i;
	    return y;
	}
float log2_P2_int_2(float x)
{
    int m, s = 23;
    int i= *(int*) & x;
    int k = i&0x007fffff;
    i = i >> s;
    i = i - 128;
    if (k > 0x003504f3){
        m = 10133971 - int((2115455*(int64_t)k) >> s);
        m = 364861 + int(( m*(int64_t)k) >> s);
    }
    else {
        m = 11853190 - int((4230910*(int64_t)k) >> s);
        m = 5231 + int(( m*(int64_t)k) >> s);
    }
    m = 0x3f800000|m;
    float y = *(float*) & m;
    y = y + i;
    return y;
}




float log2_P2_16(float x)
{
    int i =  *(int*) & x;
    int k = i & 0x007fffff;
    int j =k>>19;
    k = c0[j] + k*(c1[j] - c2[j]*k);
    k = 0x3f800000|k;
    float y = *(float*) & k;
    y = y + logbf(x)-1;
    return y;
}



const float ccc0[16]={ 0.31924307023246944, 155.67226882707465920, 1085.30689496711314018, 3593.03459793756373288, 8427.39690209945727796, 16193.91715583814827171, 27337.15929491286300592, 42152.66976593109637502, 60809.71482635141782701, 83375.96596615216182787, 109840.31702485694926995, 140132.45566615428886409, 174138.95503885357950824, 211716.14882586156693250, 252700.23341032377503655, 296915.07140786108799888};

const float ccc1[16]={ 1.442675315718795721939230, 1.441839729054538557191257, 1.439211333127110200633627, 1.434445351960698453281788, 1.427537753618031153144046, 1.418651845905544985500674, 1.408023285720251082685220, 1.395908730746511338707584,  1.382558965274269479313898, 1.368205766809496359060756, 1.353056432277055790762516, 1.337292492165651146321062, 1.321070625201178556764516, 1.304524639491434633618822, 1.287767880425064599518714,
                     1.270895713068600139602232};

const float ccc2[16]={ -0.8580042021106645570946493e-7,
                     -0.8424272456613317275460877e-7,
                     -0.8173968453984285623085584e-7,
                     -0.7870632437036466016692653e-7,
                     -0.7540771599262967678398649e-7,
                     -0.7201315911712359446992437e-7,
                     -0.6863001345918534618178014e-7,
                     -0.6532519522424452158561334e-7,
                     -0.6213904177295838630357230e-7,
                     -0.5909436890875064018917849e-7,
                     -0.5620245122601525535767548e-7,
                     -0.5346700507837163819743077e-7,
                     -0.5088685874138671396093654e-7,
                     -0.4845775030096196592612759e-7,
                     -0.4617354052967327664160751e-7,
                     -0.4402703030952421779053363e-7};

const float ccc3[16]={ 0.6236496113176763380873706e-14,
                     0.5226831099995097972215548e-14, 0.4423885152321667016164122e-14, 0.3777344945544397479101326e-14,
                     0.3250906096100919018886220e-14, 0.2817925400219220024183834e-14,
                     0.2458549384943245308258386e-14, 0.2157778293551186136890292e-14, 0.1904136154099956253133662e-14,
                     0.1688741550549467150847441e-14, 0.1504648226830272441623219e-14,
                     0.1346370475233031750875281e-14, 0.1209537033468025864197879e-14, 0.1090635631852735365244454e-14, 0.9868223300921568668538903e-15,
                     0.8957777274852012677831556e-15};

float log2_P3_16(float x)
{
    int i =  *(int*) & x;
    int k = i & 0x007fffff;
    int j =k>>19;
    i = i >> 23;
    i = i - 128;
    k = ccc0[j] + k*(ccc1[j] + k*(ccc2[j] + ccc3[j]*k));
    k = 0x3f800000|k;
    float y = *(float*) & k;
    y = y + i;
    return y;
}







const float qc0[16]={ 10.24913830398498302, 969.36375748475040126, 4594.16867008911526946, 12391.87264030057103012, 25671.31157745556236926, 45562.51953886833803056, 73034.54369796714643717, 108911.65460604154285286, 153888.09027949819247570, 208541.46092020790495697, 273344.93033489519189553, 348678.28027860712862241, 434837.95493195460176300 , 532046.17446315338106245, 640459.19906074577843735, 760174.81789335398190226};
const float qc1[16]={ 1.442192863030664966486315, 1.437127793794972162575093, 1.427623121946687424325319, 1.414253363882408774690011, 1.397536617729892224336771, 1.377939616202582169334445, 1.355882344385328281315065, 1.331742259225968074092216, 1.30585814442708001647902,  1.278533631608692992024091, 1.250040416024705402394836, 1.220621192743628257242942, 1.190492337030056375026458, 1.159846350670518501583277, 1.128854094160957885760970, 1.097666822999202067709429};

const float qc2[16]={ 0.8233108837041027789487901e-7,  0.7549794091710731166587100e-7,
                     0.6923191707461478922395292e-7,  0.6348594787623227612973456e-7,  0.5821687088918600796832076e-7, 0.5338510598810789295903100e-7,  0.4895435803800484316296464e-7,  0.4489134425333953759304120e-7,  0.4116554418520513894743326e-7,  0.3774897045855365583300894e-7,  0.3461595853730739461218970e-7, 0.3174297393811613806495342e-7,  0.2910843544459300398426600e-7,  0.2669255299405394647963636e-7, 0.2447717901900242158155188e-7,  0.2244567212666976879644878e-7};

float log2_P2_16_neq(float x)
{
    int i, j, k;
    i = *(int*)&x;
    k = i & 0x007fffff;
    if (k >= 0x003504f3) {
        if (k >= 0x005744fd) {
            if (k >= 0x006ac0c7) {
                if (k >= 0x0075257d) { j = 15; }
                else { j = 14; }
            }
            else {
                if (k >= 0x0060ccdf) { j = 13; }
                else { j = 12; }
            }
        }
        else {
            if (k >= 0x0045672a) {
                if (k >= 0x004e248c) { j = 11; }
                else { j = 10; }
            }
            else {
                if (k >= 0x003d08a4) { j = 9; }
                else { j = 8; }
            }
        }
    }
    else {
        if (k >= 0x001837f0) {
            if (k >= 0x0025fed7) {
                if (k >= 0x002d583f) { j = 7; }
                else { j = 6; }
            }
            else {
                if (k >= 0x001ef532) { j = 5; }
                else { j = 4; }
            }
        }
        else {
            if (k >= 0x000b95c2) {
                if (k >= 0x0011c3d3) { j = 3; }
                else { j = 2; }
            }
            else {
                if (k >= 0x0005aac3) { j = 1; }
                else { j = 0; }
            }
        }
    }
    i = i >> 23;
    i = i - 128;
    k = qc0[j] + k * (qc1[j] - qc2[j] * k);
    k = 0x3f800000 | k;
    float y = *(float*)&k;
    y = y + i;
    return y;
}
const float wc0[16]={ 1419.528424590505237226193, 23647.61959369672503228524, 67161.40379000605428393024, 131058.4374611078345829846, 214474.5377005683442462428, 316582.1601220630228828810, 436588.8455063237836178194, 573735.7323051596761336843, 727296.1322104270476605118, 896574.1661142022402936406, 1080903.457899767962427084, 1279645.883611576559992505, 1492190.373656303248678945, 1717951.765786645071810364, 1956369.706714843767995227, 2228405.701036860667205243};

const float wc1[16]={ 1.411670667679404905080658, 1.351820462635744052147120, 1.294507709935452100163147, 1.239624829923786491315889, 1.187068803970429332376462, 1.136740981097015135245834, 1.088546892803016984507967, 1.042396075742407631027099, 0.9982019019182484107520750,  0.9558814160764705129927412, 0.9153551799936264609323176, 0.8765471233663300844137712, 0.8393844010224950510788424, 0.8037972561863484369006076, 0.7697188895405591350503832, 0.7343528627113269964211888};

float log2_P1_16_neq(float x)
{
    int i,j,k;
    i = *(int*)&x;
    k = i & 0x007fffff;
    if (k >= 0x003504f3){
        if (k >= 0x005744fd){
            if (k >= 0x006ac0c7) {
                if (k >= 0x0075257d){j=15;}
                else { j=14; }
            } else {
                if (k >= 0x0060ccdf) {j=13;}
                else{j=12;}
            }
        } else {
            if (k >= 0x0045672a){
                if (k >= 0x004e248c){j=11;}
                else{j=10;}
            }else{
                if (k >= 0x003d08a4){j=9;}
                else{j=8;}
            }
        }
    }else{
        if (k >= 0x001837f0){
            if (k >= 0x0025fed7){
                if (k >= 0x002d583f){j=7;}
                else{j=6;}
            }else{
                if (k >= 0x001ef532){j=5;}
                else{j=4;}
            }
        }else{
            if (k >= 0x000b95c2){
                if (k >= 0x0011c3d3){j=3;}
                else{j=2;}
            }else{
                if (k >= 0x0005aac3){j=1;}
                else{j=0;}
            }
        }
    }
    i = i>> 23;
    i = i - 128;
    k = wc0[j] + k*wc1[j];
    k = 0x3f800000|k;
    float y = *(float*) & k;
    y = y + i;
    return y;
}





const float ec0[16]={ 2779.843671923950244876780, 44419.82335789006194187698, 118980.2655843728035134256, 219471.0990477012814424409, 340455.1822886087024461511, 477660.5281835939438767217, 627700.7326343952757795072, 787870.3381460258245978651, 955992.9860346542965891405, 1130307.335257767690178637, 1309380.382952045976585259, 1492040.923036912248624498, 1677327.979682627137081625, 1864450.497254609766281531, 2052755.576443561882058683, 2257654.092213512721757943};

const float ec1[16]={ 1.399405460005430532065057, 1.319394563071567274454592, 1.248040192020370094176672, 1.184009303100429665228800, 1.126229246262367056406224, 1.073827133736591476858464, 1.026085398715449857516560, 0.9824087146262929465534559, 0.9422990248570962285904000,  0.9053364538618794344987839, 0.8711645443580221434304640, 0.8394787183061690092920480, 0.8100171691194882124448640, 0.7825536076951425376685441, 0.7568914364537068717130560, 0.7308666596158131692698075};

float log2_P1_16_eq(float x)
{
    int i =  *(int*) & x;
    int k = i & 0x007fffff;
    int j =k>>19;
    i = i >> 23;
    i = i-128;
    k = ec0[j] + k*ec1[j] ;
    k = 0x3f800000|k;
    float y = *(float*) & k;
    y = y + i;
    return y;
}

half log2h_P2_int_2(half x)
{
    int m, s=10;
    int16_t i = *(int*) & x;
    int k = i & 0x03ff;
    i = i >> s;
    i = i - 16;
    if (k > 0x01a8){
        m=1237 - (int)((258*(int32_t)k)>>s);
        m=44+(int)(( m*(int32_t)k)>>s);
    }
    else {
        m=1447 - (int)((516*(int32_t)k)>>s);
        m=1+(int)((m*(int32_t)k)>>s);
    }
    m = 0x3c00|m;
    half y = *(half*) & m;
    y = y + i;
    return y;
}



/********************************************************************************************************************/
/* Best possible approximation of log(2) as a 'float'.  */
#define LOG2 0.693147180559945309417232121458176568075f

/* Best possible approximation of 1/log(2) as a 'float'.  */
#define LOG2_INVERSE 1.44269504088896340735992468100189213743f

/* sqrt(0.5).  */
#define SQRT_HALF 0.707106781186547524400844362104849039284f

float log2f_fsf_1(float x)
{

    if (x <= 0.0f)
    {
        if (x == 0.0f)
            /* Return -Infinity.  */
            return - HUGE_VALF;
        else
        {
            /* Return NaN.  */
#if defined _MSC_VER
            static float zero;
          return zero / zero;
#else
            return 0.0f / 0.0f;
#endif
        }
    }

    /* Decompose x into
         x = 2^e * y
       where
         e is an integer,
         1/2 < y < 2.
       Then log2(x) = e + log2(y) = e + log(y)/log(2).  */
    {
        int e;
        float y;

        y = frexpf (x, &e);
/*       if (y < SQRT_HALF)
        {
            y = 2.0f * y;
            e = e - 1;
        }*/
        return (float) e + logf (y) * LOG2_INVERSE;
    }
}



/******************************** koniec sekcji ***********************************************************/


template<class T, class F>
void profileExp(const std::string &desc, long int iterations, T& testVal, F func) {
    auto iter = iterations;
    while (iter--)
        func(testVal);
    iter = iterations;
    auto start = high_resolution_clock::now();
    while (iter--)
        func(testVal);
    auto stop = high_resolution_clock::now();
    auto time_eval = duration_cast<nanoseconds>(stop - start);
    std::cout << std::left << std::setw(35) << desc
              << time_eval.count() / iterations << " ns" << std::endl;
}


template<class T, class F>
void profileExpCycles(const std::string &desc, long int iterations, T &testVal, F func) {
    auto iter = iterations;
    while (iter--)
        func(testVal);
    iter = iterations;
    unsigned long long t = readcyclecounter();
    while (iter--)
        func(testVal);
    t = readcyclecounter() - t;
    std::cout << std::left << std::setw(35) << desc << t / (double)iterations
              << " cycles" << std::endl;
}

template<class T, class F>
void calculateExpRMSD(const std::string &desc, F func)
{
    T currentValue = T(1.1754944e-38f);
    T maxValue = T(1e+38f);
    int count = 0;
    double RMSD = 0;

    double maxPoh = 0.0, maxValx = 0.0, maxValy = 0.0;
    double minPoh = 0.0, minValx = 0.0, minValy = 0.0;

    while (currentValue < maxValue) {
        currentValue = nextafterf(currentValue, maxValue);
        count++;

        auto ref = log2(currentValue);
        auto cal = func(currentValue);
        double vidn = (double)(double(cal)- double(ref)) ;              /*** błąd bezwzględny - Absolut Error  ****/
        maxPoh = std::max(maxPoh, vidn);
        minPoh = std::min(minPoh, vidn);
        RMSD += std::pow(ref - cal, 2);

    }
    std::streamsize base = std::cout.precision();
    std::cout << std::left
              << std::setw(35) << desc
              << std::setw(17) << std::setprecision(base) << std::scientific << sqrtf(RMSD / count)

              << std::setw(24) << std::setprecision(base) << std::scientific << maxPoh
              << std::setw(35) << std::setprecision(base) << std::scientific << minPoh << std::endl;
}
template<class T, class F>
void calculateExpRMSD_half(const std::string &desc, F func) {
    T currentValue = T(0.000244140625);
    T maxValue = T(65504);
    int count = 0;
    double RMSD = 0;

    double maxPoh = 0.0, maxValx = 0.0, maxValy = 0.0;
    double minPoh = 0.0, minValx = 0.0, minValy = 0.0;


    while (currentValue < maxValue) {
        currentValue = half_float::nextafter(currentValue, maxValue);
        count++;

        auto ref = log2(currentValue);
        auto cal = func(currentValue);
        double vidn = (double) (double(cal) - double(ref));              /*** błąd bezwzględny - Absolut Error  ****/
        /* double vidn = (double)(double(cal) / double(ref)) - 1; */   /**** relative Error ****/
        /*double vidn = (double)cal / log2f(currentValue) - 1;*/      /**** relative Error ****/
        maxPoh = std::max(maxPoh, vidn);
        minPoh = std::min(minPoh, vidn);
        RMSD += std::pow(ref - cal, 2);
        //std::cout <<  std::setprecision(20) << currentValue <<std::endl;//
    }
    std::streamsize base = std::cout.precision();
    std::cout << std::left
              << std::setw(35) << desc
              << std::setw(17) << std::setprecision(base) << std::scientific << sqrtf(RMSD / count)
              //<< std::setw(16) << std::setprecision(4) << std::fixed << -log2(maxPoh)
              //<< std::setw(16) << std::setprecision(4) << std::fixed << -log2(sqrtf(RMSD / count) )
              << std::setw(24) << std::setprecision(base) << std::scientific << maxPoh
              << std::setw(35) << std::setprecision(base) << std::scientific << minPoh << std::endl;
}
int main() {
    float step = nextafterf(1.1754944e-38f, 1e+38f)-1.1754944e-38f ;
    half steph = half(step);


    std::cout << std::endl << "Step: " << step << " " << std::endl;

    std::cout << "Execution time:" << std::endl;

    profileExp("log2_Mitchell", iterationsCount, step, log2_Mitchell);
    profileExp("log2_P2_1", iterationsCount, step, log2_P2_1);
    profileExp("log2_P2_2", iterationsCount, step, log2_P2_2);
    profileExp("log2h_P2_int_2", iterationsCount, steph, log2h_P2_int_2);
    profileExp("log2_P2_int_2", iterationsCount, step, log2_P2_int_2);
    profileExp("log2_P1_16_neq", iterationsCount, step, log2_P1_16_neq);
    profileExp("log2_P1_16_eq", iterationsCount, step, log2_P1_16_eq);
    profileExp("log2_P2_16_neq", iterationsCount, step, log2_P2_16_neq);
    profileExp("log2_P2_16_eq", iterationsCount, step, log2_P2_16);
    profileExp("log2_P3_16_eq", iterationsCount, step, log2_P3_16);
    profileExp("log2_musl", iterationsCount, step, log2f_musl);
    profileExp("log2_fsf", iterationsCount, step, log2f_fsf);
    profileExp("log2_fsf_1", iterationsCount, step, log2f_fsf_1);
    profileExp("log2_Sun", iterationsCount, step, log2f_Sun);
    profileExp("std_math::log2f", iterationsCount, step, log2f);
    profileExp<float, float(float)>("std_math::log2", iterationsCount, step, std::log2);


    std::cout << std::endl << "Number of cycles:" << std::endl;

    profileExpCycles("log2_Mitchell", iterationsCount, step, log2_Mitchell);
    profileExpCycles("log2_P2_1", iterationsCount, step, log2_P2_1);
    profileExpCycles("log2_P2_2", iterationsCount, step, log2_P2_2);
    profileExpCycles("log2h_P2_int_2", iterationsCount, steph, log2h_P2_int_2);
    profileExpCycles("log2_P2_int_2", iterationsCount, steph, log2_P2_int_2);
    profileExpCycles("log2_P1_16_neq", iterationsCount, step, log2_P1_16_neq);
    profileExpCycles("log2_P1_16_eq", iterationsCount, step, log2_P1_16_eq);
    profileExpCycles("log2_P2_16_neq", iterationsCount, step, log2_P2_16_neq);
    profileExpCycles("log2_P2_16_eq", iterationsCount, step, log2_P2_16);
    profileExpCycles("log2_P3_16_eq", iterationsCount, step, log2_P3_16);
    profileExpCycles("log2_musl", iterationsCount, step, log2f_musl);
    profileExpCycles("log2_fsf", iterationsCount, step, log2f_fsf);
    profileExpCycles("log2_fsf_1", iterationsCount, step, log2f_fsf_1);
    profileExpCycles("log2_Sun", iterationsCount, step, log2f_Sun);
    profileExpCycles("std_math::log2f", iterationsCount, step, log2f);
    profileExpCycles<float, float(float)>("std_math::log2", iterationsCount, step, std::log2);

    std::cout << std::endl << std::left
        << std::setw(35) << "NAME"
              << std::setw(17) << "RMSD"
              << std::setw(17) << "MAX ABSOLUT ERROR"
              << std::setw(35) << "   MIN ABSOLUT ERROR" << std::endl;


    calculateExpRMSD<float>("log2_Mitchell", log2_Mitchell);
    calculateExpRMSD<float>("log2_P2_1", log2_P2_1);
    calculateExpRMSD<float>("log2_P2_2", log2_P2_2);
    calculateExpRMSD_half<half>("log2h_P2_int_2", log2h_P2_int_2);
    calculateExpRMSD<float>("log2_P2_int_2", log2_P2_int_2);
    calculateExpRMSD<float>("log2_P1_16_neq", log2_P1_16_neq);
    calculateExpRMSD<float>("log2_P1_16_eq", log2_P1_16_eq);
    calculateExpRMSD<float>("log2_P2_16_neq", log2_P2_16_neq);
    calculateExpRMSD<float>("log2_P2_16_eq", log2_P2_16);
    calculateExpRMSD<float>("log2_P3_16_eq", log2_P3_16);
    calculateExpRMSD<float>("log2_musl", log2f_musl);
    calculateExpRMSD<float>("log2_fsf", log2f_fsf);
    calculateExpRMSD<float>("log2_fsf_1", log2f_fsf_1);
    calculateExpRMSD<float>("log2_Sun", log2f_Sun);
    calculateExpRMSD<float>("std_math::log2f", log2f);
    calculateExpRMSD<float, float(float)>("std_math::log2", std::log2);

    return 0;
}

