/* rnnoise_sp_libm.h -- the "+F" RNNoise variant on the S3: single-precision libm.
 *
 * Force-included (-include) ahead of each vendored RNNoise source, with the
 * vendored text UNCHANGED. Xiph's C calls log10(), sqrt() and floor() on float
 * operands; the Xtensa LX7 has a single-precision FPU and no double, so each
 * call promotes to a software double (ROM __extendsfdf2 / __adddf3 / ... and
 * newlib's double log10/sqrt/floor), exactly the F411's situation. The
 * function-like macros below take effect after <math.h> has declared the
 * double functions, so declarations are untouched and only the call sites
 * move to the float routines. The output is NOT the as-shipped output (a
 * task-15 question, measured against the oracle like any port); the purpose
 * is to measure what the software double costs per hop, which neither
 * Cortex-M image could (no stage stamps). Never the measurement image. */
#ifndef RNNOISE_SP_LIBM_H
#define RNNOISE_SP_LIBM_H
#include <math.h>
#define log10(x) log10f(x)
#define sqrt(x)  sqrtf(x)
#define floor(x) floorf(x)
#endif
