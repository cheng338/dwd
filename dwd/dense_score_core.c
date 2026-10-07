/* Ordered dense expanded-score arithmetic.
 *
 * Finite fsum reduction adapted from CPython 3.12.12 Modules/mathmodule.c,
 * Copyright Python Software Foundation and contributors. Distributed under
 * the PSF License Version 2 and accompanying licenses: CPYTHON-LICENSE.txt.
 * https://github.com/python/cpython/blob/v3.12.12/Modules/mathmodule.c
 *
 * Products reproduce dwd._compensated_residual._products expression order:
 * frexp mantissas, Dekker high/low terms, ldexp high and low separately.
 * Every row then feeds all high terms, followed by all low terms, to fsum.
 * No reassociation, FMA, widened accumulator, altered bound or coefficient.
 */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#define DWD_SCORE_X86_64 1
#else
#define DWD_SCORE_X86_64 0
#endif

#pragma STDC FP_CONTRACT OFF

enum { OK = 0, ARGUMENT = 1, ARITHMETIC_MODE = 2, NONFINITE = 3,
       MEMORY = 4, PARTIAL_CAPACITY = 5, SUM_OVERFLOW = 6 };

/* Binary64 normal operands can be normalized/scaled by changing only their
 * exponent field. These operations are exact and retain the same rounding
 * points as frexp/ldexp. Zero, subnormal and exceptional results use libc. */
static double exact_frexp(double x, int *exponent)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    int e = (int)((bits >> 52) & 0x7ffu);
    if (e > 0 && e < 2047) {
        *exponent = e - 1022;
        bits = (bits & UINT64_C(0x800fffffffffffff)) | UINT64_C(0x3fe0000000000000);
        memcpy(&x, &bits, sizeof x);
        return x;
    }
    return frexp(x, exponent);
}

static double exact_ldexp(double x, int exponent)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    int e = (int)((bits >> 52) & 0x7ffu);
    int adjusted = e + exponent;
    if (e > 0 && e < 2047 && adjusted > 0 && adjusted < 2047) {
        bits = (bits & UINT64_C(0x800fffffffffffff)) | ((uint64_t)adjusted << 52);
        memcpy(&x, &bits, sizeof x);
        return x;
    }
    return ldexp(x, exponent);
}

static int supported_arithmetic(void)
{
#if !DWD_SCORE_X86_64
    return 0;
#else
    if (sizeof(double) != sizeof(uint64_t) || FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024)
        return 0;
    /* SSE round-to-nearest, gradual underflow, denormals accepted. */
    if ((_mm_getcsr() & (0x6000u | 0x8000u | 0x0040u)) != 0)
        return 0;
    volatile double one = 1.0, half_ulp = 0x1p-53, tiny = DBL_MIN;
    double half = tiny * 0.5;
    return one + half_ulp == one && -one - half_ulp == -one
        && one + 3.0 * half_ulp == one + 4.0 * half_ulp
        && half != 0.0 && half * 2.0 == tiny;
#endif
}

/* Same finite term ingestion and final half-even correction as math.fsum.
 * A deliberately bounded partial list fails back to Python if exhausted.
 * It never truncates a partial or changes the answer to satisfy a budget.
 */
static int finite_fsum(const double *terms, size_t count, double *out)
{
    double p[64], hi = 0.0, lo = 0.0, x, y, t, yr;
    size_t n = 0;
    for (size_t k = 0; k < count; ++k) {
        x = terms[k];
        if (!isfinite(x)) return NONFINITE;
        size_t i = 0;
        for (size_t j = 0; j < n; ++j) {
            y = p[j];
            if (fabs(x) < fabs(y)) { t = x; x = y; y = t; }
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        n = i;
        if (x != 0.0) {
            if (!isfinite(x)) return SUM_OVERFLOW;
            if (n == 64) return PARTIAL_CAPACITY;
            p[n++] = x;
        }
    }
    hi = 0.0;
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            x = hi;
            y = p[--n];
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n-1] < 0.0) ||
                      (lo > 0.0 && p[n-1] > 0.0))) {
            y = lo * 2.0;
            x = hi + y;
            yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    if (!isfinite(hi)) return SUM_OVERFLOW;
    *out = hi;
    return OK;
}

/* Caller supplies validated C-contiguous, aligned arrays with private output.
 * On any nonzero status the output is discarded and Python handles the call.
 * No Python API, global state, thread creation, BLAS, or rounding-mode writes.
 */
int dwd_expanded_score_rows(const double *rows, size_t count, size_t width,
    const double *alpha, const double *bm, const int32_t *be,
    const double *bh, const double *bl, double *output)
{
    if (!rows || !alpha || !bm || !be || !bh || !bl || !output ||
        count < 1 || count > 8 || width < 1 || width > ((size_t)1 << 20))
        return ARGUMENT;
    if (!supported_arithmetic()) return ARITHMETIC_MODE;
    for (size_t j = 0; j < width; ++j) {
        if (!isfinite(alpha[j]) || !isfinite(bm[j]) ||
            !isfinite(bh[j]) || !isfinite(bl[j]) || be[j] < -1073 || be[j] > 1024)
            return NONFINITE;
    }
    double *terms = (double *)malloc(2 * width * sizeof(double));
    if (!terms) return MEMORY;
    int status = OK;
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < width; ++j) {
            double a = rows[i * width + j];
            if (!isfinite(a)) { status = NONFINITE; goto done; }
            int ae;
            double am = exact_frexp(a, &ae);
            double ca = 134217729.0 * am;
            double ah = ca - (ca - am);
            double al = am - ah;
            double high = am * bm[j];
            double low = ((ah * bh[j] - high) + ah * bl[j] + al * bh[j]) + al * bl[j];
            int exponent = ae + be[j];
            high = exact_ldexp(high, exponent);
            low = exact_ldexp(low, exponent);
            if (!isfinite(high) || !isfinite(low)) { status = NONFINITE; goto done; }
            terms[j] = high;
            terms[width + j] = low;
        }
        status = finite_fsum(terms, 2 * width, &output[i]);
        if (status != OK) goto done;
    }
done:
    free(terms);
    return status;
}
