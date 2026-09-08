/*
 * Compiled arithmetic core for compensated kernel residuals.
 *
 * The DoubleLength/TripleLength helpers below are adapted from CPython 3.12.0,
 * Modules/mathmodule.c, Copyright Python Software Foundation and contributors.
 * Distributed under the Python Software Foundation License Version 2 and the
 * applicable accompanying licenses. See the complete CPYTHON-LICENSE.txt.
 * Original source: https://github.com/python/cpython/blob/v3.12.0/Modules/mathmodule.c
 *
 * Double and triple length extended precision algorithms from:
 * Accurate Sum and Dot Product, Takeshi Ogita, Siegfried M. Rump, Shin'Ichi Oishi
 * https://doi.org/10.1137/030601818
 *
 * Deliberately uses CPython's UNRELIABLE_FMA/Dekker fallback. Compile with
 * strict floating arithmetic, no reassociation, and implicit FMA disabled.
 */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#if defined(_WIN32)
#define DWD_EXPORT __declspec(dllexport)
#else
#define DWD_EXPORT __attribute__((visibility("default")))
#endif

typedef struct { double hi; double lo; } DoubleLength;
typedef struct { double hi; double lo; double tiny; } TripleLength;
static const TripleLength tl_zero = {0.0, 0.0, 0.0};

static DoubleLength dl_sum(double a, double b)
{
    /* CPython: Algorithm 3.1 Error-free transformation of the sum. */
    double x = a + b;
    double z = x - a;
    double y = (a - (x - z)) + (b - z);
    return (DoubleLength) {x, y};
}

static DoubleLength dl_split(double x)
{
    /* CPython: Dekker (5.5) and (5.6). */
    double t = x * 134217729.0;
    double hi = t - (t - x);
    double lo = x - hi;
    return (DoubleLength) {hi, lo};
}

static DoubleLength dl_mul(double x, double y)
{
    /* CPython: Dekker (5.12) and mul12(). */
    DoubleLength xx = dl_split(x);
    DoubleLength yy = dl_split(y);
    double p = xx.hi * yy.hi;
    double q = xx.hi * yy.lo + xx.lo * yy.hi;
    double z = p + q;
    double zz = p - z + q + xx.lo * yy.lo;
    return (DoubleLength) {z, zz};
}

static TripleLength tl_fma(double x, double y, TripleLength total)
{
    /* CPython: Algorithm 5.10 with SumKVert for K=3. */
    DoubleLength pr = dl_mul(x, y);
    DoubleLength sm = dl_sum(total.hi, pr.hi);
    DoubleLength r1 = dl_sum(total.lo, pr.lo);
    DoubleLength r2 = dl_sum(r1.hi, sm.lo);
    return (TripleLength) {sm.hi, r2.hi, total.tiny + r1.lo + r2.lo};
}

static double tl_to_d(TripleLength total)
{
    DoubleLength last = dl_sum(total.lo, total.hi);
    return total.tiny + last.lo + last.hi;
}

static double read_double(const void *base, ptrdiff_t offset)
{
    /* memcpy handles even an unaligned ndarray data pointer without C UB. */
    double value;
    memcpy(&value, (const unsigned char *)base + offset, sizeof(value));
    return value;
}

static int safe_operand(double value)
{
    double a = fabs(value);
    return a == 0.0 || (a >= 0x1p-200 && a <= 0x1p200);
}

static int supported_arithmetic(void)
{
    if (FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024)
        return 0;
    volatile double one = 1.0, half_ulp = 0x1p-53;
    volatile double tiny = DBL_MIN;
    double half = tiny * 0.5;
    return one + half_ulp == one && -one - half_ulp == -one
        && one + (half_ulp + half_ulp + half_ulp)
           == one + (half_ulp + half_ulp + half_ulp + half_ulp)
        && half != 0.0 && half * 2.0 == tiny;
}

/*
 * Byte strides permit C/F-order and sliced/negative-stride float64 views.
 * The caller owns shape, accessible storage, and non-overlapping output checks.
 * K logically has shape (n,n); x and rhs each have shape (n,).
 * Output buffers have length row_stop-row_start, indexed from zero.
 *
 * Return 0 on success; 1 invalid argument; 2 operand outside guarded domain;
 * 3 nonfinite arithmetic; 4 unsupported floating arithmetic. Outputs may be
 * partially filled on a failure and MUST be discarded by the caller.
 * No allocations, global mutable state, or changes to rounding/thread modes.
 */
DWD_EXPORT int dwd_sumprod_rows(
    const void *K, ptrdiff_t K_row_stride, ptrdiff_t K_col_stride,
    const void *x, ptrdiff_t x_stride,
    const void *rhs, ptrdiff_t rhs_stride,
    size_t n, size_t row_start, size_t row_stop,
    double shift, double intercept,
    double *scores, double *residuals, double *row_max_abs)
{
    if (!K || !x || !rhs || !scores || !residuals || !row_max_abs
        || n == 0 || n > ((size_t)1 << 20) - 3
        || row_start > row_stop || row_stop > n)
        return 1;
    if (!supported_arithmetic())
        return 4;
    if (!safe_operand(shift) || !safe_operand(intercept))
        return 2;
    for (size_t j = 0; j < n; ++j)
        if (!safe_operand(read_double(x, (ptrdiff_t)j * x_stride)))
            return 2;

    for (size_t i = row_start; i < row_stop; ++i) {
        double rhs_i = read_double(rhs, (ptrdiff_t)i * rhs_stride);
        double x_i = read_double(x, (ptrdiff_t)i * x_stride);
        if (!safe_operand(rhs_i))
            return 2;
        TripleLength score = tl_zero, residual = tl_zero;
        double maximum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            double a = read_double(K, (ptrdiff_t)i * K_row_stride
                                      + (ptrdiff_t)j * K_col_stride);
            double b = read_double(x, (ptrdiff_t)j * x_stride);
            if (!safe_operand(a))
                return 2;
            double absolute = fabs(a);
            if (absolute > maximum)
                maximum = absolute;
            score = tl_fma(a, b, score);
            residual = tl_fma(a, -b, residual);
        }
        /* Exactly the expanded product order of native_compensated_residual. */
        residual = tl_fma(rhs_i, 1.0, residual);
        residual = tl_fma(-intercept, 1.0, residual);
        residual = tl_fma(-shift, x_i, residual);
        double score_value = tl_to_d(score);
        double residual_value = tl_to_d(residual);
        if (!isfinite(score_value) || !isfinite(residual_value))
            return 3;
        size_t out = i - row_start;
        scores[out] = score_value;
        residuals[out] = residual_value;
        row_max_abs[out] = maximum;
    }
    return 0;
}
