/* Literal feature-gradient expansion plus separate warm beta expansion.
 * Includes the byte-identical previously qualified arithmetic/PSF source.
 * No Python API, allocation, BLAS, mode writes, FMA or GIL release here.
 */
#include <limits.h>
#include "dense_score_core.c"

enum { FEATURE_RESTORATION = 7, FEATURE_EXPONENT = 8, FEATURE_GUARD = 9 };
#define FEATURE_MAX_ROWS 64
#define FEATURE_MAX_ENTRIES 16384
#define FEATURE_MAX_WIDTH 1024

typedef struct {
    const double *mantissa;
    const int32_t *exponent;
    const double *high;
    const double *low;
} feature_right_split;

static int feature_valid_exponent(int64_t exponent)
{
    /* frexp of any finite binary64, including zero and the least subnormal. */
    return exponent >= -1073 && exponent <= 1024;
}

static int feature_product(double a, const feature_right_split *right,
    size_t j, double *first_high, double *first_low)
{
    /* Deliberately normalize the already normalized column again, as _products.
     * Every parenthesis and the two first scaling operations remain literal. */
    int ae;
    double am = exact_frexp(a, &ae);
    double scaled = 134217729.0 * am;
    double ah = scaled - (scaled - am);
    double al = am - ah;
    double bm = right->mantissa[j];
    double bh = right->high[j];
    double bl = right->low[j];
    double high = am * bm;
    double low = ((ah * bh - high) + ah * bl + al * bh) + al * bl;
    int64_t exponent = (int64_t)ae + (int64_t)right->exponent[j];
    if (exponent < INT_MIN || exponent > INT_MAX) return FEATURE_EXPONENT;
    high = exact_ldexp(high, (int)exponent);
    low = exact_ldexp(low, (int)exponent);
    if (!isfinite(high) || !isfinite(low)) return NONFINITE;
    *first_high = high;
    *first_low = low;
    return OK;
}

static int feature_restore(double first, int64_t exponent, double *restored)
{
    if (exponent < INT_MIN || exponent > INT_MAX) return FEATURE_EXPONENT;
    double value = exact_ldexp(first, (int)exponent);
    if (!isfinite(value)) return FEATURE_RESTORATION;
    /* Compare the first-rounded intermediate, never an ideal/combined product.
     * This catches erased nonzero values too. A final fsum may be subnormal. */
    if (first != 0.0 && fabs(value) < DBL_MIN) return FEATURE_RESTORATION;
    *restored = value;
    return OK;
}

static int feature_gradient(const double *columns, size_t rows, size_t width,
    const feature_right_split right[3], const int32_t *z_exponents,
    const int32_t *alpha_exponents, int64_t penalty_exponent,
    double *terms, double staged[FEATURE_MAX_ROWS])
{
    if (!columns || !terms || rows < 2 || rows > FEATURE_MAX_ROWS || width < 1
        || width > FEATURE_MAX_WIDTH || width > FEATURE_MAX_ENTRIES / rows || !z_exponents || !alpha_exponents)
        return ARGUMENT;
    if (!feature_valid_exponent(penalty_exponent)) return FEATURE_EXPONENT;
    for (size_t j = 0; j < width; ++j) {
        if (!feature_valid_exponent(z_exponents[j])
            || !feature_valid_exponent(alpha_exponents[j])) return FEATURE_EXPONENT;
        for (size_t family = 0; family < 3; ++family) {
            if (!right[family].mantissa || !right[family].exponent
                || !right[family].high || !right[family].low) return ARGUMENT;
            if (!isfinite(right[family].mantissa[j])
                || !isfinite(right[family].high[j])
                || !isfinite(right[family].low[j])) return NONFINITE;
            if (!feature_valid_exponent(right[family].exponent[j])) return FEATURE_EXPONENT;
        }
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < width; ++j) {
            double column = columns[i * width + j];
            if (!isfinite(column)) return NONFINITE;
            int column_exponent;
            double mantissa = exact_frexp(column, &column_exponent);
            int64_t loss_exponent = (int64_t)column_exponent + z_exponents[j];
            int64_t regularizer_exponent = (int64_t)column_exponent
                + alpha_exponents[j] + penalty_exponent;
            for (size_t family = 0; family < 3; ++family) {
                double high, low;
                int status = feature_product(mantissa, &right[family], j, &high, &low);
                if (status != OK) return status;
                int64_t exponent = family == 0 ? loss_exponent : regularizer_exponent;
                status = feature_restore(high, exponent, &terms[(2 * family) * width + j]);
                if (status != OK) return status;
                status = feature_restore(low, exponent, &terms[(2 * family + 1) * width + j]);
                if (status != OK) return status;
            }
        }
        /* Exactly one sum: all zh, all zl, hh, hl, lh, ll, in original order. */
        int status = finite_fsum(terms, 6 * width, &staged[i]);
        if (status != OK) return status;
    }
    return OK;
}

/* Q is a distinct one-family expansion, not a packing-only reducer. Retain
 * both binary64 scaling operations and the same restoration refusal domain. */
static int feature_beta(const double *columns, size_t rows, size_t width,
    const feature_right_split *right, const int32_t *alpha_exponents,
    double *terms, double staged[FEATURE_MAX_ROWS])
{
    if (!columns || !terms || !right || !alpha_exponents
        || rows < 2 || rows > FEATURE_MAX_ROWS || width < 1
        || width > FEATURE_MAX_WIDTH || width > FEATURE_MAX_ENTRIES / rows || !right->mantissa
        || !right->exponent || !right->high || !right->low) return ARGUMENT;
    for (size_t j = 0; j < width; ++j) {
        if (!feature_valid_exponent(alpha_exponents[j])
            || !feature_valid_exponent(right->exponent[j])) return FEATURE_EXPONENT;
        if (!isfinite(right->mantissa[j]) || !isfinite(right->high[j])
            || !isfinite(right->low[j])) return NONFINITE;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < width; ++j) {
            double column = columns[i * width + j];
            if (!isfinite(column)) return NONFINITE;
            int column_exponent;
            double mantissa = exact_frexp(column, &column_exponent);
            double high, low;
            int status = feature_product(mantissa, right, j, &high, &low);
            if (status != OK) return status;
            int64_t exponent = (int64_t)column_exponent + alpha_exponents[j];
            status = feature_restore(high, exponent, &terms[j]);
            if (status != OK) return status;
            status = feature_restore(low, exponent, &terms[width+j]);
            if (status != OK) return status;
        }
        /* One original all-high then all-low sum, never partial sums. */
        int status = finite_fsum(terms, 2 * width, &staged[i]);
        if (status != OK) return status;
    }
    return OK;
}
