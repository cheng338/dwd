"""High-precision oracle for finite q=1 kernel DWD MM updates in small tests."""
from decimal import Decimal, localcontext

import numpy as np


def reference_mm_oracle(K, y, penalty, alpha, iterations=3):
    """Use 80-digit bordered solves, independently of BLAS and LAPACK.

    The represented kernel and initial coefficients define the input. Each
    update solves (K + 2*penalty*step*I) @ alpha + db = K @ old_alpha - step*z,
    with sum(alpha)=0 and an unregularized intercept increment db. This checks
    the specified finite MM steps, not convergence to an optimum. Only q=1 and
    a zero initial intercept are supported by this small test oracle.
    """
    D = lambda value: Decimal.from_float(float(value))
    with localcontext() as context:
        context.prec = 80
        n = len(y)
        k = [[D(value) for value in row] for row in K]
        labels = [D(value) for value in y]
        a, b = [D(value) for value in alpha], Decimal(0)
        step = Decimal(n) / 4
        shift = 2 * D(penalty) * step
        zero, one = Decimal(0), Decimal(1)
        matrix = [[k[i][j] + (shift if i == j else zero) for j in range(n)]
                  + [one] for i in range(n)] + [[one] * n + [zero]]
        for _ in range(iterations):
            scores = [sum((v * c for v, c in zip(row, a)), zero) for row in k]
            margins = [label * (score + b) for label, score in zip(labels, scores)]
            gradients = [label * (-one if margin <= Decimal('.5')
                         else -one / (4 * margin * margin)) / n
                         for label, margin in zip(labels, margins)]
            rhs = [score - step * grad for score, grad in zip(scores, gradients)] + [zero]
            augmented = [row[:] + [r] for row, r in zip(matrix, rhs)]
            for column in range(n + 1):
                pivot = max(range(column, n + 1), key=lambda row: abs(augmented[row][column]))
                if augmented[pivot][column] == 0:
                    raise AssertionError('Singular Decimal oracle system')
                augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
                divisor = augmented[column][column]
                augmented[column] = [v / divisor for v in augmented[column]]
                for row in range(column + 1, n + 1):
                    factor = augmented[row][column]
                    for j in range(column, n + 2):
                        augmented[row][j] -= factor * augmented[column][j]
            solution = [zero] * (n + 1)
            for row in reversed(range(n + 1)):
                solution[row] = augmented[row][-1] - sum(
                    (augmented[row][j] * solution[j] for j in range(row + 1, n + 1)), zero)
            a, b = solution[:n], b + solution[n]
        return np.asarray([float(value) for value in a]), float(b)
