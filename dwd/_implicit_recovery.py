"""Automatic, bounded exact recovery at two numerical preparation failures.

Keep ordinary successful fits on their existing path and preserve numerical
hooks, target identity and verified output checks.
"""
import hashlib
import math
import struct

import numpy as np


def eligible(module, X, shift):
    return (type(X) is np.ndarray
            and X.dtype == np.dtype(np.float64)
            and math.isfinite(float(shift)) and shift > 0
            and module['get_step_implicit'] is module['_BUILTIN_IMPLICIT_STEP']
            and module['V_grad'] is module['_BUILTIN_LOSS_GRADIENT']
            and module['_linear_gradient'] is module['_BUILTIN_LINEAR_GRADIENT']
            and not module['_has_numpy_error_handler']())


def require_eligible(module, X, shift):
    if not eligible(module, X, shift):
        raise ValueError('Implicit GenDWD exact recovery cannot bypass changed numerical hooks.')


def prepare(module, X, shift):
    if not eligible(module, X, shift):
        return None
    from dwd._exact_linear_mm import ExactCoordinateMM, ExactCoordinateLimitError
    try:
        return ExactCoordinateMM(X, shift)
    except ExactCoordinateLimitError as exc:
        raise ValueError('Implicit GenDWD exact recovery exceeded its bounded resource policy.') from exc


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def _bits(value):
    return struct.pack('>d', float(value)).hex()


def update(module, recovery, X, shift, h, beta, offset):
    require_eligible(module, X, shift)
    from dwd._exact_linear_mm import ExactCoordinateLimitError
    system = recovery
    if _digest(X) != system.input_sha256:
        raise ValueError('Implicit GenDWD exact recovery feature target changed after preparation.')
    try:
        proposed, receipt = system.next_iterate(h, beta, offset)
    except ExactCoordinateLimitError as exc:
        raise ValueError('Implicit GenDWD exact recovery exceeded its bounded resource policy.') from exc
    if (type(proposed) is not np.ndarray or proposed.dtype != np.dtype(np.float64)
            or proposed.shape != (X.shape[1]+1,) or proposed.flags.writeable
            or not np.isfinite(proposed).all() or type(receipt) is not dict):
        raise ValueError('Implicit GenDWD exact recovery returned an invalid result.')
    if (receipt.get('schema') != 'f11-exact-coordinate-next-v1'
            or receipt.get('status') != 'exact_coordinates_rounded'
            or receipt.get('target') != 'original_feature_fixed_h'
            or receipt.get('rounding') != 'binary64_round_to_nearest_ties_to_even'
            or receipt.get('coordinate_rounding_verified') is not True
            or receipt.get('input_sha256') != system.input_sha256
            or receipt.get('shift_bits') != _bits(shift)
            or receipt.get('h_sha256') != _digest(h)
            or receipt.get('beta_sha256') != _digest(beta)
            or receipt.get('offset_bits') != _bits(offset)
            or receipt.get('theta_sha256') != _digest(proposed)
            or receipt.get('theta_bits') != [_bits(v) for v in proposed]):
        raise ValueError('Implicit GenDWD exact recovery returned an unbound authority record.')
    return proposed
