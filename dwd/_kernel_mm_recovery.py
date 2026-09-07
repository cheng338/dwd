"""Lazy, bounded recovery of an MM function in certified kernel coordinates.

This is a function-space MM contract, not an alternative solver for the
stronger coefficient-sum gauge. It is instantiated only after an ordinary
numerical update fails. The original matrix and all model parameters stay
unchanged. Only small diagnostics survive in the returned fitted estimator.
"""
from time import perf_counter


class KernelMMRecovery:
    def __init__(self, K, delta):
        self.K, self.delta = K, delta
        self.action = None
        self.info = {
            'attempted': False, 'accepted_actions': 0,
            'native_small_spectral_actions': 0, 'rounded_exact_actions': 0,
            'reduced_coefficient_gauge_checked': False,
            'true_mm_function_checked': False,
            'kernel_approximation_used': False,
            'positive_directions_discarded': 0,
        }

    @property
    def active(self):
        return self.action is not None

    def step(self, a, previous_offset, trigger=None):
        """Return alpha, absolute checked offset, and original-kernel scores."""
        # No exact-arithmetic module, matrix scan or rational allocation on a
        # healthy ordinary fit. Each helper has its own fixed resource caps.
        from ._exact_kernel_factor import exact_kernel_factor, BudgetExhausted, CertificateFailure
        from ._certified_kernel_mm import CertifiedKernelMM

        started = perf_counter()
        if not self.info['attempted']:
            self.info.update(attempted=True, initial_numerical_failure=str(trigger))
            certified = exact_kernel_factor(self.K)
            self.info['certificate_status'] = certified.status
            self.info['certificate'] = dict(certified.metadata)
            if not certified.certified:
                raise FloatingPointError(
                    f'{trigger} Certified MM function recovery is inapplicable: '
                    f'{certified.status}.')
            try:
                self.action = CertifiedKernelMM(self.K, certified.factor, self.delta)
            except (BudgetExhausted, CertificateFailure, FloatingPointError,
                    OverflowError) as exc:
                self.info['preparation_failure'] = f'{type(exc).__name__}: {exc}'
                raise FloatingPointError(
                    f'{trigger} Certified MM function preparation failed: '
                    f'{self.info["preparation_failure"]}.') from None
            self.info['preparation'] = dict(self.action.info)
        if self.action is None:
            raise FloatingPointError('Certified MM function recovery was already exhausted.')
        try:
            alpha, offset, scores, detail = self.action.step(a, previous_offset)
        except (BudgetExhausted, CertificateFailure, FloatingPointError,
                OverflowError) as exc:
            self.info['last_failure'] = f'{type(exc).__name__}: {exc}'
            raise FloatingPointError(
                'Certified MM function recovery failed after the original numerical '
                f'failure ({self.info["initial_numerical_failure"]}): '
                f'{self.info["last_failure"]}.') from None
        self.info['accepted_actions'] += 1
        self.info['true_mm_function_checked'] = True
        self.info[detail['accepted_candidate'] + '_actions'] += 1
        self.info['last_action'] = detail
        self.info['total_seconds'] = self.info.get('total_seconds', 0.) + perf_counter() - started
        return alpha, offset, scores
