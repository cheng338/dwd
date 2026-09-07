"""Optional score-space proximal MM with objective-based momentum restart.

The original kernel, penalty and free intercept are unchanged. Linear solves
remain the responsibility of the selected, independently verified system.
"""
import math

import numpy as np

from .gen_dwd import V, V_grad


class RestartedMM:
    """Extrapolate total decisions, then solve the original MM prox equations.

    The FISTA schedule starts with two ordinary MM updates. A proposal whose
    objective increases is discarded, momentum is reset and one ordinary step
    is recomputed. Only the returned update may be observed or counted by the
    outer solver. A failed numerical trial at nonzero momentum can also reset
    once and retry an ordinary step. Ordinary-step failures propagate; no
    unchecked point is accepted and no callback or validation function is called
    here.
    """
    def __init__(self, system, y, lambd, q, step_scale, *, proximal_step=None):
        self.system = system
        # A private function-MM action can use an equivalent coefficient
        # representative. Its offset is absolute because rhs below contains
        # the extrapolated total decisions, including the intercept.
        self.proximal_step = proximal_step
        self.y, self.lambd, self.q = y, lambd, q
        self.step_scale = step_scale
        self.previous = None
        self.momentum = 1.
        self.weight = 0.
        self.info = {'acceleration_restarts': 0, 'acceleration_numerical_restarts': 0,
                     'acceleration_proposals': 0,
                     'acceleration_accepted_updates': 0}

    def _proposal(self, total_scores):
        self.info['acceleration_proposals'] += 1
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            z = self.y * V_grad(self.y * total_scores, q=self.q) / len(self.y)
            rhs = total_scores - self.step_scale * z
            if self.proximal_step is None:
                alpha, offset = self.system.solve_constrained(rhs)
                scores = self.system.last_product.copy()
            else:
                alpha, offset, scores = self.proximal_step(rhs)
            value = float(np.mean(V(self.y * (scores + offset), q=self.q))
                          + self.lambd * (alpha @ scores))
        if not np.isfinite(value) or not np.isfinite(scores).all() or not np.isfinite(offset):
            raise FloatingPointError('Nonfinite accelerated kernel MM proposal.')
        return alpha, float(offset), scores, value

    def step(self, scores, offset, objective):
        with np.errstate(over='raise', invalid='raise'):
            total = scores + offset
        retry = False
        try:
            with np.errstate(over='raise', invalid='raise'):
                extrapolated = (total if self.previous is None else
                                total + self.weight * (total - self.previous))
            alpha, new_offset, new_scores, value = self._proposal(extrapolated)
            retry = self.weight != 0. and value > objective
        except FloatingPointError as exc:
            if self.weight == 0.:
                raise
            retry = True
            self.info['acceleration_numerical_restarts'] += 1
            self.info['acceleration_last_numerical_restart'] = str(exc)
        momentum = self.momentum
        if retry:
            self.info['acceleration_restarts'] += 1
            momentum = 1.
            alpha, new_offset, new_scores, value = self._proposal(total)
        if value > objective + 1e-9 * max(1., abs(objective)):
            raise FloatingPointError('Accelerated kernel MM could not establish objective descent '
                                     'after an ordinary MM retry.')
        next_momentum = (1. + math.sqrt(1. + 4. * momentum * momentum)) / 2.
        self.weight = (momentum - 1.) / next_momentum
        self.momentum = next_momentum
        self.previous = total
        self.info['acceleration_accepted_updates'] += 1
        return alpha, new_offset, new_scores
