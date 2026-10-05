"""Opt-in paired native microbenchmark, no model fits or saved-study access."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from test_shared_product_residual import BASELINE, compiled, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrent-work-note', default='Not recorded')
    args = parser.parse_args()
    if BASELINE is None or compiled._ACCEL is None:
        raise RuntimeError('Both frozen baseline and candidate native modules are required.')
    root = Path(args.output)
    root.parent.mkdir(parents=True, exist_ok=True)
    evidence = {'schema': 1, 'scope': 'single-thread native residual arithmetic only; no fits',
                'concurrent_work_note': args.concurrent_work_note, 'cases': [],
                'binaries': {label: {'path': str(Path(mod.__file__).resolve()),
                    'sha256': hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()}
                    for label, mod in [('baseline', BASELINE), ('candidate', compiled._ACCEL)]}}
    rng = np.random.default_rng(6100142)
    for n, kind in [(256, 'random'), (2048, 'random'), (4096, 'near_constant'),
                    (9600, 'near_constant'), (2048, 'exact_product_fallback')]:
        K = rng.random((n, n))
        x, rhs = rng.normal(size=n), rng.normal(size=n)
        shift = .125
        if kind == 'near_constant':
            K *= 1e-7
            K += 1.
            x *= 1e8
            x[-1] = -math.fsum(x[:-1].tolist())
            shift = 1e-9
        elif kind == 'exact_product_fallback':
            K.fill(1.)
        old_status, expected = evaluate(BASELINE, K, x, rhs, shift)
        status, actual = evaluate(compiled._ACCEL, K, x, rhs, shift)
        if old_status != 0 or status != 0 or any(a.tobytes() != b.tobytes() for a, b in zip(actual, expected)):
            raise AssertionError('Native microbenchmark outputs differ.')
        timings = {'baseline': [], 'candidate': []}
        # Reuse inputs, warm both modules, and alternate order in each pair.
        for repeat in range(8):
            order = [('baseline', BASELINE), ('candidate', compiled._ACCEL)]
            if repeat % 2:
                order.reverse()
            for label, module in order:
                started = time.perf_counter()
                status, values = evaluate(module, K, x, rhs, shift)
                elapsed = time.perf_counter() - started
                if status != 0:
                    raise AssertionError('Native arithmetic declined during measurement.')
                timings[label].append(elapsed)
        medians = {label: statistics.median(values) for label, values in timings.items()}
        evidence['cases'].append({'n': n, 'case': kind, 'kernel_bytes': K.nbytes,
                                  'bitwise_equal': True, 'timings_seconds': timings,
                                  'median_seconds': medians,
                                  'baseline_over_candidate': medians['baseline'] / medians['candidate']})
        print(json.dumps(evidence['cases'][-1]), flush=True)
        del K, x, rhs, expected, actual, values
        gc.collect()
    root.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
