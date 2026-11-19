"""Deterministic recovery-path control: real fit and arithmetic, unavailable cheap bound."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--budget', required=True, type=int)
    args = parser.parse_args()
    assert 1 <= args.budget <= (16 if os.name == 'nt' else 20)
    package_root = Path(args.package_root).resolve()
    output = Path(args.output)
    phases = output.with_suffix('.phases.jsonl')
    with phases.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'phase': 'started', 'pid': os.getpid(),
                                 'monotonic_ns': time.monotonic_ns()}) + '\n')

    def phase(name, **details):
        with phases.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'monotonic_ns': time.monotonic_ns(), **details}) + '\n')

    import psutil
    owner = psutil.Process()
    result = dict(status='pending', pid=owner.pid, creation_time=owner.create_time(),
                  package_root=str(package_root), worker_counts=[], dispatch_records=[],
                  setup_blas_budget=1, residual_blas_budget=args.budget,
                  control='injected ordinary-bound unavailability; real compensated recovery',
                  ordinary_bound_constructor_calls=0, portable_residual_calls=0)
    original, changed = None, False
    try:
        if not hasattr(owner, 'cpu_affinity'):
            result.update(status='unsupported', reason='psutil self-affinity API unavailable')
            return
        try:
            original = owner.cpu_affinity()
            assert original
            result['original_affinity'] = original
            owner.cpu_affinity(original[:1])
            changed = True
            result['restricted_affinity'] = owner.cpu_affinity()
            assert result['restricted_affinity'] == original[:1]
        except (psutil.AccessDenied, NotImplementedError) as error:
            result.update(status='unsupported', reason=type(error).__name__ + ': ' + str(error))
            return
        phase('self_affinity_restricted', affinity=owner.cpu_affinity())
        sys.path.insert(0, str(package_root))
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        from dwd.gen_kern_dwd import KernGDWD
        from dwd import _compiled_residual as compiled
        from dwd import _native_residual as native
        from dwd import _kernel_linear_system as linear
        assert Path(compiled.__file__).resolve() == package_root / 'dwd/_compiled_residual.py'
        if compiled._ACCEL is None:
            result.update(status='unsupported', reason='Optional native residual extension unavailable')
            return
        assert Path(compiled._ACCEL.__file__).resolve().parent == package_root / 'dwd'
        result['compiled_sha256'] = hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest()
        result['native_sha256'] = hashlib.sha256(Path(compiled._ACCEL.__file__).read_bytes()).hexdigest()
        features = ((np.arange(2048, dtype=np.float64) % 17) - 8).reshape(-1, 1) / 32
        labels = np.tile(np.array([-1, 1]), 1024)
        features.flags.writeable = labels.flags.writeable = False
        input_digest = hashlib.sha256(features.tobytes() + labels.tobytes()).hexdigest()
        result['input_sha256'] = input_digest
        result['feature_shape'] = list(features.shape)
        original_count = compiled._worker_count
        original_dispatch = native._dispatch_compiled_values
        original_bounds = linear.OrdinaryResidualBounds
        original_portable = linear.compensated_residual

        def observed_portable(*args, **kwargs):
            actual = original_portable(*args, **kwargs)
            result['portable_residual_calls'] += 1
            return actual

        def unavailable_ordinary_bounds(*positional, **keywords):
            result['ordinary_bound_constructor_calls'] += 1
            phase('ordinary_bound_unavailable_injected')
            # This is an existing supported availability-failure branch.
            # It does not manufacture residuals, factors, solutions or an
            # acceptance decision. The real compensated checks must pass.
            raise FloatingPointError('Test fixture: ordinary bound unavailable')

        def observed_count(n):
            count = original_count(n)
            result['worker_counts'].append([n, count])
            return count

        def observed_dispatch(*positional):
            # This native residual arithmetic does not invoke BLAS. Apply an
            # actual process-local library budget solely at its real sizing
            # boundary, then restore the serial dense-fit setup budget.
            with threadpool_limits(limits=args.budget, user_api='blas'):
                pools = threadpool_info()
                budgets = [int(pool['num_threads']) for pool in pools
                           if pool.get('user_api') == 'blas' and pool.get('num_threads', 0) > 0]
                assert budgets and min(budgets) == args.budget
                before = len(result['worker_counts'])
                phase('native_dispatch_enter', n=len(positional[1]), affinity=owner.cpu_affinity(),
                      blas_budgets=budgets)
                actual = original_dispatch(*positional)
                counts = result['worker_counts'][before:]
                assert counts == [[2048, 1]], 'Real native dispatch must use the affinity-aware helper'
                assert actual is not None and actual is not compiled._COMPILED_DOMAIN_REFUSAL
                assert len(actual) == 3 and all(value.shape == (2048,) for value in actual)
                result['dispatch_records'].append(dict(
                    n=len(positional[1]), affinity=owner.cpu_affinity(), pools=pools,
                    worker_counts=counts,
                    output_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in actual]))
                phase('native_dispatch_returned', worker_counts=counts)
                return actual

        compiled._worker_count = observed_count
        native._dispatch_compiled_values = observed_dispatch
        linear.OrdinaryResidualBounds = unavailable_ordinary_bounds
        linear.compensated_residual = observed_portable
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                result['setup_pools'] = threadpool_info()
                assert all(pool['num_threads'] == 1 for pool in result['setup_pools']
                           if pool.get('user_api') == 'blas')
                phase('before_public_fit', feature_shape=list(features.shape))
                model = KernGDWD(lambd=1.0, kernel='linear', backend='cholesky',
                                 max_iter=1, stopping='fixed', acceleration=None,
                                 random_state=17).fit(features, labels)
                phase('public_fit_returned', iterations=int(model.n_iter_), backend=model.backend_)
                decisions = model.decision_function(features)
                result['after_fit_pools'] = threadpool_info()
                assert all(pool['num_threads'] == 1 for pool in result['after_fit_pools']
                           if pool.get('user_api') == 'blas')
                result.update(n_iter=int(model.n_iter_), backend=model.backend_,
                              linear_system_diagnostics=model.diagnostics_['linear_system_diagnostics'],
                              psd_validation=model.diagnostics_['psd_validation'],
                              eigenvalue_check_seconds=float(model.diagnostics_['eigenvalue_check_seconds']),
                              final_objective=float(model.final_objective_),
                              termination_reason=model.termination_reason_,
                              finite_predictions=bool(np.isfinite(decisions).all()),
                              finite_parameters=bool(np.isfinite(model.dual_coef_).all()
                                                     and np.isfinite(model.intercept_).all()
                                                     and np.isfinite(model.final_objective_)),
                              decisions_sha256=hashlib.sha256(decisions.tobytes()).hexdigest(),
                              input_unchanged=input_digest == hashlib.sha256(
                                  features.tobytes() + labels.tobytes()).hexdigest())
        finally:
            compiled._worker_count = original_count
            native._dispatch_compiled_values = original_dispatch
            linear.OrdinaryResidualBounds = original_bounds
            linear.compensated_residual = original_portable
            result['test_hooks_restored'] = (
                compiled._worker_count is original_count
                and native._dispatch_compiled_values is original_dispatch
                and linear.OrdinaryResidualBounds is original_bounds
                and linear.compensated_residual is original_portable)
        assert result['n_iter'] == 1 and result['finite_predictions'] and result['finite_parameters']
        assert result['input_unchanged'] and result['eigenvalue_check_seconds'] == 0.
        assert result['psd_validation'] == 'internally constructed PSD kernel (known family)'
        assert result['ordinary_bound_constructor_calls'] > 0 and result['test_hooks_restored']
        assert result['linear_system_diagnostics']['ordinary_residual_declines'] > 0
        assert result['linear_system_diagnostics']['native_residual_attempts'] > 0
        assert result['linear_system_diagnostics']['native_residual_attempts'] == (
            result['linear_system_diagnostics'].get('native_residual_checks', 0)
            + result['linear_system_diagnostics'].get('native_residual_declines', 0))
        assert result['linear_system_diagnostics']['compensated_residual_checks'] > 0
        result['status'] = 'complete'
    finally:
        if changed:
            owner.cpu_affinity(original)
            result['restored_affinity'] = owner.cpu_affinity()
            assert set(original) == set(result['restored_affinity'])
        phase('finally', status=result['status'])
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
