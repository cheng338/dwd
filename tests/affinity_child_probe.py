"""Change affinity only for this owned child and compare native row arithmetic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mask', choices=('1', '2', 'available'), required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args()
    package_root = Path(args.package_root).resolve()
    assert 1 <= args.budget <= (16 if os.name == 'nt' else 20)
    import psutil

    owner = psutil.Process()
    result = dict(pid=owner.pid, creation_time=owner.create_time(), budget=args.budget,
                  requested_mask=args.mask, dynamic=args.dynamic, records=[], status='pending')
    original = None
    changed_affinity = False
    try:
        if not hasattr(owner, 'cpu_affinity'):
            result.update(status='unsupported', reason='psutil self-affinity API unavailable')
            return
        try:
            original = owner.cpu_affinity()
            result['original_affinity'] = original
            assert original, 'Self-affinity returned an empty mask'
            count = len(original) if args.mask == 'available' else int(args.mask)
            if len(original) < count:
                result.update(status='unsupported', reason='Requested mask exceeds the child inherited mask')
                return
            selected = original[:count]
            owner.cpu_affinity(selected)
            changed_affinity = True
            assert set(owner.cpu_affinity()) == set(selected)
        except (psutil.AccessDenied, NotImplementedError) as error:
            result.update(status='unsupported', reason=type(error).__name__ + ': ' + str(error))
            return

        # Restrict the calling thread before importing numerical libraries or
        # creating residual workers. No other process receives an affinity call.
        sys.path.insert(0, str(package_root))
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        from dwd import _compiled_residual as compiled
        from dwd import _residual_worker_scope as worker_scope
        assert Path(compiled.__file__).resolve() == package_root / 'dwd/_compiled_residual.py'
        if compiled._ACCEL is None:
            result.update(status='unsupported', reason='Optional native residual extension unavailable')
            return
        result['native_path'] = compiled._ACCEL.__file__
        assert Path(compiled._ACCEL.__file__).resolve().parent == package_root / 'dwd'
        result['package_root'] = str(package_root)
        result['compiled_path'] = compiled.__file__
        result['compiled_sha256'] = hashlib.sha256(Path(compiled.__file__).read_bytes()).hexdigest()
        result['native_sha256'] = hashlib.sha256(Path(compiled._ACCEL.__file__).read_bytes()).hexdigest()
        result['system_cpu_count'] = os.cpu_count()

        inputs = {}
        expected = {}
        for n in (2048, 4096):
            index = np.arange(n, dtype=np.float64)
            row = ((index % 17) - 8) / 32
            x = ((index % 13) - 6) / 64
            rhs = (index % 7) / 8
            for array in (row, x, rhs):
                array.flags.writeable = False
            kernel = np.broadcast_to(row, (n, n))
            inputs[n] = (kernel, x, rhs, row)
            output = [np.empty(n) for _ in range(3)]
            assert compiled._ACCEL.evaluate(kernel, x, rhs, .125, .25, 0, n, *output) == 0
            expected[n] = output

        pools_seen = []

        def run(scope, n, phase):
            kernel, x, rhs, row = inputs[n]
            before = (digest(row), digest(x), digest(rhs), kernel.shape, kernel.strides)
            count = compiled._worker_count(n)
            previous = scope.executor
            previous_workers = scope.workers
            output = compiled.compiled_values(kernel, x, rhs, .125, .25)
            assert output is not None
            for actual, reference in zip(output, expected[n]):
                np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
            assert before == (digest(row), digest(x), digest(rhs), kernel.shape, kernel.strides)
            assert not kernel.flags.writeable and not x.flags.writeable and not rhs.flags.writeable
            pool = scope.executor
            if pool is not None and all(pool is not old for old in pools_seen):
                pools_seen.append(pool)
            if count == 1:
                assert pool is None
            else:
                assert pool is not None and pool._max_workers == count and scope.workers == count
            if previous is not None and previous_workers != count:
                assert previous._shutdown and all(not thread.is_alive() for thread in previous._threads)
                assert pool is not previous
            threads = [] if pool is None else sorted(
                [dict(name=thread.name, native_id=thread.native_id, alive=thread.is_alive())
                 for thread in pool._threads], key=lambda item: item['native_id'])
            result['records'].append(dict(
                phase=phase, n=n, affinity=owner.cpu_affinity(), workers=count,
                configured_workers=None if pool is None else pool._max_workers,
                threads=threads, output_sha256=[digest(value) for value in output],
                direct_sha256=[digest(value) for value in expected[n]],
                input_sha256=list(before[:3]), kernel_shape=list(kernel.shape),
                kernel_strides=list(kernel.strides),
                previous_pool_drained=(previous is None or previous_workers == count or
                                       (previous._shutdown and all(not thread.is_alive() for thread in previous._threads))),
            ))

        with threadpool_limits(limits=args.budget, user_api='blas'):
            pools = threadpool_info()
            budgets = [int(pool['num_threads']) for pool in pools
                       if pool.get('user_api') == 'blas' and pool.get('num_threads', 0) > 0]
            assert budgets and min(budgets) <= args.budget
            result['pools'] = pools
            result['observed_blas_budget'] = min(budgets)
            result['count_only'] = {str(n): compiled._worker_count(n) for n in (16384, 20480)}
            with worker_scope.ResidualWorkerScope() as scope:
                run(scope, 2048, 'initial')
                run(scope, 4096, 'initial')
                if args.dynamic:
                    for label, mask in [('one', original[:1]), ('two', original[:2]), ('reexpanded', original)]:
                        if label == 'two' and len(original) < 2:
                            continue
                        owner.cpu_affinity(mask)
                        assert set(owner.cpu_affinity()) == set(mask)
                        run(scope, 4096, label)
            result['all_residual_pools_drained'] = all(
                pool._shutdown and all(not thread.is_alive() for thread in pool._threads)
                for pool in pools_seen)
            assert result['all_residual_pools_drained']
        result['status'] = 'complete'
    finally:
        if changed_affinity:
            owner.cpu_affinity(original)
            result['restored_affinity'] = owner.cpu_affinity()
            assert set(result['restored_affinity']) == set(original)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
