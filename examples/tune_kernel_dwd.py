"""Tune RBF kernel DWD on local training data with resumable fixed-fold CV."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dwd.gen_kern_dwd import KernGDWD
from resumable_kernel_cv import run_resumable_cv


def _positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def _seed(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError('must be between 0 and 2**32 - 1')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Local .npz with dense training arrays X and binary y; no pickled objects.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lambd', nargs='+', type=_positive_float, required=True)
    parser.add_argument('--gamma', nargs='+', type=_positive_float, required=True)
    parser.add_argument('--folds', type=_positive_int, default=5)
    parser.add_argument('--seed', type=_seed, default=42,
                        help='Estimator and shuffled stratified-fold seed (default: 42).')
    parser.add_argument('--jobs', type=_positive_int, default=1,
                        help='Maximum simultaneous folds; each uses one native thread.')
    parser.add_argument('--final-native-threads', type=_positive_int, default=1,
                        help='Native thread limit for fresh final scaling and fitting only (default: 1).')
    parser.add_argument('--max-iter', type=_positive_int, default=100)
    parser.add_argument('--scale', action='store_true', help='Fit StandardScaler within each training fold.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cv-only', action='store_true', help='Skip the fresh full-training refit.')
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error('--folds must be at least two')
    if args.data.suffix.lower() != '.npz':
        parser.error('--data must be a .npz file')
    try:
        with np.load(args.data, allow_pickle=False) as source:
            X, y = source['X'], source['y']
    except (KeyError, OSError, ValueError) as error:
        parser.error('Could not read training arrays X and y: ' + str(error))
    model = KernGDWD(kernel='rbf', random_state=args.seed, max_iter=args.max_iter)
    result = run_resumable_cv(
        model, X, y,
        {'lambd': args.lambd, 'kernel_kws': [{'gamma': gamma} for gamma in args.gamma]},
        args.run_dir, cv=StratifiedKFold(args.folds, shuffle=True, random_state=args.seed),
        jobs=args.jobs, resume=args.resume, scale=args.scale, refit_best=not args.cv_only,
        final_native_threads=args.final_native_threads)
    print('Best CV accuracy:', result['best_score'])
    print('Selected parameters:', result['best_params'])
    print('Work summary:', result['work_summary'])
    print('Results saved in:', args.run_dir.absolute())
    return result


if __name__ == '__main__':
    main()
