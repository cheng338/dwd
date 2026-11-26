# Testing an installed DWD wheel

Use checkout tests while developing source code. Use this procedure when the
question is whether a specific built wheel contains and runs the intended code.
Running `python -m pytest` from DWD's flat-layout checkout can import local
`dwd/` instead of the installation. A matching version string does not establish
artifact identity.

The existing CI helpers verify imported origins, distribution versions and
installed payload bytes against the exact wheel. They also check the expected
native or portable routes. The test bundle contains tests, fixtures, examples and
metadata, but no importable package source root.

## Prepare the inputs and environment

- Use the source, tests and checker that produced the wheel. For a published
  wheel, use its release tag or source distribution. For a development wheel,
  use its exact source archive or checkout, since several revisions can share
  the same development version.
- Put exactly one compatible DWD wheel in a dedicated wheel directory. The
  default example expects a native wheel; see the portable variant below.
- Replace every quoted absolute-path placeholder before running a command.
  Choose a new environment and an absent test-bundle destination outside the
  source checkout. Use separate destinations for separate wheel variants.
- This recipe's route checker targets standard GIL-enabled x86-64 CPython 3.11
  and 3.12 with binary64 arithmetic on Windows or Linux. It does not qualify
  other architectures or free-threaded interpreters. This bounded validation
  scope does not impose new restrictions on the package's public support policy.

Create a fresh disposable virtual environment, then activate it using the
activation command for your shell. From then on, `python` must refer to that
environment, not your working environment:

```shell
python -m venv "/absolute/path/dwd-wheel-env"
```

After activation, confirm its interpreter and prefix:

```shell
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

The installation step below installs test dependencies into this disposable
environment. It requires the dependencies to be available from your configured
package index or wheel cache. Do not substitute an editable installation.

## Install, verify and test

Run these commands with the activated environment. They use absolute paths, so
the installation and bundle preparation do not depend on the current directory:

```shell
python -I -B "/absolute/path/dwd-source/tests/ci_artifacts.py" install --wheel-dir "/absolute/path/dwd-wheels" --extras test
python -I -B "/absolute/path/dwd-source/tests/ci_artifacts.py" bundle --source "/absolute/path/dwd-source" --destination "/absolute/path/dwd-installed-tests"
cd "/absolute/path/dwd-installed-tests"
python -I -B tests/ci_artifacts.py check --dwd-wheel-dir "/absolute/path/dwd-wheels" --backend native --route-contract current --expected-platform windows --expected-python 3.12
python -I -B -m pytest tests -o "python_files=test*.py" -ra
```

Set `--expected-platform` to `linux` on Linux and `--expected-python` to `3.11`
when that is the selected interpreter. These flags assert the intended runtime;
they do not emulate it. The `cd` example works in PowerShell and POSIX shells;
with Windows Command Prompt, use `cd /d` when changing drives.

Require the artifact check to succeed before running pytest, and require both
commands to succeed before claiming qualification. Do not ignore an origin,
payload or native-route failure and rely only on the later test result. Keep
the selected wheel and check output with the test results.

For a portable wheel, use a separate fresh environment and bundle, put only the
portable wheel in its wheel directory, and change `--backend native` to
`--backend portable`. This verifies that compiled helpers are absent rather than
silently accepting a failed native route. For a base-only test leg without
CVXPY, install with `--extras test-base` in its own fresh environment; optional
SOCP tests then skip. The full `test` extra includes the conic tests.

`-I` excludes the current-directory and environment-path import shortcuts;
`-B` avoids writing bytecode. Neither flag alone establishes exact-wheel
identity. Likewise, `-P`, a different pytest import mode or changing directories
alone does not replace the artifact check and detached bundle.

The `released-1.3.19` route contract describes historical DWD 1.3.19 route
expectations. Use it with that version's matching artifacts and tests. For
DWD 2.0.1 and 2.0.2 use `current`. The [CI workflow](../.github/workflows/tests.yml)
defines the build matrix. It builds a source distribution, verifies its contents, builds the wheel from
that distribution and qualifies the installed artifact. This page reuses those
helpers; it does not introduce a second build or validation implementation.

## Focused dependency checks before release

Run these cases before a release and when changing dependency minimums,
spectral fitting, sparse conversion, pandas targets or scikit-learn integration.
Use a fresh CPython 3.12 environment and the detached test bundle for the
candidate wheel. From the bundle directory, substitute the wheel's actual
absolute filename in the first command.

```shell
python -m pip install -c tests/scipy-floor.txt "/absolute/path/to/candidate.whl[test-base]"
python -m pip check
python -I -c "import numpy, pandas, scipy, sklearn; assert (numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__) == ('1.26.3', '2.2.0', '1.11.4', '1.6.1')"
python -I -B -m pytest -q tests/test_scale_relative_psd.py::test_spectral_fit_preserves_subnormal_gram_under_strict_overflow tests/test_sparse_integer_coordinates.py::test_non_native_endian_integer_cancellation tests/test_shared_fold_search.py::test_svc_fallback_with_int32_dia_array_matches_dense_search tests/test_compatibility.py::test_routed_cross_validation_without_params
python -I -B -m pytest -q tests/test_pandas_uint64_targets.py::test_signed_nullable_class_compares_exactly_with_integral_float_when_scoring
```

The numerical cases exercise a subnormal Gram matrix under strict overflow,
exact cancellation in non-native-endian sparse integers, and DIA-to-CSR
conversion during SVC search. The routing check compares scores and
predictions from three scikit-learn cross-validation functions with manual
fold fits when metadata routing is enabled and `params` is omitted. The
nullable-target check verifies that scoring distinguishes a large pandas
integer class label from its rounded floating-point neighbor. The portable
base CI job exercises pandas 2.2.2 with NumPy 2 and also runs the routing
case with scikit-learn 1.6.1.

The Linux conic CI job on Python 3.12 installs CVXPY 1.6.7 and highspy 1.7.1
from the wheel's `socp` extra. In that environment,
`python -I -B tests/highs_conic_floor_probe.py` fits the SVM through CVXPY's
conic HiGHS interface and checks equality and lower-bound multipliers on a
linear program with a known solution.

## Comparing conic fit times

The `socp` and `test` extras support Clarabel 0.11.0 and later. For comparable
timings across machines, use the [benchmark requirements](../tests/clarabel-benchmark.txt)
in each isolated benchmark environment. Install the DWD wheel with either the
`socp` or `test` extra to include CVXPY. Then run these commands from the
matching DWD source directory.

```shell
python -m pip install -r tests/clarabel-benchmark.txt
python -I -c "import clarabel, cvxpy; assert 'CLARABEL' in cvxpy.installed_solvers(); print(clarabel.__version__)"
```

Record the imported version with each result. For direct Clarabel fits, use one
estimator worker, set `max_threads=1`, and verify the fitted linear solver
reports one thread. Compare the same timed stage on both machines.
