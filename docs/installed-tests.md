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

- For DWD 2.0.0, use the source tree and tests from the 2.0.0 tag or its source
  distribution. This version uses the `current` route contract. For other
  versions, select the source and checker that accompany the wheel.
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
DWD 2.0.0 use `current`. The [CI workflow](../.github/workflows/tests.yml)
defines the build matrix. It builds a source distribution, verifies its contents, builds the wheel from
that distribution and qualifies the installed artifact. This page reuses those
helpers; it does not introduce a second build or validation implementation.
