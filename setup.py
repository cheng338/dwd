"""Build the optional strict-arithmetic accelerators and the portable package."""
import os
import platform
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class StrictBuildExt(build_ext):
    def run(self):
        super().run()
        if not self.dry_run:
            for extension in self.extensions:
                target = Path(self.get_ext_fullpath(extension.name))
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(
                        f'Required extension was not produced: {target}'
                    )

    def build_extensions(self):
        zig = os.environ.get('DWD_BUILD_ZIG')
        flags = ['-O3', '-g0', '-std=c11', '-ffp-contract=off', '-fno-fast-math',
                 '-fno-associative-math', '-fno-unsafe-math-optimizations',
                 '-fno-finite-math-only']
        if zig:
            # An explicit build-only option for hosts without a system compiler.
            # Runtime installations never download or invoke build tools.
            if sys.platform != 'win32':
                raise RuntimeError('DWD_BUILD_ZIG currently supports Windows builds.')
            if sysconfig.get_platform().lower() != 'win-amd64':
                raise RuntimeError('DWD_BUILD_ZIG requires a Windows AMD64 Python build.')
            for extension in self.extensions:
                target = Path(self.get_ext_fullpath(extension.name)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                # Zig otherwise targets the build host's CPU (including AVX512
                # on some hosts). A win_amd64 wheel must run on baseline x86-64.
                limited = extension.py_limited_api
                abi_flags = ['-DPy_LIMITED_API=0x030B0000'] if limited else []
                python_library = ('python3.lib' if limited else
                                  f'python{sys.version_info.major}{sys.version_info.minor}.lib')
                command = [zig, 'cc', *flags, '-march=x86_64',
                           '-s', '-shared', *abi_flags,
                           '-I' + sysconfig.get_path('include'),
                           str(Path(sys.base_prefix) / 'libs' / python_library),
                           *extension.sources, '-o', str(target)]
                subprocess.run(command, check=True)
                # Linker import libraries are build products, not runtime data.
                for stem in (target.stem, *(Path(source).stem for source in extension.sources)):
                    for suffix in ('.pdb', '.lib', '.exp'):
                        product = target.parent / (stem + suffix)
                        if product.is_file():
                            product.unlink()
            return
        if self.compiler.compiler_type == 'msvc':
            flags = ['/O2', '/std:c11', '/fp:strict']
        for extension in self.extensions:
            if extension.name == 'dwd._feature_expansion_accel':
                if self.compiler.compiler_type not in ('msvc', 'unix'):
                    raise RuntimeError('Unreviewed feature-expansion compiler family: '
                                       + self.compiler.compiler_type)
                extension.extra_compile_args = (flags + ['-march=x86-64']
                                                if self.compiler.compiler_type == 'unix'
                                                else flags)
            else:
                extension.extra_compile_args = flags
        super().build_extensions()


portable = os.environ.get('DWD_BUILD_ACCEL', '1') == '0'
# This additional accelerator uses CPython internals; other supported builds
# retain the existing limited-ABI accelerators and the original Python fallback.
feature_expansion = (
    not portable
    and sys.implementation.name == 'cpython'
    and sys.version_info[:2] == (3, 12)
    and sys.platform in ('win32', 'linux')
    and sys.maxsize > 2**32
    and platform.machine().lower() in ('amd64', 'x86_64')
    and sysconfig.get_config_var('Py_GIL_DISABLED') in (None, 0)
)
extensions = [] if portable else [Extension(
    'dwd._residual_accel',
    sources=['dwd/residual_bridge.c', 'dwd/residual_core.c', 'dwd/dense_score_core.c'],
    define_macros=[('Py_LIMITED_API', '0x030B0000')],
    py_limited_api=True,
    optional=False,
), Extension(
    'dwd._ordered_reduction_accel',
    sources=['dwd/ordered_reduction_bridge.c'],
    depends=['dwd/dense_score_core.c'],
    define_macros=[('Py_LIMITED_API', '0x030B0000')],
    py_limited_api=True,
    optional=False,
)]
if feature_expansion:
    extensions.append(Extension(
        'dwd._feature_expansion_accel',
        sources=['dwd/feature_expansion_bridge.c'],
        depends=['dwd/feature_expansion_core.c', 'dwd/dense_score_core.c'],
        optional=False,
    ))
# A wheel containing the full-ABI extension must not advertise an abi3 tag.
setup(ext_modules=extensions, cmdclass={'build_ext': StrictBuildExt},
      options={} if portable or feature_expansion else
      {'bdist_wheel': {'py_limited_api': 'cp311'}})
