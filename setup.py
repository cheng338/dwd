"""Build the optional strict-arithmetic accelerator and the portable package."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class StrictBuildExt(build_ext):
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
            for extension in self.extensions:
                target = Path(self.get_ext_fullpath(extension.name)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [zig, 'cc', *flags, '-s', '-shared', '-DPy_LIMITED_API=0x030B0000',
                           '-I' + sysconfig.get_path('include'),
                           str(Path(sys.base_prefix) / 'libs/python3.lib'),
                           *extension.sources, '-o', str(target)]
                subprocess.run(command, check=True)
                # Linker import libraries are build products, not runtime data.
                for stem in (target.stem, 'residual_bridge'):
                    for suffix in ('.pdb', '.lib', '.exp'):
                        product = target.parent / (stem + suffix)
                        if product.is_file():
                            product.unlink()
            return
        if self.compiler.compiler_type == 'msvc':
            flags = ['/O2', '/std:c11', '/fp:strict']
        for extension in self.extensions:
            extension.extra_compile_args = flags
        super().build_extensions()


portable = os.environ.get('DWD_BUILD_ACCEL', '1') == '0'
extensions = [] if portable else [Extension(
    'dwd._residual_accel',
    sources=['dwd/residual_bridge.c', 'dwd/residual_core.c'],
    define_macros=[('Py_LIMITED_API', '0x030B0000')],
    py_limited_api=True,
    optional=True,
)]
setup(ext_modules=extensions, cmdclass={'build_ext': StrictBuildExt},
      options={} if portable else {'bdist_wheel': {'py_limited_api': 'cp311'}})
