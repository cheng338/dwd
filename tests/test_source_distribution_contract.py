"""Keep bundled source-only build/documentation tests runnable after extraction."""
from pathlib import Path
import re
import tomllib
import unittest
from unittest.mock import patch

from setuptools._distutils.filelist import FileList


ROOT = Path(__file__).resolve().parents[1]


class SourceDistributionContractTests(unittest.TestCase):
    def test_build_policy_dependency_is_declared_for_both_test_extras_only(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
        def names(requirements):
            return {re.split(r'[<>=!~;\[ ]', value, maxsplit=1)[0].lower()
                    for value in requirements}
        for extra in ('test', 'test-base'):
            with self.subTest(extra=extra):
                self.assertIn('setuptools', names(project['optional-dependencies'][extra]))
        self.assertNotIn('setuptools', names(project['dependencies']))

    def test_manifest_includes_every_bundled_documentation_test_fixture(self):
        required = {
            'doc/example_notebooks/Basic functionality.ipynb',
            'doc/example_notebooks/2D visualization.ipynb',
            'doc/figures/make_figures.py',
            'tests/test_documentation_examples.py',
            'tests/_feature_expansion_reference.py',
            'tests/_feature_expansion_test_support.py',
            'tests/test_feature_expansion_core.py',
            'tests/test_feature_expansion_facade.py',
            'tests/test_feature_expansion_beta.py',
            'tests/test_feature_expansion_fallback.py',
            'tests/test_feature_reporting_dispatch.py',
        }
        for member in required:
            with self.subTest(member=member):
                self.assertTrue((ROOT / member).is_file(), 'Source archive omitted ' + member)
        # Exercise setuptools' manifest parser, without building or importing DWD.
        inventory = FileList()
        inventory.set_allfiles([str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()])
        with patch('setuptools._distutils.filelist.log.warning'):
            for line in (ROOT / 'MANIFEST.in').read_text(encoding='utf-8').splitlines():
                line = line.split('#', 1)[0].strip()
                if line:
                    inventory.process_template_line(line)
        selected = {Path(name).as_posix() for name in inventory.files}
        self.assertTrue(required <= selected, 'MANIFEST omits test fixtures: ' + repr(sorted(required - selected)))


    def test_feature_expansion_c_sources_and_license_are_distributed(self):
        # In installed-wheel jobs ROOT is a source-free test bundle. Bind the
        # numerical source/license files to the selected installed package.
        import dwd
        package = Path(dwd.__file__).resolve().parent
        names = ('feature_expansion_bridge.c', 'feature_expansion_core.c',
                 'dense_score_core.c', 'CPYTHON-LICENSE.txt')
        for name in names:
            with self.subTest(member=name):
                self.assertTrue((package / name).is_file(), 'Package omitted ' + name)
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        data = project['tool']['setuptools']['package-data']['dwd']
        self.assertIn('*.c', data)
        self.assertIn('CPYTHON-LICENSE.txt', data)
        inventory = FileList()
        inventory.set_allfiles([str(Path('dwd') / name) for name in names])
        with patch('setuptools._distutils.filelist.log.warning'):
            for line in (ROOT / 'MANIFEST.in').read_text(encoding='utf-8').splitlines():
                line = line.split('#', 1)[0].strip()
                if line:
                    inventory.process_template_line(line)
        self.assertTrue({'dwd/' + name for name in names} <=
                        {Path(name).as_posix() for name in inventory.files})


if __name__ == '__main__':
    unittest.main()
