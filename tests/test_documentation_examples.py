"""Validate documented imports and search construction without running fits."""
import ast
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'doc' / 'example_notebooks'


def notebook(name):
    return json.loads((NOTEBOOKS / name).read_text(encoding='utf-8'))


def code_tree(cell):
    source = ''.join(cell['source'])
    # The display-only IPython magic is the sole non-Python syntax here.
    source = '\n'.join(line for line in source.splitlines()
                       if not line.lstrip().startswith('%matplotlib '))
    return ast.parse(source)


class CapturedEstimator:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_count = 0
        type(self).instances.append(self)

    def fit(self, X, y):
        self.fit_count += 1
        return self

    def score(self, X, y):
        return .5


class DocumentationExampleTests(unittest.TestCase):
    def test_notebook_and_figure_dwd_imports_resolve_without_fitting(self):
        trees = [ast.parse((ROOT / 'doc/figures/make_figures.py').read_text(encoding='utf-8'))]
        for name in ('Basic functionality.ipynb', '2D visualization.ipynb'):
            trees.extend(code_tree(cell) for cell in notebook(name)['cells']
                         if cell['cell_type'] == 'code')
        imports = {(node.module, alias.name) for tree in trees for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module.startswith('dwd')
                   for alias in node.names}
        self.assertTrue(imports)
        for module_name, attribute in sorted(imports):
            with self.subTest(module=module_name, attribute=attribute):
                self.assertNotIn(module_name, ('dwd', 'dwd.dwd'))
                if module_name == 'dwd.socp_dwd' and importlib.util.find_spec('cvxpy') is None:
                    self.skipTest('SOCP imports require the documented optional cvxpy dependency.')
                self.assertTrue(hasattr(importlib.import_module(module_name), attribute))

    def test_notebooks_contain_no_duplicate_literal_dictionary_keys(self):
        for name in ('Basic functionality.ipynb', '2D visualization.ipynb'):
            for index, cell in enumerate(notebook(name)['cells']):
                if cell['cell_type'] != 'code':
                    continue
                for node in ast.walk(code_tree(cell)):
                    if isinstance(node, ast.Dict):
                        keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
                        with self.subTest(notebook=name, cell=index):
                            self.assertEqual(len(keys), len(set(keys)))

    def test_kernel_example_cells_fit_once_and_search_both_gamma_values(self):
        matched = set()
        for cell in notebook('Basic functionality.ipynb')['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = code_tree(cell)
            classes = {node.func.id for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            targets = classes.intersection({'KernGDWD', 'KernGDWDCV'})
            if not targets:
                continue
            self.assertEqual(len(targets), 1)
            kind = targets.pop()
            matched.add(kind)
            # Imports have their own real-import test; replace only estimators
            # so executing the actual notebook cell cannot invoke a solver.
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            CapturedEstimator.instances = []
            namespace = {'np': np, 'KernGDWD': CapturedEstimator,
                         'KernGDWDCV': CapturedEstimator, 'X': [[0.], [1.]], 'y': [0, 1]}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tree, '<notebook example>', 'exec'), namespace)
            self.assertEqual(len(CapturedEstimator.instances), 1)
            estimator = CapturedEstimator.instances[0]
            self.assertEqual(estimator.fit_count, 1)
            if kind == 'KernGDWDCV':
                self.assertEqual(estimator.parameters['kernel_kws_vals'], [{'gamma': 1}, {'gamma': 2}])
                self.assertEqual(len(estimator.parameters['lambd_vals']), 3)
                self.assertEqual(len(estimator.parameters['q_vals']), 2)
            else:
                self.assertEqual(estimator.parameters['kernel_kws'], {'gamma': .1})
        self.assertEqual(matched, {'KernGDWD', 'KernGDWDCV'})

    def test_updated_notebooks_do_not_present_stale_execution_results(self):
        for name in ('Basic functionality.ipynb', '2D visualization.ipynb'):
            doc = notebook(name)
            notice = ''.join(doc['cells'][0]['source']).lower()
            self.assertIn('full notebook execution has not been rerun', notice)
            self.assertIn('retained as historical results', notice)
            checked = 0
            for cell in doc['cells']:
                if cell['cell_type'] != 'code':
                    continue
                nodes = list(ast.walk(code_tree(cell)))
                imports_estimator = any(
                    isinstance(node, ast.ImportFrom) and node.module.startswith('dwd') and
                    any(alias.name in {'DWD', 'GenDWD', 'GenDWDCV', 'KernGDWD', 'KernGDWDCV'}
                        for alias in node.names) for node in nodes)
                searches = [node for node in nodes if isinstance(node, ast.Call) and
                            isinstance(node.func, ast.Name) and node.func.id == 'GridSearchCV']
                if imports_estimator or searches:
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                    checked += 1
                for search in searches:
                    workers = next(keyword.value for keyword in search.keywords if keyword.arg == 'n_jobs')
                    self.assertEqual(ast.literal_eval(workers), 1)
            self.assertEqual(checked, 6 if name == 'Basic functionality.ipynb' else 2)


if __name__ == '__main__':
    unittest.main()
