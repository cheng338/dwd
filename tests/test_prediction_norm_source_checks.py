"""Source provenance checks may share paths only within one initialization."""
from collections import Counter
import functools
import importlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from dwd import _prediction_norms as norms


def function_at(filename, name='checked', module='example'):
    namespace={'__name__':module}
    exec(compile(f'def {name}():\n    return None\n',str(filename),'exec'),namespace)
    return namespace[name]


class PredictionNormSourceChecksTests(unittest.TestCase):
    def test_initialization_resolves_each_distinct_path_once(self):
        original=Path.resolve
        calls=[]
        def resolve(path,*args,**kwargs):
            calls.append(path)
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve):
            self.assertTrue(norms._source_functions_ok())
        counts=Counter(calls)
        self.assertGreater(len(counts),1)
        self.assertTrue(all(count==1 for count in counts.values()),counts)

    def test_direct_calls_do_not_reuse_previous_resolutions(self):
        original=Path.resolve
        calls=[]
        def resolve(path,*args,**kwargs):
            calls.append(path)
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve):
            self.assertTrue(norms._source_function(norms._ROW_NORMS,norms.extmath,'row_norms'))
            first=list(calls)
            calls.clear()
            self.assertTrue(norms._source_function(norms._ROW_NORMS,norms.extmath,'row_norms'))
            self.assertEqual(calls,first)
        self.assertGreater(len(first),len(set(first)))

    def test_reload_resolves_paths_again(self):
        original=Path.resolve
        calls=[]
        def resolve(path,*args,**kwargs):
            calls.append(path)
            return original(path,*args,**kwargs)
        try:
            with patch.object(Path,'resolve',resolve):
                importlib.reload(norms)
                first=set(calls)
                self.assertTrue(norms._SOURCE_OK)
                calls.clear()
                importlib.reload(norms)
                self.assertTrue(norms._SOURCE_OK)
                self.assertEqual(set(calls),first)
        finally:
            importlib.reload(norms)

    def test_reload_does_not_reuse_a_previously_valid_source(self):
        original=norms.extmath.row_norms
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            return original(*args,**kwargs)
        try:
            with patch.object(norms.extmath,'row_norms',wrapped):
                importlib.reload(norms)
                self.assertFalse(norms._SOURCE_OK)
        finally:
            importlib.reload(norms)
        self.assertTrue(norms._SOURCE_OK)

    def test_source_module_name_and_function_name_are_required(self):
        module=types.SimpleNamespace(__file__=__file__,__name__='example')
        correct=function_at(__file__)
        self.assertTrue(norms._source_function(correct,module,'checked'))
        for changed,expected in [(function_at(__file__,module='other'),'checked'),
                                 (function_at(__file__,name='other'),'checked'),
                                 (function_at(Path(__file__).with_name('foreign.py')),'checked')]:
            with self.subTest(function=changed,expected=expected):
                self.assertFalse(norms._source_function(changed,module,expected))

    def test_only_the_supported_decorator_source_is_allowed(self):
        module=types.SimpleNamespace(__file__=__file__,__name__='example')
        original=function_at(__file__)
        supported=function_at(norms._param_validation.__file__,name='wrapper',module='decorators')
        supported=functools.wraps(original)(supported)
        self.assertTrue(norms._source_function(supported,module,'checked'))
        foreign=function_at(Path(__file__).with_name('foreign_wrapper.py'),name='wrapper')
        foreign=functools.wraps(original)(foreign)
        self.assertFalse(norms._source_function(foreign,module,'checked'))
        # An allowed outer decorator cannot hide a foreign inner wrapper.
        supported.__wrapped__=foreign
        self.assertFalse(norms._source_function(supported,module,'checked'))

    def test_invalid_paths_and_callables_still_decline(self):
        function=function_at(__file__)
        for module in [object(),types.SimpleNamespace(__file__=None,__name__='example'),
                       types.SimpleNamespace(__file__=17,__name__='example')]:
            with self.subTest(module=module):
                self.assertFalse(norms._source_function(function,module,'checked'))
        module=types.SimpleNamespace(__file__=__file__,__name__='example')
        self.assertFalse(norms._source_function(object(),module,'checked'))
        for error in [ValueError('invalid path'),TypeError('invalid path')]:
            with patch.object(Path,'resolve',side_effect=error):
                self.assertFalse(norms._source_function(function,module,'checked'))

    def test_resolution_errors_are_not_cached_or_silently_changed(self):
        module=types.SimpleNamespace(__file__=__file__,__name__='example')
        function=function_at(__file__)
        cache={}
        for error in [PermissionError('source inaccessible'),OSError('filesystem failure')]:
            with patch.object(Path,'resolve',side_effect=error):
                with self.assertRaises(type(error)):
                    norms._source_function(function,module,'checked',_resolved=cache)
            self.assertEqual(cache,{})
        self.assertTrue(norms._source_function(function,module,'checked',_resolved=cache))

    def test_initialization_still_checks_later_helpers_after_a_decline(self):
        calls=[]
        def checked(function,module,name,**kwargs):
            calls.append(name)
            if name=='_norm_sum_rbf_with_norms':
                raise OSError('last helper inaccessible')
            return False
        with patch.object(norms,'_source_function',checked):
            with self.assertRaisesRegex(OSError,'last helper'):
                norms._source_functions_ok()
        self.assertEqual(len(calls),7)

    def test_resolution_passes_do_not_retain_filesystem_state(self):
        original=Path.resolve
        target=Path(norms.extmath.__file__)
        for iteration in range(2):
            calls=[]
            def resolve(path,*args,**kwargs):
                calls.append(path)
                if iteration and path==target:
                    raise PermissionError('changed access')
                return original(path,*args,**kwargs)
            with patch.object(Path,'resolve',resolve):
                if iteration:
                    with self.assertRaisesRegex(PermissionError,'changed access'):
                        norms._source_functions_ok()
                else:
                    self.assertTrue(norms._source_functions_ok())
            self.assertIn(target,calls)

    def test_symlink_retarget_is_seen_by_the_next_direct_check(self):
        with tempfile.TemporaryDirectory(prefix='dwd-source-check-') as directory:
            root=Path(directory)
            target=root/'source.py';other=root/'other.py';link=root/'alias.py'
            target.write_text('# source\n',encoding='utf-8')
            other.write_text('# other\n',encoding='utf-8')
            try:
                link.symlink_to(target)
            except OSError as error:
                self.skipTest(f'File symlinks unavailable: {error}')
            module=types.SimpleNamespace(__file__=str(target),__name__='example')
            function=function_at(link)
            self.assertTrue(norms._source_function(function,module,'checked'))
            link.unlink()
            link.symlink_to(other)
            self.assertFalse(norms._source_function(function,module,'checked'))


if __name__=='__main__':
    unittest.main()
