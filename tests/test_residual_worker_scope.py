"""Solver-local residual executor ownership and cleanup contracts."""
import contextvars, ctypes, gc, sys, threading, types, unittest, weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
from dwd import _compiled_residual as c
from dwd import _residual_worker_scope as s

def data(n=65):
    rng=np.random.default_rng(23)
    return rng.normal(size=(n,n)),rng.normal(size=n),rng.normal(size=n),.125,.25
def bits(values):return [a.tobytes() for a in values]

@unittest.skipIf(c._ACCEL is None, "Optional compiled residual extension unavailable.")
class WorkerScopeTests(unittest.TestCase):
    def setUp(self):
        self.args=data()
        self.budget=self.enterContext(patch.object(c,'_worker_count',return_value=2))
    def assert_drained(self,executor):
        self.assertTrue(executor._shutdown)
        self.assertTrue(all(not t.is_alive() for t in executor._threads))
    def test_reuses_one_pool_then_drains_and_does_not_cross_calls(self):
        with s.ResidualWorkerScope() as scope:
            a=c.compiled_values(*self.args);pool=scope.executor
            b=c.compiled_values(*self.args)
            self.assertIs(scope.executor,pool);self.assertEqual(bits(a),bits(b))
        self.assert_drained(pool);self.assertIsNone(s._SCOPE.get())
        with s.ResidualWorkerScope() as other:
            c.compiled_values(*self.args);self.assertIsNot(other.executor,pool)
    def test_missing_helper_small_and_serial_never_construct_pool(self):
        for args,helper,budget in ((data(7),c._ACCEL,2),(self.args,None,2),(self.args,c._ACCEL,1)):
            with s.ResidualWorkerScope() as scope,patch.object(c,'_ACCEL',helper),patch.object(c,'_worker_count',return_value=budget):
                c.compiled_values(*args);self.assertIsNone(scope.executor)
    def test_idle_reused_pool_does_not_retain_call_arrays(self):
        with s.ResidualWorkerScope() as scope:
            args=data();references=[weakref.ref(a) for a in args[:3]]
            got=c.compiled_values(*args);references += [weakref.ref(a) for a in got]
            self.assertIsNotNone(scope.executor)
            del args,got
            gc.collect()
            self.assertTrue(all(ref() is None for ref in references))
    def test_budget_changes_replace_and_serial_retires_pool(self):
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);old=scope.executor
            self.budget.return_value=3;c.compiled_values(*self.args)
            self.assert_drained(old);self.assertEqual(scope.workers,3)
            old=scope.executor;self.budget.return_value=1;c.compiled_values(*self.args)
            self.assert_drained(old);self.assertIsNone(scope.executor)
    def test_nested_solver_scope_restores_outer_without_sharing(self):
        with s.ResidualWorkerScope() as outer:
            first=c.compiled_values(*self.args);pool=outer.executor
            with s.ResidualWorkerScope() as inner:
                second=c.compiled_values(*self.args);inside=inner.executor
                self.assertIsNot(pool,inside)
            self.assert_drained(inside);self.assertIs(s._SCOPE.get(),outer)
            self.assertEqual(bits(first),bits(second));c.compiled_values(*self.args)
            self.assertIs(outer.executor,pool)
    def test_concurrent_and_copied_contexts_never_share_owner_pool(self):
        with s.ResidualWorkerScope() as scope:
            expected=c.compiled_values(*self.args);owner=scope.executor
            ctx=contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=2) as threads:
                result=threads.submit(ctx.run,c.compiled_values,*self.args).result()
                def independent():
                    with s.ResidualWorkerScope() as local:
                        got=c.compiled_values(*self.args);return got,local.executor
                value,other=threads.submit(independent).result()
            self.assertIs(scope.executor,owner);self.assertIsNot(owner,other)
            self.assertEqual(bits(expected),bits(result));self.assertEqual(bits(value),bits(result))
            self.assert_drained(other)
    def test_changed_native_hook_uses_original_per_call_executor(self):
        original=c._ACCEL
        with s.ResidualWorkerScope() as scope,patch.object(c,'_ACCEL',types.SimpleNamespace(evaluate=original.evaluate)):
            a=c.compiled_values(*self.args);self.assertIsNotNone(a);self.assertIsNone(scope.executor)
    def test_changed_executor_factory_keeps_original_context_semantics(self):
        events=[]
        class Observed(ThreadPoolExecutor):
            def __enter__(self):events.append('enter');return super().__enter__()
            def __exit__(self,*exc):events.append('exit');return super().__exit__(*exc)
        with s.ResidualWorkerScope() as scope,patch.object(c,'ThreadPoolExecutor',Observed):
            c.compiled_values(*self.args);c.compiled_values(*self.args)
            self.assertIsNone(scope.executor)
        self.assertEqual(events,['enter','exit','enter','exit'])
    @unittest.skipUnless(sys.platform == "win32", "Windows CRT floating-state control required.")
    def test_custom_conversion_cannot_poison_later_standard_call(self):
        control=ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes=[ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint]
        control.restype=ctypes.c_int
        class ChangingFloat:
            def __float__(self):
                current=ctypes.c_uint()
                assert control(ctypes.byref(current),0x100,0x300)==0
                return .125
        args=list(self.args);args[3]=ChangingFloat()
        with s.ResidualWorkerScope() as scope:
            expected=c.compiled_values(*self.args)
            self.assertIsNone(c.compiled_values(*args))
            self.assertEqual(bits(c.compiled_values(*self.args)),bits(expected))
    def test_worker_exception_is_original_and_pool_is_retired(self):
        failure=RuntimeError('worker value conversion')
        class BrokenFloat:
            def __float__(self):raise failure
        args=list(self.args);args[3]=BrokenFloat()
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);pool=scope.executor
            with self.assertRaises(RuntimeError) as caught:c.compiled_values(*args)
            self.assertIs(caught.exception,failure);self.assertIsNone(scope.executor)
            self.assert_drained(pool)
            self.assertIsNotNone(c.compiled_values(*self.args));self.assertIsNot(scope.executor,pool)
    def test_main_interrupt_joins_workers_and_preserves_exception(self):
        failure=KeyboardInterrupt('main poll interruption')
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);pool=scope.executor
            with patch.object(c,'wait',side_effect=failure),self.assertRaises(KeyboardInterrupt) as caught:
                c.compiled_values(*self.args)
            self.assertIs(caught.exception,failure);self.assertIsNone(scope.executor)
            self.assert_drained(pool)
    def test_partial_submit_failure_joins_prior_future(self):
        failure=RuntimeError('submit failed')
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);pool=scope.executor;real=pool.submit;count=0
            def submit(*args,**kw):
                nonlocal count
                count+=1
                if count==2:raise failure
                return real(*args,**kw)
            with patch.object(pool,'submit',side_effect=submit),self.assertRaises(RuntimeError) as caught:
                c.compiled_values(*self.args)
            self.assertIs(caught.exception,failure);self.assert_drained(pool)
    def test_domain_refusal_does_not_poison_next_call_or_publish_partial_values(self):
        with s.ResidualWorkerScope() as scope:
            expected=c.compiled_values(*self.args);pool=scope.executor
            bad=list(self.args);bad[0]=bad[0].copy();bad[0][-1,-1]=np.inf
            self.assertIs(c._compiled_values_or_refusal(*bad),c._COMPILED_DOMAIN_REFUSAL)
            self.assertEqual(bits(c.compiled_values(*self.args)),bits(expected))
            self.assertIs(scope.executor,pool)
    def test_broken_executor_propagates_then_future_call_builds_new(self):
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);pool=scope.executor;pool.shutdown(wait=True)
            with self.assertRaises(RuntimeError):c.compiled_values(*self.args)
            self.assertIsNone(scope.executor);self.assertIsNotNone(c.compiled_values(*self.args))
    def test_busy_scope_falls_back_without_reentrant_pool_deadlock(self):
        with s.ResidualWorkerScope() as scope:
            c.compiled_values(*self.args);pool=scope.executor;scope.busy=True
            try:self.assertIsNotNone(c.compiled_values(*self.args))
            finally:scope.busy=False
            self.assertIs(scope.executor,pool)
    def test_decorator_drains_on_return_callback_exception_and_recursion(self):
        pools=[];failure=ValueError('callback')
        @s.scoped_residual_workers
        def solve(depth=0,fail=False):
            got=c.compiled_values(*self.args);pools.append(s._SCOPE.get().executor)
            if depth:solve(depth-1)
            if fail:raise failure
            return got
        solve(1)
        with self.assertRaises(ValueError) as caught:solve(fail=True)
        self.assertIs(caught.exception,failure);self.assertEqual(len(set(pools)),3)
        for pool in pools:self.assert_drained(pool)
        self.assertIsNone(s._SCOPE.get())
    def test_actual_estimator_callback_exception_drains_and_clears_fitted_state(self):
        from dwd.gen_kern_dwd import KernGDWD
        from sklearn.preprocessing import StandardScaler
        rng=np.random.default_rng(381);X=StandardScaler().fit_transform(rng.normal(size=(256,8)));y=np.tile([-1,1],128)
        pools=[];failure=RuntimeError('actual fit callback')
        def callback(state):
            scope=s._SCOPE.get()
            if scope.executor is not None:
                pools.append(scope.executor)
                raise failure
        model=KernGDWD(kernel='rbf',kernel_kws={'gamma':1e-7},lambd=1e-10,
                       max_iter=4,stopping='fixed',callback=callback)
        with self.assertRaises(RuntimeError) as caught:model.fit(X,y)
        self.assertIs(caught.exception,failure);self.assertTrue(pools)
        for pool in pools:self.assert_drained(pool)
        self.assertFalse(hasattr(model,'dual_coef_'));self.assertIsNone(s._SCOPE.get())

if __name__=='__main__':unittest.main()
