"""Small, tested sundials4py bridge; no network physics or event actions.

This adapter uses IDAS without sensitivities and dense linear algebra. Network
assembly may register a structurally colored numerical Jacobian callback.
The owner must close the session before discarding it. NumPy result arrays are
copies; callback arrays are borrowed views and must not escape the callback.
"""

from dataclasses import dataclass

import numpy as np

from errors import (TrialDomainError, LookupBoundsError, ModelDomainExceeded,
                    SolverConvergenceError, SolverSetupError)


@dataclass
class IdaStep:
    time: float
    y: np.ndarray
    ydot: np.ndarray
    roots: np.ndarray
    flag: int


class IdaSession:
    def __init__(self, residual, y0, ydot0, differential, *, time=0.0,
                 rtol=1e-8, atol, roots=None, root_directions=(),
                 suppress_algebraic_error=False):
        # Optional dependency: ordinary component imports still work on 3.11.
        from sundials4py import core, idas

        self.core, self.idas = core, idas
        self.residual, self.roots = residual, roots
        self.callback_error = None
        self.last_trial_error = None
        self.recoverable_errors = 0
        self.callback_calls = 0
        self.structured_calls = self.structured_residuals = self.structured_colors = 0
        self.structured_verified = False
        self._structured_callback = None
        self.root_directions = tuple(root_directions)
        self.root_count = len(self.root_directions)
        if bool(self.root_count) != (roots is not None):
            raise ValueError("Supply one direction per root, or no root callback")
        arrays = [np.asarray(a, dtype=float) for a in (y0, ydot0, differential, atol)]
        y0, ydot0, differential, atol = arrays
        if y0.ndim != 1 or not y0.size or any(a.shape != y0.shape for a in arrays):
            raise ValueError("State, derivatives, variable IDs and tolerances must be equal-length vectors")
        if (any(not np.isfinite(a).all() for a in arrays)
                or not np.isfinite(time) or not np.isfinite(rtol) or rtol <= 0
                or np.any(atol <= 0) or not np.isin(differential, [0, 1]).all()):
            raise ValueError("Require finite data, positive tolerances and 0/1 variable IDs")
        self.time = float(time)
        status, self.context = core.SUNContext_Create(core.SUN_COMM_NULL)
        self._check(status, "SUNContext_Create")
        self.y = self._vector(y0)
        self.ydot = self._vector(ydot0)
        self.ids = self._vector(differential)
        self.atol = self._vector(atol)
        self.matrix = core.SUNDenseMatrix(y0.size, y0.size, self.context)
        self.linear_solver = core.SUNLinSol_Dense(self.y, self.matrix, self.context)
        self.owner = idas.IDACreate(self.context)
        self.mem = self.owner.get()  # IDAView owns memory; functions take its capsule.
        self._callbacks = (self._residual_callback, self._root_callback)
        try:
            self._check(idas.IDAInit(self.mem, self._callbacks[0], time, self.y, self.ydot), "IDAInit")
            self._check(idas.IDASetLinearSolver(self.mem, self.linear_solver, self.matrix), "linear solver")
            self._check(idas.IDASVtolerances(self.mem, rtol, self.atol), "tolerances")
            self._check(idas.IDASetId(self.mem, self.ids), "variable IDs")
            self._check(idas.IDASetSuppressAlg(self.mem, bool(suppress_algebraic_error)), "algebraic error control")
            self._check(idas.IDASetMaxNumSteps(self.mem, 10000), "step budget")
            if roots is not None:
                self._check(idas.IDARootInit(self.mem, self.root_count, self._callbacks[1]), "root registration")
                self._check(idas.IDASetRootDirection(self.mem, list(root_directions)), "root directions")
        except Exception:
            self.close()
            raise

    def _vector(self, values):
        vector = self.core.N_VNew_Serial(len(values), self.context)
        self.core.N_VGetNumpyArray(vector)[:] = values
        return vector

    def _check(self, status, operation):
        if status < 0:
            error = self.callback_error or self.last_trial_error
            if isinstance(error, LookupBoundsError):
                raise ModelDomainExceeded(str(error), time=self.time) from error
            if self.callback_error is not None:
                raise self.callback_error
            # Only solve/IC return codes use the IDA convergence-status namespace.
            if operation in ('integration', 'consistent initialization') and status in (-1, -2, -3, -4, -9, -11, -12, -13, -14):
                raise SolverConvergenceError(operation, status, self.time,
                                             retryable=status in (-3, -4, -11, -13)) from error
            raise SolverSetupError(f"{operation} failed with SUNDIALS status {status}") from error

    def _residual_callback(self, t, y, ydot, out, _):
        self.callback_calls += 1
        try:
            view = self.core.N_VGetNumpyArray
            result = view(out)
            self.residual(t, view(y), view(ydot), result)
            if not np.isfinite(result).all():
                raise TrialDomainError("Non-finite assembled residual")
            self.last_trial_error = None
            return 0
        except (TrialDomainError, LookupBoundsError) as error:
            self.recoverable_errors += 1
            self.last_trial_error = error
            return 1
        except Exception as error:
            self.callback_error = error
            return -1

    def _root_callback(self, t, y, ydot, out, _):
        try:
            view = self.core.N_VGetNumpyArray
            self.roots(t, view(y), view(ydot), out)
            if not np.isfinite(out).all():
                raise ValueError("Non-finite root margin")
            return 0
        except Exception as error:
            # Root functions do not have the residual's recoverable contract.
            self.callback_error = error
            return -1

    def consistent(self, horizon=0.01):
        """Hold differential y fixed; correct algebraic y and differential ydot."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Initialization horizon must be positive and finite")
        self.callback_error = self.last_trial_error = None
        self._check(self.idas.IDACalcIC(self.mem, self.idas.IDA_YA_YDP_INIT,
                                       self.time + horizon), "consistent initialization")
        self._check(self.idas.IDAGetConsistentIC(self.mem, self.y, self.ydot), "consistent state retrieval")
        return self.snapshot()

    def snapshot(self, flag=0):
        roots = np.zeros(self.root_count, dtype=np.int32)
        if flag == self.idas.IDA_ROOT_RETURN:
            self._check(self.idas.IDAGetRootInfo(self.mem, roots), "root identity")
        view = self.core.N_VGetNumpyArray
        return IdaStep(self.time, view(self.y).copy(), view(self.ydot).copy(), roots, flag)

    def advance(self, target):
        """Return at the first root or requested endpoint, never integrate past it."""
        if not np.isfinite(target) or target <= self.time:
            raise ValueError("Target must be finite and later than the current time")
        self._check(self.idas.IDASetStopTime(self.mem, target), "stop time")
        self.callback_error = self.last_trial_error = None
        flag, reached = self.idas.IDASolve(self.mem, target, self.y, self.ydot, self.idas.IDA_NORMAL)
        self._check(flag, "integration")
        self.time = reached
        return self.snapshot(flag)

    def restart(self, state, *, horizon=0.01, rtol=None, max_step=None):
        """Same-size restart after the owner changes equations; resets IDA history."""
        view = self.core.N_VGetNumpyArray
        if state.y.shape != view(self.y).shape or state.ydot.shape != state.y.shape:
            raise ValueError("A changed vector size requires a new session")
        self.callback_error = self.last_trial_error = None
        self._check(self.idas.IDAClearStopTime(self.mem), "clear old stop time")
        view(self.y)[:] = state.y
        view(self.ydot)[:] = state.ydot
        self.time = state.time
        self._check(self.idas.IDAReInit(self.mem, self.time, self.y, self.ydot), "restart")
        if rtol is not None:
            self._check(self.idas.IDASVtolerances(self.mem, rtol, self.atol), "restarted tolerances")
        if max_step is not None:
            self._check(self.idas.IDASetMaxStep(self.mem, max_step if np.isfinite(max_step) else 0.),
                        "restarted max step")
        if self.root_count:
            self._check(self.idas.IDARootInit(self.mem, self.root_count, self._callbacks[1]), "restarted roots")
            self._check(self.idas.IDASetRootDirection(self.mem, list(self.root_directions)), "restarted root directions")
        return self.consistent(horizon)

    def statistics(self):
        result = {"residual_callback_calls": self.callback_calls,
                  "recoverable_errors": self.recoverable_errors,
                  "structured_jacobians": self.structured_calls,
                  "structured_residuals": self.structured_residuals,
                  "total_residual_evaluations": self.callback_calls + self.structured_residuals}
        for key, name in (("steps", "IDAGetNumSteps"), ("jacobians", "IDAGetNumJacEvals"),
                          ("jacobian_residual_calls", "IDAGetNumLinResEvals")):
            status, value = getattr(self.idas, name)(self.mem)
            self._check(status, name)
            result[key] = value
        return result

    def close(self):
        # Destroy IDA before its vectors, linear solver, context and callbacks.
        self.mem = None
        self.owner = None
        self._callbacks = ()
        self._structured_callback = None
        # Release context-dependent native objects explicitly. Callback/error
        # reference cycles can otherwise make cyclic GC free the context first.
        self.linear_solver = None
        self.matrix = None
        self.atol = self.ids = self.ydot = self.y = None
        self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
