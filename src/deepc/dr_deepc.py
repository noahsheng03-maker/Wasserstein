from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class DRDeePCResult:
    g: np.ndarray
    u_future: np.ndarray
    y_future_samples: list[np.ndarray]
    objective: float
    status: str


class WassersteinDRDeePC:
    def __init__(
        self,
        up: np.ndarray,
        uf: np.ndarray,
        yp_batches: list[np.ndarray],
        yf_batches: list[np.ndarray],
        input_lower: np.ndarray,
        input_upper: np.ndarray,
        output_lower: np.ndarray,
        output_upper: np.ndarray,
        y_ref: np.ndarray,
        alpha: float,
        l_obj: float = 1.0,
        l_con: float = 1.0,
        input_weights: np.ndarray | None = None,
        output_weight: float = 1.0,
        past_weight: float = 1.0,
    ) -> None:
        self.up = up
        self.uf = uf
        self.yp_batches = yp_batches
        self.yf_batches = yf_batches
        self.input_lower = input_lower
        self.input_upper = input_upper
        self.output_lower = output_lower
        self.output_upper = output_upper
        self.y_ref = y_ref
        self.alpha = alpha
        self.n_cols = up.shape[1]
        self.n_batches = len(yp_batches)
        self.input_weights = input_weights
        self.output_weight = output_weight
        self.past_weight = past_weight
        self.l_obj = l_obj
        self.l_con = l_con
        self.input_dim = uf.shape[0]
        self.output_dim = yf_batches[0].shape[0]
        self._build_problem()

    def _build_problem(self) -> None:
        self.g = cp.Variable(self.n_cols)
        self.tau = cp.Variable()
        self.s = cp.Variable(self.n_batches, nonneg=True)
        self.u_ini_param = cp.Parameter(self.up.shape[0])
        self.y_ini_param = cp.Parameter(self.yp_batches[0].shape[0])
        self.epsilon_param = cp.Parameter(nonneg=True)
        self.u_future_expr = self.uf @ self.g

        objective_terms = []
        if self.input_weights is None:
            objective_terms.append(cp.sum_squares(self.u_future_expr))
        else:
            objective_terms.append(cp.sum(cp.multiply(self.input_weights, cp.square(self.u_future_expr))))

        self.y_future_exprs = []
        constraints = [
            self.up @ self.g == self.u_ini_param,
            self.u_future_expr >= self.input_lower,
            self.u_future_expr <= self.input_upper,
        ]

        for batch_idx, (yp_i, yf_i) in enumerate(zip(self.yp_batches, self.yf_batches)):
            y_p = yp_i @ self.g
            y_f = yf_i @ self.g
            self.y_future_exprs.append(y_f)
            objective_terms.append(self.output_weight * cp.sum_squares(y_f - self.y_ref))
            objective_terms.append(self.past_weight * cp.sum_squares(y_p - self.y_ini_param))

            # CVaR-style robustified box violation surrogate
            box_violations = cp.hstack([y_f - self.output_upper, self.output_lower - y_f])
            constraints.extend([self.tau + box_violations[j] <= self.s[batch_idx] for j in range(box_violations.shape[0])])

        self.objective_expr = (
            (1.0 / self.n_batches) * cp.sum(cp.hstack(objective_terms))
            + self.l_obj * self.epsilon_param * cp.norm2(self.g)
        )
        constraints.append(
            -self.tau * self.alpha
            + self.l_con * self.epsilon_param * cp.norm2(self.g)
            + (1.0 / self.n_batches) * cp.sum(self.s)
            <= 0
        )
        self.problem = cp.Problem(cp.Minimize(self.objective_expr), constraints)

    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        epsilon: float,
        solver: str = "CLARABEL",
        solver_options: dict | None = None,
    ) -> DRDeePCResult:
        self.u_ini_param.value = np.asarray(u_ini).reshape(-1)
        self.y_ini_param.value = np.asarray(y_ini).reshape(-1)
        self.epsilon_param.value = float(epsilon)
        solver_options = {} if solver_options is None else dict(solver_options)
        try:
            self.problem.solve(solver=solver, verbose=False, warm_start=True, **solver_options)
        except cp.SolverError:
            self.problem.solve(solver="SCS", verbose=False, warm_start=True)

        if self.g.value is None:
            raise RuntimeError(f"DR-DeePC solve failed with status {self.problem.status}")

        sample_predictions = [np.asarray(expr.value).reshape(-1) for expr in self.y_future_exprs]
        return DRDeePCResult(
            g=np.asarray(self.g.value).reshape(-1),
            u_future=np.asarray(self.u_future_expr.value).reshape(-1),
            y_future_samples=sample_predictions,
            objective=float(self.problem.value),
            status=self.problem.status,
        )
