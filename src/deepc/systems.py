from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearHoverSystem:
    """A 3D hover-like linear system for DeePC experiments.

    This is not the original quadcopter simulator from the paper. It is a
    structured linear benchmark with the same input/output dimensions and
    horizon scales so that the experiment code can be developed and debugged
    against the original protocol before plugging in a higher-fidelity model.
    """

    dt: float = 0.04
    process_noise_std: float = 0.005
    measurement_noise_std: float = 0.01
    hover_thrust: float = 0.27

    def __post_init__(self) -> None:
        dt = self.dt
        self.a = np.block(
            [
                [np.eye(3), dt * np.eye(3)],
                [np.zeros((3, 3)), 0.94 * np.eye(3)],
            ]
        )
        self.b = np.block(
            [
                [0.5 * dt**2 * np.eye(3)],
                [dt * np.eye(3)],
            ]
        )
        self.c = np.block([np.eye(3), np.zeros((3, 3))])
        self.state_dim = self.a.shape[0]
        self.input_dim = self.b.shape[1]
        self.output_dim = self.c.shape[0]

    def rollout(
        self,
        u_sequence: np.ndarray,
        x0: np.ndarray | None = None,
        process_noise_std: float | None = None,
        measurement_noise_std: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        process_noise_std = self.process_noise_std if process_noise_std is None else process_noise_std
        measurement_noise_std = self.measurement_noise_std if measurement_noise_std is None else measurement_noise_std
        rng = np.random.default_rng() if rng is None else rng
        horizon = u_sequence.shape[1]
        x = np.zeros(self.state_dim) if x0 is None else x0.copy()
        xs = np.zeros((self.state_dim, horizon + 1))
        ys = np.zeros((self.output_dim, horizon))
        xs[:, 0] = x
        for k in range(horizon):
            w = process_noise_std * rng.standard_normal(self.state_dim)
            v = measurement_noise_std * rng.standard_normal(self.output_dim)
            u_eff = u_sequence[:, k].copy()
            u_eff[0] -= self.hover_thrust
            x = self.a @ x + self.b @ u_eff + w
            y = self.c @ x + v
            xs[:, k + 1] = x
            ys[:, k] = y
        return xs, ys
