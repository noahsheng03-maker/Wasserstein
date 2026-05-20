from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveRadiusConfig:
    epsilon_init: float
    epsilon_min: float
    epsilon_max: float
    eta_up: float
    eta_down: float
    error_ref: float
    stress_ref: float
    shift_ref: float
    weight_error: float
    weight_stress: float
    weight_shift: float


class AdaptiveRadiusScheduler:
    def __init__(self, config: AdaptiveRadiusConfig) -> None:
        self.config = config
        self.epsilon = config.epsilon_init

    def update(self, error_metric: float, stress_metric: float, shift_metric: float) -> float:
        delta = (
            self.config.weight_error * (error_metric - self.config.error_ref)
            + self.config.weight_stress * (stress_metric - self.config.stress_ref)
            + self.config.weight_shift * (shift_metric - self.config.shift_ref)
        )
        rate = self.config.eta_up if delta > 0.0 else self.config.eta_down
        self.epsilon += rate * delta
        self.epsilon = min(max(self.epsilon, self.config.epsilon_min), self.config.epsilon_max)
        return self.epsilon
