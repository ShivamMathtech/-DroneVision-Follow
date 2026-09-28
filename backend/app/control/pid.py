from __future__ import annotations

import time


class PID:
    def __init__(self, kp: float = 0.25, ki: float = 0.0, kd: float = 0.03,
                 output_limit: float = 0.25, integral_limit: float = 1.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit, self.integral_limit = abs(output_limit), abs(integral_limit)
        self.integral = 0.0
        self.previous_error: float | None = None
        self.previous_time: float | None = None

    def update(self, error: float, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        dt = max(0.001, now - self.previous_time) if self.previous_time is not None else 0.02
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        derivative = (error - self.previous_error) / dt if self.previous_error is not None else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error, self.previous_time = error, now
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None

