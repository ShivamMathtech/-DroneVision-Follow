from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DesiredCommand:
    yaw_rate: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    throttle: float = 0.0


class SafetyLayer:
    """Clamps simulation outputs and latches emergency stop; never produces motor signals."""

    def __init__(self, max_axis: float = 0.25) -> None:
        self.max_axis = abs(max_axis)
        self.emergency_stopped = False
        self.geofence_ok = True
        self.manual_override = False
        self.target_valid = False

    def apply(self, command: DesiredCommand) -> DesiredCommand:
        if self.emergency_stopped or self.manual_override or not self.geofence_ok or not self.target_valid:
            return DesiredCommand()
        return DesiredCommand(*(
            max(-self.max_axis, min(self.max_axis, value))
            for value in (command.yaw_rate, command.pitch, command.roll, command.throttle)
        ))

    def emergency_stop(self) -> None:
        self.emergency_stopped = True

    def reset_estop(self) -> None:
        self.emergency_stopped = False

    def status(self) -> dict:
        return {
            "mode": "SIMULATION", "physical_control_enabled": False,
            "emergency_stopped": self.emergency_stopped, "geofence_ok": self.geofence_ok,
            "manual_override": self.manual_override, "target_valid": self.target_valid,
            "max_axis": self.max_axis,
        }

