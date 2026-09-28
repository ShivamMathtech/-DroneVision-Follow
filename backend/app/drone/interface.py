from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict

from app.control.safety import DesiredCommand, SafetyLayer


class DroneInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def telemetry(self) -> dict: ...

    @abstractmethod
    def send_desired_command(self, command: DesiredCommand) -> dict: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...


class MockDrone(DroneInterface):
    """Simulation telemetry sink. It has no actuator or radio transport."""

    def __init__(self, safety: SafetyLayer) -> None:
        self.safety = safety
        self.connected = True
        self.last_command = DesiredCommand()

    def connect(self) -> None:
        self.connected = True
        self.safety.reset_estop()

    def disconnect(self) -> None:
        self.connected = False
        self.last_command = DesiredCommand()

    def telemetry(self) -> dict:
        return {
            "mode": "SIMULATION", "connection_status": "CONNECTED" if self.connected else "DISCONNECTED",
            "armed": False, "altitude": 0.0, "speed": 0.0, "heading": 0.0,
            "battery": None, "gps_latitude": None, "gps_longitude": None,
            "command": asdict(self.last_command),
        }

    def send_desired_command(self, command: DesiredCommand) -> dict:
        if not self.connected:
            return {"accepted": False, "reason": "simulation interface disconnected"}
        self.last_command = self.safety.apply(command)
        return {"accepted": True, "command": asdict(self.last_command), "mode": "SIMULATION"}

    def emergency_stop(self) -> None:
        self.safety.emergency_stop()
        self.last_command = DesiredCommand()

