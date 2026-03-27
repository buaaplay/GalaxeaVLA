from .action_adapter import Last0ActionAdapter
from .config import Last0InferenceConfig
from .schemas import Last0InferenceResult, Last0Observation, Last0RobotState, PoseCommand

__all__ = [
    "Last0ActionAdapter",
    "Last0InferenceConfig",
    "Last0InferenceResult",
    "Last0InferenceRuntime",
    "Last0Observation",
    "Last0RobotState",
    "PoseCommand",
]


def __getattr__(name: str):
    if name == "Last0InferenceRuntime":
        from .runtime import Last0InferenceRuntime

        return Last0InferenceRuntime
    raise AttributeError(name)
