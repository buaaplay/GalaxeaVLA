from __future__ import annotations

from typing import Any, Mapping

from .schemas import Last0Observation


def build_last0_observation(data: Mapping[str, Any]) -> Last0Observation:
    return Last0Observation.from_dict(dict(data))
