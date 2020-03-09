"""Trajectory"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    import numpy as np


class Trajectory(NamedTuple):
    observation: np.Array
    action: float
    reward: float
    next_observation: Optional[np.Array]
