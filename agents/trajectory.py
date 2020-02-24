"""Trajectory"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class Trajectory(NamedTuple):
    observation: np.Array
    action: float
    reward: float
    next_observation: Optional[np.Array]
