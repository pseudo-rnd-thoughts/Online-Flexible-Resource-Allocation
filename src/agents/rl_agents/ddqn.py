"""
Double DQN agent implemented based on Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)
"""

from __future__ import annotations

from abc import ABC
from typing import Callable

import gin.tf
import tensorflow as tf

from agents.rl_agents.dqn import DqnAgent, ResourceWeightingDQN, TaskPricingDQN


@gin.configurable
class DdqnAgent(DqnAgent, ABC):
    """
    Implementation of a double deep q network agent
    """

    def __init__(self, network_input_width: int, network_num_outputs: int,
                 build_network: Callable[[int], tf.keras.Sequential],
                 **kwargs):
        DqnAgent.__init__(network_input_width, network_num_outputs, build_network, **kwargs)

    def _train(self) -> float:
        # Todo
        pass


@gin.configurable
class TaskPricingDDQN(DdqnAgent, TaskPricingDQN):
    """
    Todo
    """
    def __init__(self, network_input_width: int, network_num_outputs: int,
                 build_network: Callable[[int], tf.keras.Sequential], **kwargs):
        super().__init__(network_input_width, network_num_outputs, build_network, **kwargs)


@gin.configurable
class ResourceWeightingDDQN(DdqnAgent, ResourceWeightingDQN):
    """
    Todo
    """
    def __init__(self, network_input_width: int, network_num_outputs: int,
                 build_network: Callable[[int], tf.keras.Sequential], **kwargs):
        DdqnAgent.__init__(network_input_width, network_num_outputs, build_network, **kwargs)
        ResourceWeightingDQN.__init__(self)
