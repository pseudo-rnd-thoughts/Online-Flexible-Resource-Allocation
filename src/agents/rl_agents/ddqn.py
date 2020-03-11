"""
Double DQN agent implemented based on Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)
"""

from __future__ import annotations

from abc import ABC

import gin.tf

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DdqnAgent(DqnAgent, ABC):
    """
    Implementation of a double deep q network agent
    """

    def __init__(self, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

    def _train(self) -> float:
        # Todo
        pass


@gin.configurable
class TaskPricingDdqnAgent(DdqnAgent, TaskPricingDqnAgent):
    """
    Task pricing double dqn agent
    """
    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'DDQN TP {agent_num}'


@gin.configurable
class ResourceWeightingDdqnAgent(DdqnAgent, ResourceWeightingDqnAgent):
    """
    Resource weighting double dqn agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'DDQN RW {agent_num}'
