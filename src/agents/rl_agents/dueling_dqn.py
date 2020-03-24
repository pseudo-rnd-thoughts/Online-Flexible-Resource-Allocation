"""
Dueling DQN agent based on Dueling Network Architectures for Deep Reinforcement Learning
 (https://arxiv.org/abs/1511.06581)
"""

from __future__ import annotations

from abc import ABC

import gin.tf

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network


@gin.configurable
class DuelingDQN(DqnAgent, ABC):
    """
    Implementations of a dueling DQN agent
    """

    def __init__(self, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)


@gin.configurable
class TaskPricingDuelingDqnAgent(DuelingDQN, TaskPricingDqnAgent):
    """
    Task pricing dueling DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, f'Dueling DQN TP {agent_num}', network, **kwargs)


@gin.configurable
class ResourceWeightingDuelingDqnAgent(DuelingDQN, ResourceWeightingDqnAgent):
    """
    Resource Weighting Dueling DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, f'Dueling DQN RW {agent_num}', network, **kwargs)
