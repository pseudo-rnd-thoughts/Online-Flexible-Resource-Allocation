"""
Dueling DQN agent based on Dueling Network Architectures for Deep Reinforcement Learning (https://arxiv.org/abs/1511.06581)
"""

from abc import ABC

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network


class DuelingDQN(DqnAgent, ABC):
    """
    Implementations of a dueling DQN agent
    """

    def __init__(self, network: Network, target_update_frequency: int = 2500, initial_exploration: float = 1,
                 final_exploration: float = 0.1, final_exploration_frame: int = 20000, **kwargs):
        DqnAgent.__init__(self, network, target_update_frequency, initial_exploration, final_exploration,
                          final_exploration_frame, **kwargs)


class TaskPricingDuelingDqnAgent(DuelingDQN, TaskPricingDqnAgent):
    """
    Task pricing dueling DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'Dueling DQN TP {agent_num}'


class ResourceWeightingDuelingDqnAgent(DuelingDQN, ResourceWeightingDqnAgent):
    """
    Resource Weighting Dueling DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DuelingDQN.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'Dueling DQN RW {agent_num}'
