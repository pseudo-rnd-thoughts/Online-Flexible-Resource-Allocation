"""
Double DQN agent implemented based on Deep Reinforcement Learning with Double Q-learning
 (https://arxiv.org/abs/1509.06461)
"""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import gin.tf
import numpy as np

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network
from agents.rl_agents.rl_agent import Trajectory


@gin.configurable
class DdqnAgent(DqnAgent, ABC):
    """
    Implementation of a double deep q network agent
    """

    def __init__(self, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

    def _loss(self, trajectory: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate the double bellman update for the action
        obs = self.network_obs(trajectory.state.task, trajectory.state.tasks,
                               trajectory.state.server, trajectory.state.time_step)
        target = np.array(self.model_network(obs))
        action = int(trajectory.action)

        if trajectory.next_state is None:
            target[0][action] = trajectory.reward
        else:
            next_obs = self.network_obs(trajectory.next_state.task, trajectory.next_state.tasks,
                                        trajectory.next_state.server, trajectory.next_state.time_step)
            target[0][action] = trajectory.reward + \
                self.discount_factor * self.target_network(next_obs)[0][np.argmax(self.model_network(next_obs))]

        return target, self.model_network(obs)


@gin.configurable
class TaskPricingDdqnAgent(DdqnAgent, TaskPricingDqnAgent):
    """
    Task pricing double dqn agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, f'DDQN TP {agent_num}', network, **kwargs)


@gin.configurable
class ResourceWeightingDdqnAgent(DdqnAgent, ResourceWeightingDqnAgent):
    """
    Resource weighting double dqn agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, f'DDQN RW {agent_num}', network, **kwargs)
