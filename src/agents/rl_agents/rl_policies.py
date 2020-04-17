"""
Reinforcement learning policies to allows for epsilon greedy policies
"""

from __future__ import annotations

from abc import ABC
from typing import List, Dict

import random as rnd

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


class EpsilonGreedyPolicy(ABC):
    """
    Epsilon Greedy policy; actions are taken randomly epsilon time otherwise takes greedy actions
    """

    def __init__(self, initial_epsilon: float = 1, final_epsilon: float = 0.1, exploration_steps: int = 10000):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_steps = exploration_steps

        self.epsilon = initial_epsilon


class EpsilonGreedyTaskPricingPolicy(EpsilonGreedyPolicy, TaskPricingAgent):
    """
    Allows for epsilon greedy exploration of the environment for the task pricing agent
    """

    def __init__(self, agent: TaskPricingDqnAgent, **kwargs):
        TaskPricingAgent.__init__(self, f'Greedy {agent.name}', agent.limit_number_task_parallel)
        EpsilonGreedyPolicy.__init__(self, **kwargs)

        self.agent = agent

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        if rnd.random() < self.epsilon:
            return float(rnd.randint(0, self.agent.num_actions-1))
        else:
            # noinspection PyProtectedMember
            return self.agent._get_action(auction_task, allocated_tasks, server, time_step)


class EpsilonGreedyResourceAllocationPolicy(EpsilonGreedyPolicy, ResourceWeightingAgent):
    """
    Allows for epsilon greedy exploration of the environment for the resource weighting agent
    """

    def __init__(self, agent: ResourceWeightingDqnAgent, **kwargs):
        ResourceWeightingAgent.__init__(self, f'Greedy {agent.name}')
        EpsilonGreedyPolicy.__init__(self, **kwargs)

        self.agent = agent

    def _get_actions(self, allocated_tasks: List[Task], server: Server, time_step: int) -> Dict[Task, float]:
        # Update such that all actions are not random
        if rnd.random() < self.epsilon:
            return {task: float(rnd.randint(0, self.agent.num_actions-1)) for task in allocated_tasks}
        else:
            # noinspection PyProtectedMember
            return self.agent._get_actions(allocated_tasks, server, time_step)
