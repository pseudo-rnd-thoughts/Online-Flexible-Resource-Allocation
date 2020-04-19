"""
Reinforcement learning policies to allows for epsilon greedy policies
"""

from __future__ import annotations

import random as rnd
from abc import ABC
from typing import List, Dict

import tensorflow as tf

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


class EpsilonGreedyPolicy(ABC):
    """
    Epsilon Greedy policy; actions are taken randomly epsilon time otherwise takes greedy actions
    """

    def __init__(self, initial_epsilon: float = 1, final_epsilon: float = 0.1, epsilon_steps: int = 10000,
                 update_frequency: int = 25):
        # Exploration attributes: initial, final and total steps
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.diff_epsilon = final_epsilon - initial_epsilon
        self.epsilon_steps = epsilon_steps

        # Exploration factor
        self.epsilon = initial_epsilon

        # Number of actions and exploration update frequency
        self.total_actions = 0
        self.update_frequency = update_frequency

    def update_epsilon(self):
        """
        Update the epsilons
        """
        self.total_actions += 1
        if self.total_actions % self.update_frequency == 0:
            self.epsilon = max(self.total_actions / self.epsilon_steps * self.diff_epsilon + self.initial_epsilon, 0)


class EpsilonGreedyTaskPricingPolicy(EpsilonGreedyPolicy, TaskPricingAgent):
    """
    Allows for epsilon greedy exploration of the environment for the task pricing agent
    """

    def __init__(self, agent: TaskPricingDqnAgent, **kwargs):
        TaskPricingAgent.__init__(self, f'Greedy {agent.name}', agent.limit_parallel_tasks)
        EpsilonGreedyPolicy.__init__(self, **kwargs)

        self.agent = agent

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        self.update_epsilon()
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
        self.update_epsilon()
        # Update such that all actions are not random
        actions = {}
        for task in allocated_tasks:
            if rnd.random() < self.epsilon:
                actions[task] = float(rnd.randint(0, self.agent.num_actions-1))
            else:
                observation = tf.expand_dims(self.agent.network_obs(task, allocated_tasks, server, time_step), axis=0)
                action = tf.math.argmax(self.agent.model_network(observation), axis=1, output_type=tf.int32)
                actions[task] = float(action)
        return actions
