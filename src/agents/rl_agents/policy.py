"""
Policies for the DQN and PG agents
"""

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def action(self, *args):
        pass


class RandomPolicy(Policy):

    def __init__(self):
        Policy.__init__(self, 'Random Policy')

    def action(self, *args):
        return np.random.uniform(0, 10)


class DqnGreedyPolicy(Policy):

    def __init__(self, dqn_agent):
        Policy.__init__(self, 'Dqn Greedy Policy')
        self.dqn_agent = dqn_agent

    def action(self, *args):
        return self.dqn_agent.get_action(*args)


class DqnEpsilonGreedyPolicy(Policy):

    def __init__(self, dqn_agent):
        Policy.__init__(self, 'Dqn Epsilon Greedy Policy')
        self.dqn_agent = dqn_agent

        self.epsilon = 1

    def action(self, *args):
        if np.random.uniform() < self.epsilon:
            size = len(args[0]) if len(args) == 3 else len(args[1])
            return np.random.uniform(0, 10, size)
        else:
            return self.dqn_agent.get_action(*args)
