import random as rnd
from abc import ABC, abstractmethod

import tensorflow as tf

from agents.dqn import DqnAgent


class Policy(ABC):

    @abstractmethod
    def action(self, state):
        pass


class EpsilonGreedyPolicy(Policy):

    def __init__(self, agent: DqnAgent, epsilon: float = 0.1):
        self.agent = agent
        self.epsilon = epsilon

    def action(self, state):
        if rnd.random() < self.epsilon:
            return rnd.randint(0, self.agent.num_actions)
        else:
            return tf.math.argmax(self.agent.model_network(state))


class GreedyPolicy(Policy):

    def __init__(self, agent: DqnAgent):
        self.agent = agent

    def action(self, state):
        return tf.math.argmax(self.agent.model_network(state))
