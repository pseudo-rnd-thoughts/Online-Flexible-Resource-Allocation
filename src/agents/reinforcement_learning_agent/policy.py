import random as rnd
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from agents.reinforcement_learning_agent.dqn import DqnAgent


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
            return rnd.randint(0, self.agent.num_actions - 1)
        else:
            return np.argmax(self.agent.model_network(tf.cast([state], tf.float32)))


class GreedyPolicy(Policy):

    def __init__(self, agent: DqnAgent):
        self.agent = agent

    def action(self, state):
        return np.argmax(self.agent.model_network(tf.cast([state], tf.float32)))


class EpsilonGreedyPolicyCategorial(Policy):
    pass


class GreedyPolicyCategorial(Policy):
    pass
