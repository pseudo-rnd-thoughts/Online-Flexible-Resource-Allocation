from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import random as rnd

import tensorflow as tf


class RLAgent(ABC):

    def __init__(self, name: str, replay_buffer_length: int = 100000, batch_size: int = 32,
                 optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 loss_func: tf.keras.losses.Loss = tf.keras.losses.Huber(), discount_factor: float = 0.9,
                 target_update_freq: float = 0.01):
        self.name = name

        self.replay_buffer = deque(maxlen=replay_buffer_length)
        self.replay_buffer_length = replay_buffer_length

        self.optimiser = optimiser
        self.loss_func = loss_func

        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.discount_factor = discount_factor

        self.target_update_freq = target_update_freq

        self.total_steps = 0
        self.training_steps = 0

    def observation(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        self.training_steps += 1

    def train(self):
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.replay_buffer, self.batch_size))
        states = tf.cast(np.stack(states), tf.float32)
        actions = tf.cast(np.stack(actions), tf.int32)
        next_states = tf.cast(np.stack(next_states), tf.float32)
        rewards = tf.cast(np.stack(rewards), tf.float32)
        dones = tf.cast(np.stack(np.logical_not(dones)), tf.float32)

        loss = self._train(states, actions, next_states, rewards, dones)
        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self._update_target()

        return loss

    @abstractmethod
    def _train(self, states, actions, next_states, rewards, dones):
        pass

    @abstractmethod
    def _update_target(self):
        pass
