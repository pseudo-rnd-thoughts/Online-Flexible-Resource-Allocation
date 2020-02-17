"""Task pricing agent"""

from __future__ import annotations

import random as rnd
from typing import List, TYPE_CHECKING

import numpy as np
import tensorflow as tf

import core.log as log
from agents.dqn_agent import DqnAgent, Trajectory

if TYPE_CHECKING:
    from core.server import Server
    from core.task import Task


class TaskPricingNetwork(tf.keras.Model):
    """Task Pricing network with a LSTM layer, ReLU layer and Linear layer"""

    def __init__(self, lstm_connections: int = 10, relu_connections: int = 20, num_outputs: int = 25):
        super().__init__()

        self.task_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_connections), input_shape=[None, 9])
        self.relu_layer = tf.keras.layers.Dense(relu_connections, activation='relu')
        self.q_layer = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs, training=None, mask=None):
        task_output = self.task_layer(inputs)
        relu_output = self.relu_layer(task_output)
        return self.q_layer(relu_output)


class TaskPricingAgent(DqnAgent):

    def __init__(self, name: str, num_prices: int = 26,
                 discount_factor: float = 0.9, default_reward: float = -0.1):
        super().__init__(name, TaskPricingNetwork, num_prices)
        self.discount_factor = discount_factor
        self.default_reward = default_reward

    def __str__(self) -> str:
        return f'{self.name} - Num prices: {self.num_outputs}, Discount factor: {self.discount_factor}, ' \
               f'Default reward: {self.default_reward}'

    def price_task(self, new_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                   greedy_policy: bool = True) -> float:
        observation = np.array([[new_task.normalise_task_info(server, time_step) + [1]] + [
            task.normalise_task_info(server, time_step) + [0]
            for task in allocated_tasks
        ]]).astype(np.float32)
        log.neural_network('TPA Obs', observation)

        if server in self.last_server_trajectory:
            self.last_server_trajectory[server].next_state = observation
            self.last_server_trajectory.pop(server)  # Possibly not needed

        if greedy_policy and rnd.random() < self.epsilon:
            action = rnd.randint(0, self.num_outputs)
        else:
            action_q_values = self.network_model.call(observation)
            log.neural_network('TPA Action Q Value', action_q_values)
            action = np.argmax(action_q_values)

        trajectory = Trajectory(observation, action, self.default_reward, None)
        self.last_server_trajectory[server] = trajectory
        self.replay_buffer.append(trajectory)

        return action

    def task_allocated(self, server: Server, price: int):
        self.last_server_trajectory[server].reward = price
