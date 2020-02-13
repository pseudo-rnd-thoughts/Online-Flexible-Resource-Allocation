"""Task pricing agent"""

import random as rnd
from typing import List, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from agents.dqn_agent import DqnAgent, Trajectory

if TYPE_CHECKING:
    from core.server import Server
    from core.task import Task


class TaskPricingNetwork(tf.keras.Model):
    """Task Pricing network with a LSTM layer, ReLU layer and Linear layer"""

    def __init__(self, lstm_connections: int = 10, relu_connections: int = 20, num_outputs: int = 25):
        super().__init__()

        self.task_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_connections))
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
        return f'{self.name} - Num prices: {self.num_outputs}, discount factor: {self.discount_factor}, ' \
               f'default reward: {self.default_reward}'

    def price_task(self, new_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                   greedy_policy: bool = True) -> float:
        new_task_observation = new_task.normalise_new_task(server, time_step)
        observation = [
            new_task_observation + task.normalise_task_progress(server, time_step)
            for task in allocated_tasks
        ]

        if server in self.last_server_trajectory:
            self.last_server_trajectory[server].next_state = observation
            self.last_server_trajectory.pop(server)  # Possibly not needed

        if greedy_policy and rnd.random() < self.epsilon:
            action = rnd.randint(0, self.num_outputs)
        else:
            action_q_values = self.network_model.call(observation)
            action = np.argmax(action_q_values)[0]

        trajectory = Trajectory(observation, action, self.default_reward, None)
        self.last_server_trajectory[server] = trajectory
        self.replay_buffer.append(trajectory)

        return action

    def task_allocated(self, server, price):
        self.last_server_trajectory[server].reward = price
        return self.last_server_trajectory[server]
