"""Resource weighting agent"""

import random as rnd
from typing import Optional, List, Dict

import numpy as np
import tensorflow as tf

from agents.dqn_agent import DqnAgent, Trajectory
from core.server import Server
from core.task import Task


class ResourceWeightingNetwork(tf.keras.Model):
    """Resource weighting network using three layer network - LSTM tasks, ReLU layer, linear layer"""

    def __init__(self, lstm_connections: int = 10, relu_connections: int = 20, num_outputs: int = 25):
        super().__init__()

        self.task_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_connections))
        self.relu_layer = tf.keras.layers.Dense(relu_connections, activation='relu')
        self.q_layer = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs, training=None, mask=None):
        """
        Propagates the forward-call of the network
        :param inputs: The inputs
        :param training: Unused variable
        :param mask: Unused variable
        """
        task_output = self.task_layer(inputs)
        relu_output = self.relu_layer(task_output)
        return self.q_layer(relu_output)


class ResourceWeightingAgent(DqnAgent):
    """Resource weighting agent using a resource weighting network"""

    last_trajectory: Optional[Trajectory] = None

    def __init__(self, agent_num, num_weights: int = 10, discount_other_task_reward: float = 0.2):
        super().__init__('Resource Weighting agent {}'.format(agent_num), ResourceWeightingNetwork, num_weights)
        self.discount_other_task_reward = discount_other_task_reward

    def weight_task(self, task: Task, other_tasks: List[Task], server: Server, time_step: int,
                    greedy_policy: bool = True) -> float:
        task_observation = task.normalise_new_task(server, time_step)
        observation = [
            task_observation + task.normalise_task_progress(server, time_step)
            for task in other_tasks
        ]

        if greedy_policy and rnd.random() < self.epsilon:
            action = rnd.randint(0, self.num_outputs)
        else:
            action_q_values = self.network_model.call(observation)
            action = np.argmax(action_q_values)[0]

        trajectory = Trajectory(observation, action, 0, None)
        self.last_server_trajectory[server] = trajectory

        self.replay_buffer.append(trajectory)

        return action

    def update_next_state(self, task: Task, other_tasks: List[Task], server: Server, time_step: int,
                          task_reward: int, other_task_reward: Dict[Task, int]):
        self.last_server_trajectory[server].reward = task_reward + self.discount_other_task_reward * sum(other_task_reward.values())
        self.last_server_trajectory[server].next_state = self.task_observation(task, other_tasks, server, time_step)

    def task_observation(self, task: Task, other_tasks: List[Task], server: Server, time_step: int):
        task_observation = task.normalise_new_task(server, time_step)
        observation = [
            task_observation + task.normalise_task_progress(server, time_step)
            for task in other_tasks
        ]
        return observation
