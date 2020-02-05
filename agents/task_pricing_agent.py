from random import randint, random
from typing import List

from tensorflow import keras
from numpy import argmax as np_argmax

from agents.replay_buffer import ReplayBuffer
from core.server import Server
from core.task import Task


class TaskPricingAgent:

    num_outputs = 26
    epsilon = 0.5
    replay_buffer = ReplayBuffer(1024)

    def __init__(self):
        self.network: keras.Sequential = keras.Sequential()
        self.network.add(keras.layers.Bidirectional(keras.layers.LSTM(40)))
        self.network.add(keras.layers.Dense(30, activation='relu'))
        self.network.add(keras.layers.Dense(self.num_outputs, activation='linear'))

    def price_task(self, new_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        new_task_observation = new_task.normalise_new_task(server, time_step)
        observation = [
            new_task_observation + task.normalise_task_progress(server, time_step)
            for task in allocated_tasks
        ]

        if random() < self.epsilon:
            random_action = randint(0, self.num_outputs)
            self.replay_buffer.push(observation, random_action, 0, observation)
        else:
            action_q_values = self.network.call(observation)
            max_q_action = np_argmax(action_q_values)[0]

            self.replay_buffer.push(observation, max_q_action, 0, observation)
            return -1 if max_q_action == 0 else max_q_action

    def task_allocated(self, task, price):
        self.replay_buffer.update_observation(task, price)