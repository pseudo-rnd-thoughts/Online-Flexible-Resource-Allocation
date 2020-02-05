from typing import List

from pygments.formatters import other

from agents.replay_buffer import ReplayBuffer

from tensorflow import keras

from core.server import Server
from core.task import Task


class ResourceWeightingAgent:

    num_outputs = 25
    epsilon = 0.5
    replay_buffer = ReplayBuffer(1024)

    def __init__(self):
        self.network: keras.Sequential = keras.Sequential()
        self.network.add(keras.layers.Bidirectional(keras.layers.LSTM(40)))
        self.network.add(keras.layers.Dense(30, activation='relu'))
        self.network.add(keras.layers.Dense(self.num_outputs, activation='linear'))

    def weight_resource(self, task: Task, other_tasks: List[Task], server: Server, time_step: int) -> float:
        task_observation = task.normalise_task_progress(server, time_step)
        observation = [
            task_observation + other_task.normalise_task_progress(server, time_step)
            for other_task in other_tasks
        ]