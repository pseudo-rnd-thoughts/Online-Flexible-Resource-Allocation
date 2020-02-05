from collections import namedtuple
from typing import List

Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: List[Transition] = []
        self.position: int = 0

    def push(self, observation, action, reward, next_observation):
        self.memory.append(Transition(observation, action, reward, next_observation))

    def update_observation(self, task, price):
        self.memory[-1].reward = price
