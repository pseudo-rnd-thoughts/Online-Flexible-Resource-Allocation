"""
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
"""

import operator
from typing import List
import random as rnd
from math import inf

from setting import load_environment_setting


class OnlineFlexibleResourceAllocationEnv:
    # Task format: [name, required storage, required comp, required results data, auction time, deadline,
    #               storage progress, comp progress, results progress, price]
    # Server format: [name, storage cap, comp cap, bandwidth cap]
    
    # State: Tuple[Dict[Server, List[Task]], Optional[Task]]
    
    def __init__(self, environment_settings: List[str]):
        self.environment_settings = environment_settings
        self.unallocated_tasks = []  # List[Task]
        self.state = {}  # Dict[Server, List[Task]]
        self.total_time_step = 0  # Int
        self.time_step = 0  # Int
        self.auction_task = None  # Optional[Task]
    
    @staticmethod
    def make(settings: List[str]):
        return OnlineFlexibleResourceAllocationEnv(settings)
    
    def reset(self):
        # regenerate the environment based on one of the random environment settings saved
        assert len(self.environment_settings) > 0
        
        environment_setting = rnd.choice(self.environment_settings)
        new_servers, new_tasks, new_total_time_steps = load_environment_setting(environment_setting)
        
        self.unallocated_tasks = sorted(new_tasks, key=operator.itemgetter(4))
        self.state = {server: [] for server in new_servers}
        self.total_time_step = new_total_time_steps
        
        self.auction_task = self.unallocated_tasks.pop() if self.unallocated_tasks[0][4] == self.time_step else None
        return self.state, self.auction_task
    
    def step(self, actions):  # Dict[Server, Union[float, Dict[Task, float]]] -> Dict[Server, List[Task]],
        #                                                             Dict[Server, reward], bool, Dict[str, str]
        if self.auction_task is not None:
            # Auction (Action = Dict[Server, float])
            self._assert_auction_actions(actions)
            min_price, min_server, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price is not None and price < min_price:
                    min_price, min_server, second_min_price = price, server, second_min_price
    
            rewards = {
                server: second_min_price if server is min_server else 0
                for server in self.state.keys()
            }
            self.state[min_server].append(self.auction_task)
        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            
            rewards = {
                server: {
                    task: task[9] if task[10] == 4 else -task[9]  # Else task[10] = 5
                    for task in tasks if task[10] < 4
                }
                for server, tasks in self.state
            }
            self.state = {server: [task for task in tasks if task[9] < 4] for server, tasks in self.state}
        
        self.auction_task = self.unallocated_tasks.pop() if self.unallocated_tasks[0][4] == self.time_step else None
        return (self.state, self.auction_task), rewards, self.time_step == self.total_time_step, {}  # Add if a task was completed

    def _assert_auction_actions(self, actions):
        pass

def run():
    task_pricing_agents = {}
    