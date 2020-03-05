"""Resource weighting agent"""

from __future__ import annotations

import random as rnd

from typing import TYPE_CHECKING
import numpy as np

import core.log as log
from agents.dqn_agent import DqnAgent
from agents.resource_weighting_network import ResourceWeightingNetwork
from agents.trajectory import Trajectory

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from typing import List, Optional
    from env.server import Server
    from env.task import Task


class ResourceWeightingAgent(DqnAgent):
    """Resource weighting agent using a resource weighting network"""

    def __init__(self, name: str, num_weights: int = 10, discount_other_task_reward: float = 0.2,
                 successful_task_reward: float = 1, failed_task_reward: float = -2,
                 task_multiplier: float = 2.0):
        super().__init__(name, ResourceWeightingNetwork, num_weights)

        self.discount_other_task_reward = discount_other_task_reward
        self.successful_task_reward = successful_task_reward
        self.failed_task_reward = failed_task_reward
        self.task_multiplier = task_multiplier

    def weight(self, task: Task, other_tasks: List[Task], server: Server, time_step: int,
               greedy_policy: bool = True) -> float:
        """
        Get the action weight for the task
        :param task: The task to calculate the weight for
        :param other_tasks: The other tasks to consider
        :param server: The server of the tasks
        :param time_step: The current time step
        :param greedy_policy: If to get the policy greedy
        :return: The action weight
        """
        if len(other_tasks) > 1:
            observation = self.network_observation(task, other_tasks, server, time_step)

            if greedy_policy and rnd.random() < self.epsilon:
                action = rnd.randint(1, self.num_outputs)
                log.debug(f'\t{self.name} RWA - {server.name} Server and {task.name} Task has greedy action: {action}')
                assert 0 < action <= self.num_outputs, 'greedy'
            else:
                action_q_values = self.network_model.call(observation)
                assert len(action_q_values[0]) == self.num_outputs, f'{str(action_q_values)} {self.num_outputs} {len(action_q_values)}'
                action = np.argmax(action_q_values) + 1
                log.debug(f'\t{self.name} TPA - {server.name} Server and {task.name} Task has argmax action: {action}')
                assert 0 < action <= self.num_outputs, 'argmax'

            return action
        else:
            return 1.0

    @staticmethod
    def network_observation(task: Task, other_tasks: List[Task], server: Server, time_step: int):
        assert any(_task != task for _task in other_tasks)

        task_observation = task.normalise(server, time_step)
        observation = np.array([[
            task_observation + task.normalise(server, time_step)
            for task in other_tasks if other_tasks != task
        ]]).astype(np.float32)

        return observation

    def add_incomplete_task_observation(self, observation: np.Array, action: float,
                                        next_observation: Optional[np.Array], rewards: List[Task]):
        assert all(len(ob) == self.network_model.input_width for ob in observation[0]), observation
        assert 0 < action <= self.num_outputs, action
        assert next_observation is None or all(len(ob) == self.network_model.input_width for ob in next_observation[0]), next_observation
        assert all(reward_task.stage is TaskStage.COMPLETED or reward_task.stage is TaskStage.FAILED
                   for reward_task in rewards)

        reward = sum(self.successful_task_reward if reward_task.stage is TaskStage.COMPLETED else self.failed_task_reward
                     for reward_task in rewards)
        self.replay_buffer.append(Trajectory(observation, action+1, reward, next_observation))

    def add_finished_task(self, observation: np.Array, action: float, finished_task: Task, rewards: List[Task]):
        assert all(len(ob) == self.network_model.input_width for ob in observation[0]), observation
        assert 0 < action <= self.num_outputs, action
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED, finished_task
        assert all(reward_task.stage is TaskStage.COMPLETED or reward_task.stage is TaskStage.FAILED
                   for reward_task in rewards), rewards

        reward = self.successful_task_reward * self.task_multiplier if finished_task.stage is TaskStage.COMPLETED else \
            self.failed_task_reward * self.task_multiplier
        for reward_task in rewards:
            if reward_task.name != finished_task.name:
                reward += self.successful_task_reward if reward_task.stage is TaskStage.COMPLETED else self.failed_task_reward

        self.replay_buffer.append(Trajectory(observation, action+1, reward, None))
