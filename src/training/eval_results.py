"""
Log of the evaluation results and actions
"""

from typing import List

import tensorflow as tf

from env.task_stage import TaskStage


class EvalResults:
    """
    Agent evaluation results
    """

    def __init__(self):
        # Auction attributes
        self.total_winning_prices: float = 0
        self.winning_prices: List[float] = []
        self.num_auctions: int = 0
        self.auction_actions: List[float] = []

        # Resource allocation attributes
        self.num_completed_tasks: int = 0
        self.num_failed_tasks: int = 0
        self.total_prices: float = 0
        self.num_resource_allocations: int = 0
        self.weighting_actions: List[float] = []

        self.env_attempted_tasks: List[int] = [0]
        self.env_completed_tasks: List[int] = [0]
        self.env_failed_tasks: List[int] = [0]

    def auction(self, actions, rewards):
        """
        Auction case

        Args:
            actions: Dictionary of actions
            rewards: Dictionary of rewards
        """
        for server, price in rewards.items():
            self.total_winning_prices += price
            self.winning_prices.append(price)
        for server, action in actions.items():
            self.auction_actions.append(action)
        self.num_auctions += 1
        self.env_attempted_tasks[-1] += 1

    def resource_allocation(self, actions, rewards):
        """
        Resource allocation case

        Args:
            actions: Dictionary of actions
            rewards: Dictionary of rewards
        """
        for server, tasks in rewards.items():
            for task in tasks:
                if task.stage is TaskStage.COMPLETED:
                    self.num_completed_tasks += 1
                    self.env_completed_tasks[-1] += 1
                    self.total_prices += task.price
                elif task.stage is TaskStage.FAILED:
                    self.num_failed_tasks += 1
                    self.env_failed_tasks[-1] += 1
                    self.total_prices -= task.price
                else:
                    raise Exception(f'Unexpected task stage: {task.stage}, {str(task)}')
        for server, task_actions in actions.items():
            for task, action in task_actions.items():
                self.weighting_actions.append(action)
        self.num_resource_allocations += 1

    def finished_env(self):
        self.env_attempted_tasks.append(0)
        self.env_completed_tasks.append(0)
        self.env_failed_tasks.append(0)

    def save(self, episode):
        """
        Save the evaluation results

        Args:
            episode: Episode number
        """
        tf.summary.scalar('Eval total winning prices', self.total_winning_prices, episode)
        tf.summary.scalar('Eval total prices', self.total_prices, episode)
        if episode % 50 == 0:
            tf.summary.histogram('Eval auction actions', self.auction_actions, episode)
            tf.summary.histogram('Eval winning auction prices', self.winning_prices, episode)

        tf.summary.scalar('Eval number of completed tasks', self.num_completed_tasks, episode)
        tf.summary.scalar('Eval number of failed tasks', self.num_failed_tasks, episode)
        percent = (self.num_completed_tasks + self.num_failed_tasks) / self.num_auctions
        tf.summary.scalar('Eval percent all tasks run', percent, episode)
        ratio = self.num_completed_tasks / (self.num_failed_tasks + 1)
        tf.summary.scalar('Eval completed failed task ratio', ratio, episode)
        tf.summary.histogram('Eval weightings', self.weighting_actions, episode)
