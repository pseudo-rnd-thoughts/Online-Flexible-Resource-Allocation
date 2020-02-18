"""
Environment for online flexible resource allocation
"""

from __future__ import annotations

from collections import namedtuple
from math import inf
import random as rnd
import operator
from typing import List, TYPE_CHECKING, Optional, Dict, Union, Tuple

import core.log as log
import setting

Server = namedtuple('Server', ('name', 'storage_cap', 'comp_cap', 'bandwidth_cap'))
Task = namedtuple('Task', ('name', 'required_storage', 'required_comp', 'required_results_data', 'auction_time',
                           'deadline', 'storage_progress', 'comp_progress', 'results_progress', 'price'))


class OnlineFlexibleResourceAllocationEnv:
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-agent mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """
    
    def __init__(self, environment_settings: List[str]):
        # The available environment settings usable
        self.environment_settings: List[str] = environment_settings
        
        # The current env setting chosen
        self.current_env_setting: str = ''
        
        # Hidden unallocated tasks and the public state for the allocation of server to list of tasks
        self.unallocated_tasks: List[Task] = []
        self.state: Dict[Server, List[Task]] = {}
        
        # The current time step and the total number of time steps
        self.time_step: int = 0
        self.total_time_steps: int = 0
        
        # The current task to be auctioned
        self.auction_task: Optional[Task] = None
    
    @staticmethod
    def make(settings: List[str]):
        """
        Creates the environment using the provided settings
        :param settings: The available settings
        :return: A new OnlineFlexibleResourceAllocation environment
        """
        return OnlineFlexibleResourceAllocationEnv(settings)
    
    def reset(self):
        # regenerate the environment based on one of the random environment settings saved
        assert len(self.environment_settings) > 0
        
        # Select the env setting and load the environment settings
        env_setting: str = rnd.choice(self.environment_settings)
        new_servers, new_tasks, new_total_time_steps = setting.load_environment_setting(env_setting)
        
        # Update the environment variables
        self.current_env_setting = env_setting
        self.unallocated_tasks = sorted(new_tasks, key=lambda task: task.auction_time)
        self.state = {server: [] for server in new_servers}
        self.total_time_steps = new_total_time_steps
        
        self.auction_task = self.unallocated_tasks.pop() if self.unallocated_tasks[0].auction_time == self.time_step else None
        return self.state, self.auction_task
        
    def copy_state(self, state):
        return {}  # TODO
    
    def step(self, actions: Dict[Server, Union[float, Dict[Task, float]]]) -> Tuple[Tuple[Dict[Server, List[Task]], Task], Dict[Server, float], bool, Dict[str, str]]:
        info: Dict[str, str] = {}
        rewards: Dict[Server, float] = {}
        next_state: Dict[Server, List[Task]] = self.copy_state(self.state)
        
        # If there is an auction task then the actions must be auction
        if self.auction_task is not None:
            # Auction (Action = Dict[Server, float])
            self._assert_auction_actions(actions)
            info['step type'] = 'auction'
            
            min_price, min_servers, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price is not None:
                    if price < min_price:
                        min_price, min_servers, second_min_price = price, [server], second_min_price
                    elif price == min_price:
                        min_servers.append(server)
            
            if min_servers:
                winning_server = rnd.choice(min_servers)
                rewards = {winning_server: second_min_price}
                next_state[winning_server].append(self.auction_task)
        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            self._assert_resource_allocation_actions(actions)
            info['step type'] = 'resource allocation'
            rewards = {
                server: {
                    task: task[9] if task[10] == 4 else -task[9]  # Else task[10] = 5
                    for task in tasks if task[10] < 4
                }
                for server, tasks in next_state
            }
            next_state = {server: [task for task in tasks if task[9] < 4] for server, tasks in next_state}
        
        self.auction_task = self.unallocated_tasks.pop() if self.unallocated_tasks[0][4] == self.time_step else None
        return (next_state, self.auction_task), rewards, self.time_step == self.total_time_steps, info
    
    def allocation_resources(self, server, task_weightings):
        if len(self.tasks) == 0:
            return
        elif len(self.tasks) == 1:
            task = self.tasks[0]
            if task.stage is TaskStage.LOADING:
                task.allocate_loading_resources(
                    min(self.bandwidth_capacity, task.required_storage - task.loading_progress, self.storage_capacity),
                    time_step)
            elif task.stage is TaskStage.COMPUTING:
                task.allocate_compute_resources(
                    min(self.computational_capacity, task.required_computation - task.compute_progress), time_step)
            elif task.stage is TaskStage.SENDING:
                task.allocate_sending_resources(
                    min(self.bandwidth_capacity, task.required_results_data - task.sending_results_progress), time_step)
            else:
                raise Exception(f'Unexpected task stage: {task.stage}')
            
            return
        
        loading_weights: Dict[Task, float] = {}
        compute_weights: Dict[Task, float] = {}
        sending_weights: Dict[Task, float] = {}
        
        # Stage 1: Finding the weighting for each of the tasks
        for task in self.tasks:
            weighting = self.resource_weighting_agent.weight_task(
                task, [_task for _task in self.tasks if task is not _task], self, time_step, greedy)
            log.debug(f'\t\tTask {task.name} {task.stage}: {weighting}')
            
            if task.stage is TaskStage.LOADING:
                loading_weights[task] = weighting
            elif task.stage is TaskStage.COMPUTING:
                compute_weights[task] = weighting
            elif task.stage is TaskStage.SENDING:
                sending_weights[task] = weighting
        
        available_storage: float = self.storage_capacity
        available_computation: float = self.computational_capacity
        available_bandwidth: float = self.bandwidth_capacity
        
        # Stage 2: Allocate the compute resources to tasks
        completed_compute_stage: bool = True
        while completed_compute_stage and compute_weights:
            compute_unit: float = available_computation / sum(compute_weights.values())
            completed_compute_stage = False
            
            for task, weight in compute_weights.items():
                if task.required_computation - task.compute_progress <= weight * compute_unit:
                    compute_resources: float = task.required_computation - task.compute_progress
                    
                    task.allocate_compute_resources(compute_resources, time_step)
                    available_computation -= compute_resources
                    available_storage -= task.loading_progress
                    
                    completed_compute_stage = True
                    compute_weights.pop(task)
        
        if compute_weights:
            compute_unit = available_computation / sum(compute_weights.values())
            for task, weight in compute_weights.items():
                task.allocate_compute_resources(compute_unit * weight, time_step)
        
        # Stage 3: Allocate the bandwidth resources to task
        completed_bandwidth_stage: bool = True
        while completed_bandwidth_stage and (loading_weights or sending_weights):
            bandwidth_unit: float = available_bandwidth / (
                    sum(loading_weights.values()) + sum(sending_weights.values()))
            completed_bandwidth_stage = False
            
            for task, weight in sending_weights.items():
                if task.required_results_data - task.sending_results_progress <= weight * bandwidth_unit:
                    sending_resources: float = task.required_results_data - task.sending_results_progress
                    task.allocate_sending_resources(sending_resources, time_step)
                    
                    available_bandwidth -= sending_resources
                    available_storage -= task.loading_progress
                    
                    completed_bandwidth_stage = True
                    
                    sending_weights.pop(task)
            
            for task, weight in loading_weights.items():
                if task.required_storage - task.loading_progress <= weight * bandwidth_unit and \
                        task.loading_progress + min(task.required_storage - task.loading_progress,
                                                    weight * bandwidth_unit) <= available_storage:
                    loading_resources: float = task.required_storage - task.loading_progress
                    task.allocate_loading_resources(loading_resources, time_step)
                    
                    available_bandwidth -= loading_resources
                    available_storage -= task.loading_progress
                    
                    completed_bandwidth_stage = True
                    
                    loading_weights.pop(task)
        
        if loading_weights or sending_weights:
            bandwidth_unit: float = available_bandwidth / (
                    sum(loading_weights.values()) + sum(sending_weights.values()))
            if loading_weights:
                for task, weight in loading_weights.items():
                    task.allocate_loading_resources(bandwidth_unit * weight, time_step)
            
            if sending_weights:
                for task, weight in sending_weights.items():
                    task.allocate_sending_resources(bandwidth_unit * weight, time_step)
    
    def _assert_auction_actions(self, actions):
        pass
    
    def _assert_resource_allocation_actions(self, actions):
        pass
