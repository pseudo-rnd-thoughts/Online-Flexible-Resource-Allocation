"""
Core function for training of agents
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import random as rnd
import tensorflow as tf
import numpy as np

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.rl_agents.rl_agent import TaskPricingRLAgent, ResourceWeightingRLAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.env_state import EnvState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def allocate_agents(state: EnvState, task_pricing_agents: List[TaskPricingAgent],
                    resource_weighting_agents: List[ResourceWeightingAgent]) -> Tuple[Dict[Server, TaskPricingAgent],
                                                                                      Dict[Server, ResourceWeightingAgent]]:
    """
    Todo
    Args:
        state:
        task_pricing_agents:
        resource_weighting_agents:

    Returns:

    """
    server_task_pricing_agents = {
        server: rnd.choice(task_pricing_agents)
        for server in state.server_tasks.keys()
    }
    server_resource_allocation_agents = {
        server: rnd.choice(resource_weighting_agents)
        for server in state.server_tasks.keys()
    }
    return server_task_pricing_agents, server_resource_allocation_agents


def eval_agent(env_filenames: List[str], episode: int,
               task_pricing_agents: List[TaskPricingAgent],
               resource_weighting_agents: List[ResourceWeightingAgent]):
    """
    Todo
    Args:
        env_filenames:
        episode:
        task_pricing_agents:
        resource_weighting_agents:

    Returns:

    """
    total_task_prices, completed_tasks, failed_tasks = 0, 0, 0
    for env_filename in env_filenames:
        eval_env = OnlineFlexibleResourceAllocationEnv.load(env_filename)
        state = eval_env.state
        server_task_pricing_agents, server_resource_allocation_agents = allocate_agents(state, task_pricing_agents,
                                                                                        resource_weighting_agents)

        done = False
        while not done:
            if eval_env.state.auction_task:
                bidding_actions = {
                    server: server_task_pricing_agents[server].auction(state.auction_task, state.server_tasks[server],
                                                                       server, state.time_step)
                    for server in state.server_tasks.keys()
                }
                state, rewards, done, info = eval_env.step(bidding_actions)
                total_task_prices += next((price for price in rewards.values()), 0)
            else:
                weighting_actions = {
                    server: {
                        task: server_resource_allocation_agents[server].weight(task, tasks, server, state.time_step)
                        for task in tasks
                    }
                    for server, tasks in state.server_tasks.items()
                }
                state, rewards, done, info = eval_env.step(weighting_actions)
                for finished_tasks in rewards.values():
                    for finished_task in finished_tasks:
                        if finished_task.stage is TaskStage.COMPLETED:
                            completed_tasks += 1
                        else:
                            failed_tasks += 1

    tf.summary.scalar('Eval total price', total_task_prices, step=episode)
    tf.summary.scalar('Eval total completed tasks', completed_tasks, step=episode)
    tf.summary.scalar('Eval total failed tasks', failed_tasks, step=episode)


def train_agent(training_env: OnlineFlexibleResourceAllocationEnv, episode: int,
                task_pricing_agents: List[TaskPricingRLAgent],
                resource_weighting_agents: List[ResourceWeightingRLAgent]):
    """
    Todo
    Args:
        training_env:
        episode:
        task_pricing_agents:
        resource_weighting_agents:

    Returns:

    """
    state = training_env.reset()
    server_task_pricing = {
        server: rnd.choice(task_pricing_agents)
        for server in state.server_tasks.keys()
    }
    server_resource_allocation = {
        server: rnd.choice(resource_weighting_agents)
        for server in state.server_tasks.keys()
    }

    # Update rl_agents experience replay variables to store the previous auction observations and trajectories of successful auctions
    server_auction_observations: Dict[Server, Optional[Tuple[np.ndarray, float, Optional[Task]]]] = {
        server: None for server in state.server_tasks.keys()
    }
    auction_trajectories: List[Tuple[Task, Server, np.Array, float, Optional[np.ndarray]]] = []

    # Loops over the environment till the environment ends
    done = False
    while not done:
        assert len(state.server_tasks) > 0
        # If the state has a task to be auctioned then find the pricing actions from each servers
        if state.auction_task:
            actions = {
                server: server_task_pricing[server].auction(state.auction_task, state.server_tasks[server],
                                                            server, state.time_step)
                for server in state.server_tasks.keys()
            }

            # Make the environment step with the pricing action
            next_state, rewards, done, info = training_env.step(actions)

            # If to update rl_agents experience replay then need to add the old server trajectory with the new observation
            #   and add successful server auctions to the auction trajectory list
            for server in state.server_tasks.keys():
                # If a server auction observation exists
                if server_auction_observations[server]:
                    # Get the old observation, action and auction task
                    obs, action, auction_task = server_auction_observations[server]
                    # Get the new observation
                    next_obs = server_task_pricing[server]. \
                        network_observation(state.auction_task, state.server_tasks[server], server, state.time_step)

                    # If the server won the auction task then add to the auction trajectory list
                    if auction_task:
                        auction_trajectories.append((auction_task, server, obs, action, next_obs))
                    else:
                        # Else add the observation as a failure to the task pricing rl_agents
                        server_task_pricing[server].add_failed_auction_task(obs, action, next_obs)

                # If the server won the task (then will be in the rewards) then add to the observation with the auction task
                observation = server_task_pricing[server].network_observation(state.auction_task,
                                                                              state.server_tasks[server],
                                                                              server, state.time_step)
                if server in rewards:
                    server_auction_observations[server] = (observation, actions[server], state.auction_task)
                else:
                    server_auction_observations[server] = (observation, actions[server], None)

            # Update the state with the next state
            state = next_state
        else:
            # Resource allocation stage

            # The actions are a dictionary of weighting for each server task
            actions = {
                server: {
                    task: server_resource_allocation[server].weight(task, state.server_tasks[server],
                                                                    server, state.time_step)
                    for task in tasks
                }
                for server, tasks in state.server_tasks.items()
            }

            # Apply the resource weighting actions to the environment
            next_state, rewards, done, info = training_env.step(actions)

            # For each task completed (or failed) then update the auction trajectory
            for server, reward_tasks in rewards.items():
                for reward_task in reward_tasks:
                    # Find the relevant auction trajectory where the auction task name is equal to the finished task
                    task_trajectory = next(_task_trajectory for _task_trajectory in auction_trajectories if
                                           _task_trajectory[0] == reward_task)
                    auction_trajectories.remove(task_trajectory)
                    task, server, obs, action, next_obs = task_trajectory

                    # Update the task pricing rl_agents with the finished auction task
                    server_task_pricing[server].add_finished_task(obs, action, reward_task, next_obs)

            # Add the experience for the resource allocation rl_agents
            for server, tasks in state.server_tasks.items():
                if len(tasks) > 1:
                    for task in tasks:
                        # Get last observation
                        obs = server_resource_allocation[server].\
                            network_observation(task, state.server_tasks[server], server, state.time_step)

                        # Get the modified task in the next state, the task may be missing as the task was finished
                        next_task = next((_task for _task in next_state.server_tasks[server] if task == _task), None)
                        # If the task wasn't finished
                        if next_task:
                            # If the next state contains other task than the modified task
                            if len(next_state.server_tasks[server]) > 1:
                                # Get the next observation (imagining that no new tasks were auctioned)
                                next_obs = server_resource_allocation[server]. \
                                    network_observation(next_task, next_state.server_tasks[server], server,
                                                        next_state.time_step)

                                # Add the task observation with the rewards of other tasks completed
                                server_resource_allocation[server]. \
                                    add_incomplete_task_observation(obs, actions[server][task], next_obs, rewards[server])
                            else:
                                # Add the task observation but without the next observations
                                server_resource_allocation[server]. \
                                    add_incomplete_task_observation(obs, actions[server][task], None, rewards[server])
                        else:
                            # Task was finished so finds the finished tasks in rewards
                            finished_task: Task = next(_task for _task in rewards[server] if _task == task)
                            # Update the resource allocation rl_agents with the
                            server_resource_allocation[server]. \
                                add_finished_task(obs, actions[server][task], finished_task, rewards[server])

            # Update the state with the next state
            state = next_state


def run_training(training_env: OnlineFlexibleResourceAllocationEnv, eval_envs: List[str],
                 total_episodes: int, task_pricing_agents: List[TaskPricingRLAgent],
                 resource_weighting_agents: List[ResourceWeightingRLAgent], eval_frequency: int):
    """
    Todo
    Args:
        training_env:
        eval_envs:
        total_episodes:
        task_pricing_agents:
        resource_weighting_agents:
        eval_frequency:

    Returns:

    """
    # Loop over the episodes
    for episode in range(total_episodes):
        train_agent(training_env, episode, task_pricing_agents, resource_weighting_agents)

        # Every eval_frequency episodes, the agents are evaluated
        if episode % eval_frequency == 0:
            eval_agent(eval_envs, episode, task_pricing_agents, resource_weighting_agents)
