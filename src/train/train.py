"""Training file"""

from __future__ import annotations

# Load the important classes
import random as rnd
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import core.log as log
from settings import env_settings_io, env_io
from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Optional
    import numpy as np

    from env.server import Server
    from env.task import Task


def run_env(env: OnlineFlexibleResourceAllocationEnv, server_task_pricing_agents: Dict[Server, TaskPricingAgent],
            server_resource_allocation_agents: Dict[Server, ResourceWeightingAgent],
            update_agent_experience_replay: bool = True) -> Tuple[float, int, int]:
    """
    Runs the environments with a dictionary of servers to task pricing and resource allocation agents, and
        if to update agent internal experience replay
    Args:
        env: The environment to run
        server_task_pricing_agents: A dictionary of servers to task pricing agents
        server_resource_allocation_agents: A dictionary of servers to resource allocation agents
        update_agent_experience_replay: If to update agent experience replay

    Returns: A tuple of the total price of tasks, the number of completed tasks and the number of failed tasks
    """
    total_price, num_completed_tasks, num_failed_tasks = 0.0, 0, 0

    # Get the environment state and log the relevant info for starting the environment
    state = env.state
    log.info(f'Environment: {str(env)}')
    log.info(f"Task Pricing agents - {{{', '.join([f'{server.name} Server: {task_pricing_agent.name}' for server, task_pricing_agent in server_task_pricing_agents.items()])}}}")
    log.info(f"Resource allocation agents - {{{', '.join([f'{server.name} Server: {resource_weighting_agent.name}' for server, resource_weighting_agent in server_resource_allocation_agents.items()])}}}\n")

    # Update agent experience replay variables to store the previous auction observations and trajectories of successful auctions
    server_auction_observations: Dict[Server, Optional[Tuple[np.Array, float, Optional[Task]]]] = {
        server: None for server in state.server_tasks.keys()}
    auction_trajectories: List[Tuple[Task, Server, np.Array, float, Optional[np.Array]]] = []

    # Loops over the environment till the environment ends
    done = False
    while not done:
        assert len(state.server_tasks) > 0
        # If the state has a task to be auctioned then find the pricing actions from each servers
        if state.auction_task:
            actions = {
                server: server_task_pricing_agents[server].price(state.auction_task, server, state.server_tasks[server], state.time_step)
                for server in state.server_tasks.keys()
            }
            log.info('Auction prices - {' + ', '.join([f'{server.name} Server: {price}' for server, price in actions.items()]) + '}')

            # Make the environment step with the pricing action
            next_state, rewards, done, info = env.step(actions)
            log.info(f"Auction Rewards - {{{', '.join([f'{server.name} Server: {price}' for server, price in rewards.items()])}}}")
            log.info(f'Next State: {str(next_state)}\n')

            # If to update agent experience replay then need to add the old server trajectory with the new observation
            #   and add successful server auctions to the auction trajectory list
            if update_agent_experience_replay:
                for server in state.server_tasks.keys():
                    # If a server auction observation exists
                    if server_auction_observations[server]:
                        # Get the old observation, action and auction task
                        observation, action, auction_task = server_auction_observations[server]
                        # Get the new observation
                        next_observation = server_task_pricing_agents[server].\
                            network_observation(state.auction_task, state.server_tasks[server], server, state.time_step)

                        # If the server won the auction task then add to the auction trajectory list
                        if auction_task:
                            log.debug(f'Added auction task ({auction_task.name}) to auction trajectory')
                            auction_trajectories.append((auction_task, server, observation, action, next_observation))
                        else:
                            # Else add the observation as a failure to the task pricing agent
                            log.debug(f'Add failed auction task trajectory for {server.name} server')
                            server_task_pricing_agents[server].add_failed_auction_task(observation, action, next_observation)

                    # If the server won the task (then will be in the rewards) then add to the observation with the auction task
                    observation = server_task_pricing_agents[server].network_observation(state.auction_task, state.server_tasks[server], server, state.time_step)
                    if server in rewards:
                        log.debug(f'Updating {server.name} server auction observations with {state.auction_task.name} auction task')
                        server_auction_observations[server] = (observation, actions[server], state.auction_task)
                    else:
                        log.debug(f'Updating {server.name} server auction observations without auction task')
                        server_auction_observations[server] = (observation, actions[server], None)

            # Update the state with the next state
            state = next_state
        else:
            # Resource allocation stage

            # The actions are a dictionary of weighting for each server task
            actions = {
                server: {
                    task: server_resource_allocation_agents[server].weight(task, state.server_tasks[server], server, state.time_step)
                    for task in tasks
                }
                for server, tasks in state.server_tasks.items()
            }
            log.info('Resource allocation weights - {' +
                     ', '.join([f'{server.name} Server: [' + ', '.join([f'{task.name} Task: {actions[server][task]}' for task in tasks]) + ']'
                                for server, tasks in actions.items()]) + '}')

            # Apply the resource weighting actions to the environment
            next_state, rewards, done, info = env.step(actions)
            log.info(f"Resource allocation Rewards - {{" +
                     ', '.join([f"{server.name}: [{', '.join([task.name for task in tasks])}]" for server, tasks in rewards.items()]) + '}')
            log.info(f'Env Done: {done}')
            log.info(f'Next State: {str(next_state)}\n')

            # If to update the agents experience replays
            if update_agent_experience_replay:
                # For each task completed (or failed) then update the auction trajectory
                for server, reward_tasks in rewards.items():
                    for reward_task in reward_tasks:
                        # Find the relevant auction trajectory where the auction task name is equal to the finished task
                        task_trajectory = next(_task_trajectory for _task_trajectory in auction_trajectories if _task_trajectory[0] == reward_task)
                        auction_trajectories.remove(task_trajectory)
                        task, server, observation, action, next_observation = task_trajectory

                        # Update the task pricing agent with the finished auction task
                        server_task_pricing_agents[server].add_finished_task(observation, action, reward_task, next_observation)

                        # Update the global environment variables of the total price, num completed tasks and num failed tasks
                        if reward_task.stage is TaskStage.COMPLETED:
                            total_price += reward_task.price
                            num_completed_tasks += 1
                        else:
                            total_price -= reward_task.price
                            num_failed_tasks += 1

                # Add the experience for the resource allocation agent
                for server, tasks in state.server_tasks.items():
                    if len(tasks) > 1:
                        for task in tasks:
                            # Get last observation
                            observation = server_resource_allocation_agents[server].network_observation(task, state.server_tasks[server], server, state.time_step)

                            # Get the modified task in the next state, the task may be missing as the task was finished
                            next_task = next((_task for _task in next_state.server_tasks[server] if task == _task), None)
                            # If the task wasn't finished
                            if next_task:
                                # If the next state contains other task than the modified task
                                if len(next_state.server_tasks[server]) > 1:
                                    # Get the next observation (imagining that no new tasks were auctioned)
                                    next_observation = server_resource_allocation_agents[server]. \
                                        network_observation(next_task, next_state.server_tasks[server], server, next_state.time_step)

                                    # Add the task observation with the rewards of other tasks completed
                                    server_resource_allocation_agents[server]. \
                                        add_incomplete_task_observation(observation, actions[server][task], next_observation, rewards[server])
                                else:
                                    # Add the task observation but without the next observations
                                    server_resource_allocation_agents[server]. \
                                        add_incomplete_task_observation(observation, actions[server][task], None, rewards[server])
                            else:
                                # Task was finished so finds the finished tasks in rewards
                                finished_task: Task = next(_task for _task in rewards[server] if _task == task)
                                # Update the resource allocation agent with the
                                server_resource_allocation_agents[server]. \
                                    add_finished_task(observation, actions[server][task], finished_task, rewards[server])

            # Update the state with the next state
            state = next_state
    # return the resulting information for the total price, number of completed tasks and number of failed tasks
    return total_price, num_completed_tasks, num_failed_tasks


def eval_env(training_envs: List[str], task_pricing_agents: List[TaskPricingAgent],
             resource_weighting_agents: List[ResourceWeightingAgent]):
    log.warning('Evaluate environments')
    total_price, num_completed_tasks, num_failed_tasks = 0, 0, 0

    for training_env in training_envs:
        log.info(f'Training env: {training_env}')
        env = env_settings_io.load_environment(training_env)

        server_task_pricing_agents, server_resource_allocation_agents = allocate_agents(_env, task_pricing_agents, resource_weighting_agents)
        env_total_price, env_num_completed_tasks, env_num_failed_tasks = run_env(env, server_task_pricing_agents, server_resource_allocation_agents,
                                                                                 update_agent_experience_replay=False)
        total_price += env_total_price
        num_completed_tasks += env_num_completed_tasks
        num_failed_tasks += env_num_failed_tasks

    log.warning(f'Eval - Total price: {total_price}, Completed Tasks: {num_completed_tasks}, Failed Tasks: {num_failed_tasks}')
    plt.plot(total_price, label='Total price')
    plt.plot(num_completed_tasks, label='Completed tasks')
    plt.plot(num_failed_tasks, label='Failed tasks')
    plt.legend()


def allocate_agents(env: OnlineFlexibleResourceAllocationEnv, task_pricing_agents: List[TaskPricingAgent],
                    resource_weighting_agents: List[ResourceWeightingAgent]):
    assert env.state.time_step == 0
    assert len(task_pricing_agents) > 0
    assert len(resource_weighting_agents) > 0

    server_task_pricing_agents: Dict[Server, TaskPricingAgent] = {
        server: rnd.choice(task_pricing_agents)
        for server in env.state.server_tasks.keys()
    }
    server_resource_allocation_agents: Dict[Server, ResourceWeightingAgent] = {
        server: rnd.choice(resource_weighting_agents)
        for server in env.state.server_tasks.keys()
    }
    return server_task_pricing_agents, server_resource_allocation_agents


if "__main__" == __name__:
    log.console_debug_level = log.LogLevel.INFO
    log.debug_filename = 'training.log'

    # Setup the environment
    _env = OnlineFlexibleResourceAllocationEnv.make('../settings/basic_env.json')

    # Setup the agents
    _task_pricing_agents = [TaskPricingAgent('Default {} TPA'.format(agent_num)) for agent_num in range(10)]
    _resource_weighting_agents = [ResourceWeightingAgent('Default {} RWA'.format(agent_num)) for agent_num in range(10)]

    # Create the training envs
    _training_envs: List[str] = [f'../settings/eval_envs/eval_env_{training_env_num}.json' for training_env_num in range(10)]
    for training_env_num in range(10):
        _env.reset()
        env_io.save_environment(_env, f'../settings/eval_envs/eval_env_{training_env_num}.json')

    # Loop over the episodes
    for episode in range(1):
        log.info(f'Episode: {episode}')
        _env.reset()

        _server_task_pricing_agents, _server_resource_allocation_agents = allocate_agents(_env, _task_pricing_agents, _resource_weighting_agents)
        _total_price, _num_completed_tasks, _num_failed_tasks = run_env(_env, _server_task_pricing_agents, _server_resource_allocation_agents)
        log.warning(f'Episode {episode} - Total Price: {_total_price}, Num Completed Task: {_num_completed_tasks}, Num Failed Tasks: {_num_failed_tasks}')

        # Every 3 episodes, the agents are trained
        if episode % 3 == 0:
            for _task_pricing_agent in _server_task_pricing_agents.values():
                _task_pricing_agent.train()
            for _resource_weighting_agent in _server_resource_allocation_agents.values():
                _resource_weighting_agent.train()

        # Every 15 episodes, the agents are evaluated
        if episode % 15 == 0:
            eval_env(_training_envs, _task_pricing_agents, _resource_weighting_agents)
