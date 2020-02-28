"""Training file"""

from __future__ import annotations

# Load the important classes
import random as rnd
from typing import Dict, TYPE_CHECKING, List, Tuple, Optional

import matplotlib.pyplot as plt

import core.log as log
import settings.env_io as env_io
import settings.env_settings_io as env_settings_io
from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.server import Server
    import numpy as np
    from env.task import Task


def run_env(env: OnlineFlexibleResourceAllocationEnv, server_task_pricing_agents: Dict[Server, TaskPricingAgent],
            server_resource_allocation_agents: Dict[Server, ResourceWeightingAgent],
            update_agent_experience_replay: bool = True) -> Tuple[float, int, int]:
    state = env.state
    log.info(f'Initial state: {str(state)}')

    log.info(f"Task Pricing agents - {{{', '.join([f'{server.name} Server: {task_pricing_agent.name}' for server, task_pricing_agent in server_task_pricing_agents.items()])}}}")
    log.info(f"Resource allocation agents - {{{', '.join([f'{server.name} Server: {resource_weighting_agent.name}' for server, resource_weighting_agent in server_resource_allocation_agents.items()])}}}")

    total_price, num_completed_tasks, num_failed_tasks = 0.0, 0, 0
    server_auction_observations: Dict[Server, Optional[Tuple[np.Array, float, Optional[Task]]]] = {
        server: None for server in state.server_tasks.keys()}
    auction_trajectories: List[Tuple[Task, Server, np.Array, float, Optional[np.Array]]] = []
    done = False
    log.info(f'Initial State: {str(state)}')
    while not done:
        if state.auction_task is not None:
            actions = {
                server: server_task_pricing_agents[server].price(state.auction_task, server,
                                                                 state.server_tasks[server], state.time_step)
                for server in state.server_tasks.keys()
            }
            log.info('Auction prices -> ' + ', '.join([f'{server.name}: {price}' for server, price in actions.items()]))

            next_state, rewards, done, info = env.step(actions)
            log.info(f"Auction Rewards - {', '.join([f'{server.name} Server: {price}' for server, price in rewards.items()])}")
            log.info(f'Next State: {str(next_state)}\n')

            if update_agent_experience_replay:
                for server in state.server_tasks.keys():
                    if server_auction_observations[server]:
                        observation, action, auction_task = server_auction_observations[server]
                        next_observation = server_task_pricing_agents[server].network_observation(state.auction_task, state.server_tasks[server], server, state.time_step)

                        if auction_task:
                            log.debug(f'Added auction task ({auction_task.name}) to auction trajectory')
                            auction_trajectories.append((auction_task, server, observation, action, next_observation))
                        else:
                            log.debug(f'Add failed auction task trajectory for {server.name} server')
                            server_task_pricing_agents[server].add_failed_auction_task(observation, action, next_observation)

                    if server in rewards:
                        log.debug(f'Updating {server.name} server auction observations with {state.auction_task.name} auction task')
                        server_auction_observations[server] = (server_task_pricing_agents[server].network_observation(state.auction_task, state.server_tasks[server], server, state.time_step), actions[server], state.auction_task)
                    else:
                        log.debug(f'Updating {server.name} server auction observations without auction task')
                        server_auction_observations[server] = (server_task_pricing_agents[server].network_observation(state.auction_task, state.server_tasks[server], server, state.time_step), actions[server], None)

            state = next_state
        else:
            actions = {
                server: {
                    task: server_resource_allocation_agents[server].weight(task, state.server_tasks[server], server, state.time_step) + 1
                    if len(tasks) > 1 else 1
                    for task in tasks
                }
                for server, tasks in state.server_tasks.items()
            }
            log.info('Resource allocation weights -> ' + ', '.join([
                f"{server.name} Server - [{', '.join([f'{task.name} Task: {actions[server][task]}' for task in tasks])}]"
                for server, tasks in actions.items()]))

            next_state, rewards, done, info = env.step(actions)
            log.info(f"Resource allocation Rewards - " + ', '.join(
                [f"{server.name}: [{', '.join([task.name for task in tasks])}]" for server, tasks in rewards.items()]))
            log.info(f'Env Done: {done}')
            log.info(f'Next State: {str(next_state)}\n')

            if update_agent_experience_replay:
                for server, reward_tasks in rewards.items():
                    for reward_task in reward_tasks:
                        log.debug(f'Auction trajectories ({len(auction_trajectories)}): [' +
                                  ','.join([auction_trajectory[0].name for auction_trajectory in auction_trajectories]) + ']')
                        task_trajectory = next(_task_trajectory for _task_trajectory in auction_trajectories if _task_trajectory[0] == reward_task)
                        task, server, observation, action, next_observation = task_trajectory
                        server_task_pricing_agents[server].add_finished_task(observation, action, reward_task, next_observation)

                        if reward_task.stage is TaskStage.COMPLETED:
                            total_price += reward_task.price
                            num_completed_tasks += 1
                        else:
                            total_price -= reward_task.price
                            num_failed_tasks += 1

                for server, tasks in state.server_tasks.items():
                    for task in tasks:
                        observation = server_resource_allocation_agents[server].weight(task, state.server_tasks[server], server, state.time_step)
                        next_task = next((_task for _task in next_state.server_tasks[server] if task == _task), None)  # Get the modified task
                        if next_task:
                            if len(next_state.server_tasks[server]) > 1:
                                next_observation = server_resource_allocation_agents[server]. \
                                    weight(next_task, next_state.server_tasks[server], server, next_state.time_step)
                                server_resource_allocation_agents[server]. \
                                    add_incomplete_task_observation(observation, actions[server][task] - 1, next_observation, rewards[server])
                            else:
                                server_resource_allocation_agents[server]. \
                                    add_incomplete_task_observation(observation, actions[server][task] - 1, None, rewards[server])
                        else:
                            finished_task = next(_task for _task in rewards[server] if _task == task)
                            server_resource_allocation_agents[server]. \
                                add_finished_task(observation, actions[server][task] - 1, finished_task, rewards[server])

            state = next_state
    return total_price, num_completed_tasks, num_failed_tasks


def eval_env(training_envs: List[str], task_pricing_agents: List[TaskPricingAgent],
             resource_weighting_agents: List[ResourceWeightingAgent]):
    log.info('Evaluate environments')
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
    _env = OnlineFlexibleResourceAllocationEnv.make('../settings/basic_env_setting.json')

    # Setup the agents
    _task_pricing_agents = [TaskPricingAgent('Default {}'.format(agent_num)) for agent_num in range(10)]
    _resource_weighting_agents = [ResourceWeightingAgent('Default {}'.format(agent_num)) for agent_num in range(10)]

    # Create the training envs
    _training_envs: List[str] = [f'../settings/eval_envs/eval_env_{training_env_num}.json'
                                 for training_env_num in range(10)]
    for training_env_num in range(10):
        _env.reset()
        env_io.save_environment(_env, f'../settings/eval_envs/eval_env_{training_env_num}.json')

    # Loop over the episodes
    for episode in range(100):
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

    plt.legend()
