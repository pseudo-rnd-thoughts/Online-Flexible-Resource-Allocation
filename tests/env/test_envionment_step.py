"""Testing the environment step function"""

from __future__ import annotations

from typing import Dict, List

from tqdm import tqdm

from agents.heuristic_agent.human import HumanTaskPricing
from agents.heuristic_agent.random import RandomTaskPricingAgent, RandomResourceWeightingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv, StepType
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_env_step_rnd_action():
    print()
    env = OnlineFlexibleResourceAllocationEnv('../../src/env/settings/basic.env')
    random_task_pricing, random_resource_weighting = RandomTaskPricingAgent(0), RandomResourceWeightingAgent(0)

    for _ in tqdm(range(20)):
        (states, step_type), done = env.reset(), False

        num_tasks = len(env.unallocated_tasks) + (1 if step_type == StepType.AUCTION else 0)
        # print(f'Auction tasks ({num_tasks}) at [' + ', '.join(str(task.auction_time) for task in env.unallocated_tasks) + ']')
        auctioned_tasks = 0

        num_servers = len(states)
        # print(f'Num of servers: {num_servers}')
        while not done:
            # print(f'\tUnallocated tasks: {len(env.unallocated_tasks)}')
            # print(f'State num of servers: {len(state.server_tasks)}')
            assert len(states) == num_servers
            # print(f'Step Type: {step_type}')
            # print(f'Auction task: {env.auction_task}')
            if step_type == StepType.AUCTION:
                assert all(len(state) == len(env.server_tasks[server]) + 1 for server, state in states.items()), states
                actions: Dict[Server, float] = {
                    server: random_task_pricing.bid(state)
                    for server, state in states.items()
                }
                # print(f'\tAuction of {state.auction_task.name}, time step: {state.time_step}')
                auctioned_tasks += 1
            else:
                # print(f'Auction states: {states}')
                assert all(len(state) == len(env.server_tasks[server]) for server, state in states.items()), states
                actions: Dict[Server, List[float]] = {
                    server: random_resource_weighting.weight(state)
                    for server, state in states.items()
                }
                assert all(len(action) == len(env.server_tasks[server]) for server, action in actions.items()), actions
                # print(f'\tResource allocation')
            # print(f'Step, time step: {env.state.time_step}')
            # print(f'Actions: {actions}')
            (state, step_type), reward, done = env.step(actions)
            assert all(task.auction_time <= state.time_step <= task.deadline
                       for _, tasks in env.server_tasks.items() for task in tasks)

        # print(f'Num unallocated tasks: {num_tasks}, auctioned tasks: {auctioned_tasks}\n')
        assert len(env.unallocated_tasks) == 0
        assert num_tasks == auctioned_tasks


# noinspection DuplicatedCode
def test_env_auction_step():
    print()
    human_task_pricing = HumanTaskPricing(0)

    servers_tasks = {
        Server('Test', 220.0, 35.0, 22.0): [
            Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0, price=1),
            Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0, price=1),
            Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0, price=1)
        ]
    }
    tasks = [
        Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    ]
    env, (states, step_type) = OnlineFlexibleResourceAllocationEnv.custom_env('auction step test', 3, servers_tasks, tasks)
    assert step_type == StepType.AUCTION
    print('State')
    print(states)

    actions = {
        server: human_task_pricing.bid(state)
        for server, state in states.items()
    }

    next_state, rewards, done = env.step(actions)
    print('Next state')
    print(next_state)

    print('Rewards - [' + ', '.join(f'{task.name} Task: {price}' for task, price in rewards.items()) + ']')


# noinspection DuplicatedCode
def test_env_resource_allocation_step():
    print()
    servers_tasks = {
        Server('Test', 220.0, 35.0, 22.0): [
            Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.LOADING, loading_progress=50.0, price=1),
            Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0, price=1),
            Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0, price=1)
        ]
    }
    env, (states, step_type) = OnlineFlexibleResourceAllocationEnv.custom_env('resource weighting step test', 5, servers_tasks, [])
    assert step_type == StepType.RESOURCE_ALLOCATION
    print('State')
    print(states)

    actions = {
        server: [1.0, 1.0, 2.0]
        for server, state in states.items()
    }

    next_state, rewards, done, info = env.step(actions)
    print('Next state')
    print(next_state)

    print('rewards - {' + ', '.join(f'{server.name} server: [' + ', '.join(f'{task.name} Task: {task.stage}' for task in tasks) + ']'
                                    for server, tasks in rewards.items()) + '}')
