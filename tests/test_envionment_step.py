"""Testing the environment step function"""

from __future__ import annotations

from tqdm import tqdm

from agents.heuristic_agents.human_agent import HumanTaskPricing, HumanResourceWeighting
from agents.heuristic_agents.random_agent import RandomTaskPricingAgent, RandomResourceWeightingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_env_step_rnd_action():
    print()
    env = OnlineFlexibleResourceAllocationEnv.make('../src/train_agents/env_settings/basic_env.json')
    random_task_pricing = RandomTaskPricingAgent(0)
    random_resource_weighting = RandomResourceWeightingAgent(0)

    for _ in tqdm(range(20)):
        state = env.reset()

        num_tasks = len(env._unallocated_tasks) + (1 if state.auction_task else 0)
        # print(f'Auction tasks ({num_tasks}) at [' + ', '.join(str(task.auction_time) for task in env.unallocated_tasks) + ']')
        auctioned_tasks = 0

        num_servers = len(state.server_tasks)
        # print(f'Num of servers: {num_servers}')

        done = False
        while not done:
            # print(f'\tUnallocated tasks: {len(env.unallocated_tasks)}')
            # print(f'State num of servers: {len(state.server_tasks)}')
            assert len(state.server_tasks) == num_servers
            if state.auction_task:
                actions = {
                    server: random_task_pricing.bid(state.auction_task, allocated_tasks, server, state.time_step)
                    for server, allocated_tasks in state.server_tasks.items()
                }
                # print(f'\tAuction of {state.auction_task.name}, time step: {state.time_step}')
                auctioned_tasks += 1
            else:
                actions = {
                    server: {
                        task: random_resource_weighting.weight(task, tasks, server, state.time_step)
                        for task in tasks
                    }
                    for server, tasks in state.server_tasks.items()
                }
                # print(f'\tResource allocation')
            # print(f'Step, time step: {env.state.time_step}')
            state, reward, done, info = env.step(actions)
            assert all(task.auction_time <= state.time_step <= task.deadline
                       for _, tasks in state.server_tasks.items() for task in tasks)

        # print(f'Num unallocated tasks: {num_tasks}, auctioned tasks: {auctioned_tasks}\n')
        assert num_tasks == auctioned_tasks


# noinspection DuplicatedCode
def test_env_auction_step():
    print()
    human_task_pricing = HumanTaskPricing(0)

    servers_tasks = {
        Server('Test', 220.0, 35.0, 22.0): [
            Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
            Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0),
            Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
        ]
    }
    tasks = [
        Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    ]
    env, state = OnlineFlexibleResourceAllocationEnv.custom_env('auction step test', 3, servers_tasks, tasks)
    print('State')
    print(state)

    actions = {
        server: human_task_pricing.bid(state.auction_task, tasks, server, state.time_step)
        for server, tasks in state.server_tasks.items()
    }

    next_state, rewards, done, info = env.step(actions)
    print('Next state')
    print(next_state)

    print('Rewards - [' + ', '.join(f'{task.name} Task: {price}' for task, price in rewards.items()) + ']')


# noinspection DuplicatedCode
def test_env_resource_allocation_step():
    print()
    human_resource_weighting = HumanResourceWeighting(0)

    servers_tasks = {
        Server('Test', 220.0, 35.0, 22.0): [
            Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
            Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0),
            Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
        ]
    }
    env, state = OnlineFlexibleResourceAllocationEnv.custom_env('resource weighting step test', 5, servers_tasks, [])
    print('State')
    print(state)

    actions = {
        server: {
            task: human_resource_weighting.weight(task, tasks, server, state.time_step)
            for task in tasks
        }
        for server, tasks in state.server_tasks.items()
    }

    next_state, rewards, done, info = env.step(actions)
    print('Next state')
    print(next_state)

    print('rewards - {' + ', '.join(f'{server.name} server: [' + ', '.join(f'{task.name} Task: {task.stage}' for task in tasks) + ']'
                                    for server, tasks in rewards.items()) + '}')
