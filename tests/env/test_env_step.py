"""Testing the environment step function"""

from __future__ import annotations

from typing import Dict

from tqdm import tqdm

from agents.heuristic_agents.random_agent import RandomTaskPricingAgent, RandomResourceWeightingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.server import Server
from env.task import Task


def test_env_step_rnd_action():
    """
    Tests the environment works with random actions
    """
    print()

    # Generate the environment
    env = OnlineFlexibleResourceAllocationEnv('env/settings/basic.env')

    # Random action agents
    random_task_pricing, random_resource_weighting = RandomTaskPricingAgent(0), RandomResourceWeightingAgent(0)

    # Run the environment multiple times
    for _ in tqdm(range(200)):
        state = env.reset()

        # Number of auction opportunities
        num_auction_opportunities = len(env._unallocated_tasks) + (1 if state.auction_task else 0)
        # Number of auction and resource allocation steps taken
        num_auctions, num_resource_allocations = 0, 0
        # Number of environment server
        num_servers = len(state.server_tasks)

        # Take random steps over the environment
        done = False
        while not done:
            # Check that the number of servers is constant
            assert len(state.server_tasks) == num_servers

            # Generate the actions
            if state.auction_task:
                actions: Dict[Server, float] = {
                    server: random_task_pricing.bid(state.auction_task, allocated_tasks, server, state.time_step)
                    for server, allocated_tasks in state.server_tasks.items()
                }
                num_auctions += 1
            else:
                actions: Dict[Server, Dict[Task, float]] = {
                    server: random_resource_weighting.weight(tasks, server, state.time_step)
                    for server, tasks in state.server_tasks.items()
                }
                num_resource_allocations += 1

            # Take the action on the environment
            state, reward, done, info = env.step(actions)
            assert all(task.auction_time <= state.time_step <= task.deadline
                       for _, tasks in state.server_tasks.items() for task in tasks)

        # Check that the number of auction and resource allocation steps are correct
        assert state.auction_task is None
        assert len(env._unallocated_tasks) == 0
        assert num_auctions == num_auction_opportunities
        assert num_resource_allocations == env._total_time_steps + 1


def test_env_auction_step():
    env, state = OnlineFlexibleResourceAllocationEnv.load_env('settings/auctions.env')

    server_0, server_1, server_2 = list(state.server_tasks.keys())
    assert server_0.name == 'Basic 0' and server_1.name == 'Basic 1' and server_2.name == 'Basic 2'

    # Tests a normal circumstance for the Vickrey auction with second price winning
    actions = {
        server_0: 1.0,
        server_1: 3.0,
        server_2: 0.0
    }

    next_state, rewards, done, info = env.step(actions)
    assert server_0 in rewards and rewards[server_0] == 3.0
    assert len(state.server_tasks[server_0]) + 1 == len(next_state.server_tasks[server_0]) and \
        len(state.server_tasks[server_1]) == len(next_state.server_tasks[server_1]) and \
        len(state.server_tasks[server_2]) == len(next_state.server_tasks[server_2])
    state = next_state

    # Test a case where server provide the same price
    actions = {
        server_0: 3.0,
        server_1: 3.0,
        server_2: 0.0
    }
    next_state, rewards, done, _ = env.step(actions)
    assert (server_0 in rewards and rewards[server_0] == 3.0) or (server_1 in rewards and rewards[server_1] == 3.0)
    assert len(next_state.server_tasks[server_0]) == len(state.server_tasks[server_0]) + 1 or \
        len(next_state.server_tasks[server_1]) == len(state.server_tasks[server_1]) + 1

    # Test where no server provides a price
    actions = {
        server_0: 0.0,
        server_1: 0.0,
        server_2: 0.0
    }
    state, rewards, done, _ = env.step(actions)
    assert len(rewards) == 0

    # Test where only a single server provides a price
    actions = {
        server_0: 1.0,
        server_1: 0.0,
        server_2: 0.0
    }
    next_state, rewards, done, _ = env.step(actions)
    assert server_0 in rewards and rewards[server_0] == 1.0
    assert len(next_state.server_tasks[server_0]) == len(state.server_tasks[server_0]) + 1

    # Test all of the server bid
    actions = {
        server_0: 2.0,
        server_1: 3.0,
        server_2: 1.0
    }
    state, rewards, done, _ = env.step(actions)
    assert server_2 in rewards and rewards[server_2] == 2.0


def test_env_resource_allocation_step():
    print()

    env, state = OnlineFlexibleResourceAllocationEnv.load_env('env/settings/resource_allocation.env')
    print(state)
    # TODO
