"""
Test the Online Flexible Resource Allocation Env step function with a random step test, auction test and resource
    allocation test
"""

import numpy as np

from env.environment import OnlineFlexibleResourceAllocationEnv, StepType


def random_action(env, server_states, step_type):
    if step_type is StepType.AUCTION:
        actions = {
            server: np.random.uniform(0, 10)
            for server, state in server_states.items()
        }
    else:
        actions = {
            server: np.random.uniform(0, 10, len(state))
            for server, state in server_states.items()
        }
    (server_states, step_type), rewards, done = env.step(actions)
    return server_states, step_type, rewards, done


def test_env_rnd_step(num_tests: int = 10):
    print()
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')

    for test_num in range(num_tests):
        server_state, step_type = env.reset()
        done = False

        while not done:
            if step_type is StepType.AUCTION:
                actions = {
                    server: np.random.uniform(0, 10)
                    for server, state in server_state.items()
                }
            else:  # step_type is StepType.RESOURCE_ALLOCATION:
                assert step_type is StepType.RESOURCE_ALLOCATION
                actions = {
                    server: np.random.uniform(0, 10, len(state))
                    for server, state in server_state.items()
                }

            (server_state, step_type), reward, done = env.step(actions)


def test_env_auction_step():
    print()
    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('setting/auction.env')


def test_env_resource_allocation_step():
    print()
    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('settings/resource_allocation.env')
