"""
Test the Online Flexible Resource Allocation Env step function with a random step test, auction test and resource
    allocation test
"""

import numpy as np

from env.environment import OnlineFlexibleResourceAllocationEnv, StepType


def rnd_action(env, server_states, step_type):
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
    auction_actions = [
        {
            'Basic 0': 1.0,
            'Basic 1': 0.0,
            'Basic 2': 3.0
        }, 
        {
            'Basic 0': 5.0,
            'Basic 1': 4.0,
            'Basic 2': 5.0
        }, 
        {
            'Basic 0': 4.0,
            'Basic 1': 4.0,
            'Basic 2': 5.0
        },
        {
            'Basic 0': 0.0,
            'Basic 1': 0.0,
            'Basic 2': 0.0
        }
    ]
    step_num = 0
    
    def generate_actions():
        return { 
            server: auction_actions[step_num][server.name] 
            for server, state in server_state.items() 
        }
    
    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('setting/auction.env')
    server_names = {server.name: server for server, _ in server_state.items()}
    assert step_type is StepType.AUCTION
    
    auction_actions = generate_actions()
    (server_state, step_type), rewards, done = env.step(auction_actions)
    assert rewards[server_names['Basic 0']] == 1.0
    
    step_num += 1
    auction_actions = generate_actions()
    (server_state, step_type), rewards, done = env.step(auction_actions)
    assert rewards[server_names['Basic 1']] == 4.0

    step_num += 1
    auction_actions = generate_actions()
    (server_state, step_type), rewards, done = env.step(auction_actions)
    assert rewards[server_names['Basic 0']] == 4.0 or rewards[server_names['Basic 1']] == 4.0

    step_num += 1
    auction_actions = generate_actions()
    (server_state, step_type), rewards, done = env.step(auction_actions)
    assert len(rewards) == 0


def test_env_resource_allocation_step():
    print()
    env, (server_state, step_type) = OnlineFlexibleResourceAllocationEnv.load_env('settings/resource_allocation.env')

    assert step_type is StepType.RESOURCE_ALLOCATION
