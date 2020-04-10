
import numpy as np

from env.environment import OnlineFlexibleResourceAllocationEnv, StepType


def human_test():
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')

    (states, step_type), done = env.reset(), False
    while not done:
        if step_type is StepType.AUCTION:
            print(f'Auction step')
            actions = {}
            for server, state in states.items():
                print(f'{server.name} Server: {state} ({len(state)})')
                actions[server] = float(input('Price: '))
        else:
            print(f'Resource allocation step')
            actions = {}
            for server, state in states.items():
                task_actions = []
                print(f'{server.name} Server: {state} ({len(state)})')
                for task in range(len(state)):
                    task_actions.append(float(input('Task weighting: ')))
                actions[server] = task_actions

        (state, step_type), reward, done = env.step(actions)
        print(f'Done: {done}, Rewards: {reward}')


def machine_test():
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')
    server_states, step_type = env.reset()
    random_actions(env, server_states, step_type, steps=20)
    env.save_env('settings/auctions.env')
    random_actions(env, server_states, step_type, steps=5)
    env.save_env('settings/resource_allocation.env')


if __name__ == '__main__':
    machine_test()
    # human_test()
