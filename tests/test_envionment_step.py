"""Testing the environment step function"""

import random as rnd
from tqdm import tqdm

from env.environment import OnlineFlexibleResourceAllocationEnv


def test_env_step_action_rnd():
    print()
    env = OnlineFlexibleResourceAllocationEnv.make('../src/settings/basic_env.json')

    for _ in tqdm(range(20)):
        state = env.reset()

        num_tasks = len(env.unallocated_tasks) + (1 if state.auction_task else 0)
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
                    server: rnd.randint(1, 20) for server in state.server_tasks.keys()
                }
                # print(f'\tAuction of {state.auction_task.name}, time step: {state.time_step}')
                auctioned_tasks += 1
            else:
                actions = {
                    server: {
                        task: rnd.randint(1, 20)
                        for task in tasks
                    }
                    for server, tasks in state.server_tasks.items()
                }
                # print(f'\tResource allocation')
            # print(f'Step, time step: {env.state.time_step}')
            state, reward, done, info = env.step(actions)

        # print(f'Num unallocated tasks: {num_tasks}, auctioned tasks: {auctioned_tasks}\n')
        assert num_tasks == auctioned_tasks


def test_env_auction_step():
    # Todo
    pass


def test_env_resource_allocation_step():
    # Todo
    pass
