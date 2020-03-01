"""Testing the environment step function"""

import random as rnd
from tqdm import tqdm

from env.environment import OnlineFlexibleResourceAllocationEnv


def test_env_step_action_rnd():
    env = OnlineFlexibleResourceAllocationEnv.make('../settings/basic_env.json')

    for _ in tqdm(range(1000)):
        state = env.reset()
        done = False

        while not done:
            if state.auction_task:
                actions = {
                    server: rnd.randint(1, 20) for server in state.server_tasks.keys()
                }
            else:
                actions = {
                    server: {
                        task: rnd.randint(1, 20)
                        for task in tasks
                    }
                    for server, tasks in state.server_tasks.items()
                }

            state, reward, done, info = env.step(actions)


def test_env_auction_step():
    # Todo
    pass


def test_env_resource_allocation_step():
    # Todo
    pass
