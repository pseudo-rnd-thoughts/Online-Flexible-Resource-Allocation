from agents.rl_agents.policy import RandomPolicy
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_env_save_load():
    env = OnlineFlexibleResourceAllocationEnv.make('settings/basic.env')
    env_state = env.reset()

    random_policy = RandomPolicy()

    for _ in range(40):
        actions = {server: random_policy.action() for server, tasks in env_state.server_tasks.items()}
        env_state, rewards, done = env.step(actions)

    env.save('settings/tmp/auction.env')
    loaded_env, loaded_env_state = env.load('settings/tmp/auction.env')

    for task, loaded_task in zip(env.unallocated_tasks, loaded_env.unallocated_tasks):
        assert task == loaded_task
    for server, tasks in env.server_tasks.items():
        pass


def test_env_load_settings():
    env = OnlineFlexibleResourceAllocationEnv.make('settings/basic.env')
    env_state = env.reset()
