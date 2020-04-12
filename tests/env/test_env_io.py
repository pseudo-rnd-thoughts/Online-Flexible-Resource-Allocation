from env.environment import OnlineFlexibleResourceAllocationEnv
from tests.env.test_env_step import rnd_action


def test_env_save_load():
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')
    server_state, step_type = env.reset()
    for _ in range(40):
        server_state, step_type, rewards, done = rnd_action(env, server_state, step_type)
        if done:
            break

    env.save_env('settings/tmp/test_save.env')

    load_env, (loaded_server_state, loaded_step_type) = OnlineFlexibleResourceAllocationEnv.load_env('settings/tmp/test_save.env')
    for task, loaded_task in zip(env.unallocated_tasks, load_env.unallocated_tasks):
        assert task.deep_eq(loaded_task)
    # TODO assert the same server and tasks


def test_env_load_settings():
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')
    env.reset()
    assert env.time_step == 0
