"""
Tests the environment io functions: load_settings, load_env, save_env
"""

from env.environment import OnlineFlexibleResourceAllocationEnv
from tests.env.test_env_step import rnd_action


def test_env_save_load():
    # Load a random environment
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')

    # Take a fixed number of random steps through the environment
    server_state, step_type = env.reset()
    for _ in range(40):
        server_state, step_type, rewards, done = rnd_action(env, server_state, step_type)
        if done:
            break

    # Save the environment to the tmp folder
    env_file = 'settings/tmp/test_save.env'
    env.save_env(env_file)

    # Load the environment again
    load_env, (loaded_server_state, loaded_step_type) = OnlineFlexibleResourceAllocationEnv.load_env(env_file)

    # Check that the loaded unallocated tasks are equal to the original unallocated tasks
    for task, loaded_task in zip(env.unallocated_tasks, load_env.unallocated_tasks):
        assert task.deep_eq(loaded_task)

    # Check that the loaded state and the new state are equal
    for server, state in server_state.items():
        loaded_server, loaded_state = next((loaded_server, loaded_state)
                                           for loaded_server, loaded_state in loaded_server_state.items()
                                           if loaded_server.name == server.name)
        assert server == loaded_server
        assert state == loaded_state


def test_env_load_settings():
    # Load a setting
    env = OnlineFlexibleResourceAllocationEnv('settings/basic.env')
    env.reset()
    assert env.time_step == 0

    # As the environment is random then it is difficult to test any further about the effectiveness of the environment
