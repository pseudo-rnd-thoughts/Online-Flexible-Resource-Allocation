"""
Tests environment input and output for save and loading of environments
"""

from agents.heuristic_agents.random_agent import RandomTaskPricingAgent, RandomResourceWeightingAgent
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_env_save_load():
    # TODO add comments
    env = OnlineFlexibleResourceAllocationEnv('env/settings/basic.env')
    state = env.reset()

    random_task_pricing = RandomTaskPricingAgent(0)
    random_resource_weighting = RandomResourceWeightingAgent(0)

    for _ in range(40):
        if state.auction_task is not None:
            actions = {
                server: random_task_pricing.bid(state.auction_task, tasks, server, state.time_step)
                for server, tasks in state.server_tasks.items()
            }
        else:
            actions = {
                server: random_resource_weighting.weight(tasks, server, state.time_step)
                for server, tasks in state.server_tasks.items()
            }
        state, rewards, done, info = env.step(actions)

    env.save_env('env/settings/tmp/save.env')
    loaded_env, loaded_env_state = env.load_env('env/settings/tmp/save.env')

    for task, loaded_task in zip(env._unallocated_tasks, loaded_env._unallocated_tasks):
        assert task == loaded_task
    for server, tasks in state.server_tasks.items():
        loaded_server, loaded_tasks = next(((loaded_server, loaded_tasks)
                                            for loaded_server, loaded_tasks in state.server_tasks.items()
                                            if loaded_server.name == server.name), (None, None))
        assert loaded_server is not None and loaded_tasks is not None
        assert server.name == loaded_server.name and server.storage_cap == loaded_server.storage_cap and \
            server.computational_cap == loaded_server.computational_cap and \
            server.bandwidth_cap == loaded_server.bandwidth_cap
        for task, loaded_task in zip(tasks, loaded_tasks):
            assert task.name == loaded_task.name and task.required_storage == loaded_task.required_storage and \
                task.required_computation == loaded_task.required_computation and \
                task.required_results_data == loaded_task.required_results_data and \
                task.auction_time == loaded_task.auction_time and task.deadline == loaded_task.deadline and \
                task.stage is loaded_task.stage and task.loading_progress == loaded_task.loading_progress and \
                task.compute_progress == loaded_task.compute_progress and \
                task.sending_progress == loaded_task.sending_progress and task.price == loaded_task.price
            task.assert_valid()


def test_env_load_settings():
    env = OnlineFlexibleResourceAllocationEnv('env/settings/basic.env')
    env_state = env.reset()
