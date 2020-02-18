"""
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
"""

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from core.environment import OnlineFlexibleResourceAllocationEnv


def run(episodes=100000, train_size=64, settings=['settings/basic_env_settings.json']):
    server_task_pricing_agents = {}
    server_resource_allocation_agents = {}
    
    task_pricing_agents = [TaskPricingAgent('TPA {}'.format(agent_num))
                           for agent_num in range(10)]
    resource_weighting_agents = [ResourceWeightingAgent('RWA {}'.format(agent_num))
                                 for agent_num in range(10)]
    
    unfinished_auctioned_tasks = {}
    
    env = OnlineFlexibleResourceAllocationEnv.make(settings)
    state, auction_task = env.reset()
    for episode in range(episodes):
        for _ in range(train_size):
            if auction_task is not None:
                actions = {
                    server: task_pricing_agents[server].price_task(auction_task, server, server_tasks, env.time_step)
                    for server, server_tasks in state.items()
                }
                
                next_state, rewards, done, info = env.step(actions)
                for winning_server, price in rewards.items():
                    unfinished_auctioned_tasks[auction_task] = (winning_server, price, state[winning_server])
                    
            else:
                actions = {
                    server: {
                        task: resource_weighting_agents[server]
                            .weight_task(task, [_task for _task in tasks if _task is not task], server, env.time_step)
                        for task in tasks
                    }
                    for server, tasks in state.items()
                }
                
                (next_state, auction_task), rewards, done, info = env.step(actions)
                for server, tasks in state.items():
                    for task in tasks:
                        if task in next_state[server]:
                            next_task = next_state[server][task]
                            resource_weighting_agents[server].add_task(task, [_task for _task in tasks if _task is not task],
                                                                         next_task, [_task for _task in next_state[server] if _task is not next_task],
                                                                         server, env.time_step, sum(rewards[server].values()))
                        else:
                            resource_weighting_agents[server].add_completed_task(task, [_task for _task in tasks if _task is not task],
                                                                        server, env.time_step,
                                                                        rewards[server][task], sum(rewards[server].values() - rewards[server][task]))
                            
                            _, price, old_state = unfinished_auctioned_tasks[task]
                            if task[9] == 4:
                                task_pricing_agents[server].add_completed_task(task, price, old_state)
                            else:
                                task_pricing_agents[server].add_completed_task(task, -price, old_state)
