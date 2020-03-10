"""
Core function for training of agents
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import random as rnd
import tensorflow as tf

from env.env_state import EnvState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task_stage import TaskStage
from agents.rl_agents.rl_agent import AgentState

if TYPE_CHECKING:
    from env.server import Server
    from agents.task_pricing_agent import TaskPricingAgent
    from agents.resource_weighting_agent import ResourceWeightingAgent
    from agents.rl_agents.rl_agent import TaskPricingRLAgent, ResourceWeightingRLAgent


def allocate_agents(state: EnvState, task_pricing_agents: List[TaskPricingAgent],
                    resource_weighting_agents: List[ResourceWeightingAgent]) \
        -> Tuple[Dict[Server, TaskPricingAgent], Dict[Server, ResourceWeightingAgent]]:
    """
    Allocates agents to servers

    Args:
        state: Environment state with a list of servers
        task_pricing_agents: List of task pricing agents
        resource_weighting_agents: List of resource weighting agents

    Returns: A tuple of dictionaries, one for the server, task pricing agents and
        the other, server, resource weighting agents

    """
    server_task_pricing_agents: Dict[Server, TaskPricingAgent] = {
        server: rnd.choice(task_pricing_agents)
        for server in state.server_tasks.keys()
    }
    server_resource_allocation_agents: Dict[Server, ResourceWeightingAgent] = {
        server: rnd.choice(resource_weighting_agents)
        for server in state.server_tasks.keys()
    }
    return server_task_pricing_agents, server_resource_allocation_agents


def eval_agent(env_filenames: List[str], episode: int,
               task_pricing_agents: List[TaskPricingAgent],
               resource_weighting_agents: List[ResourceWeightingAgent]):
    """
    Evaluation of agents using a list of preset environments

    Args:
        env_filenames: Evaluation environment filenames
        episode: The episode of evaluation
        task_pricing_agents: List of task pricing agents
        resource_weighting_agents: List of resource weighting agents

    """

    total_task_prices, completed_tasks, failed_tasks = 0, 0, 0
    for env_filename in env_filenames:
        eval_env, state = OnlineFlexibleResourceAllocationEnv.load(env_filename)
        server_task_pricing_agents, server_resource_allocation_agents = allocate_agents(state, task_pricing_agents,
                                                                                        resource_weighting_agents)

        done = False
        while not done:
            if state.auction_task:
                bidding_actions = {
                    server: server_task_pricing_agents[server].bid(state.auction_task, state.server_tasks[server],
                                                                   server, state.time_step)
                    for server in state.server_tasks.keys()
                }
                state, rewards, done, info = eval_env.step(bidding_actions)
            else:
                weighting_actions = {
                    server: {
                        task: server_resource_allocation_agents[server].weight(task, tasks, server, state.time_step)
                        for task in tasks
                    }
                    for server, tasks in state.server_tasks.items()
                }
                state, rewards, done, info = eval_env.step(weighting_actions)
                for finished_tasks in rewards.values():
                    for finished_task in finished_tasks:
                        total_task_prices += finished_task.price * (1 if finished_task.stage is TaskStage.COMPLETED else -1)
                        if finished_task.stage is TaskStage.COMPLETED:
                            completed_tasks += 1
                        else:
                            failed_tasks += 1

    print(f'Eval {episode} - total price: {total_task_prices}, completed tasks: {completed_tasks}, failed tasks: {failed_tasks}')
    tf.summary.scalar('Eval total price', total_task_prices, step=episode)
    tf.summary.scalar('Eval total completed tasks', completed_tasks, step=episode)
    tf.summary.scalar('Eval total failed tasks', failed_tasks, step=episode)


def train_agent(training_env: OnlineFlexibleResourceAllocationEnv, task_pricing_agents: List[TaskPricingRLAgent],
                resource_weighting_agents: List[ResourceWeightingRLAgent]):
    """
    Trains reinforcement learning agents through the provided environment

    Args:
        training_env: Training environment used
        task_pricing_agents: A list of reinforcement learning task pricing agents
        resource_weighting_agents: A list of reinforcement learning resource weighting agents

    """
    # Reset the environment getting a new training environment for this episode
    state = training_env.reset()

    # Allocate the servers with their random task pricing and resource weighting agents
    server_task_pricing: Dict[Server, TaskPricingRLAgent] = {
        server: rnd.choice(task_pricing_agents) for server in state.server_tasks.keys()
    }
    server_resource_allocation: Dict[Server, ResourceWeightingRLAgent] = {
        server: rnd.choice(resource_weighting_agents) for server in state.server_tasks.keys()
    }

    # Store each server's auction observations with it being (None for first auction because no observation was seen previously)
    #   the agent state for the auction (auction task, server tasks, server, time), the action taken and if the auction task was won
    server_auction_agent_states: Dict[Server, Optional[Tuple[AgentState, float, bool]]] = {
        server: None for server in state.server_tasks.keys()
    }

    # For successful auctions, then the agent state of the winning bid, the action taken and the following observation are
    #   all stored in order to be added as an agent observation after the task finishes in order to know if the task was completed or not
    successful_auction_agent_states: List[Tuple[AgentState, float, AgentState]] = []

    # The environment is looped over till the environment is done (the current time step > environment total time steps)
    done = False
    while not done:
        # If the state has a task to be auctioned then find the pricing of each servers as the action
        if state.auction_task:
            # Get the bids for each server
            auction_prices = {
                server: server_task_pricing[server].bid(state.auction_task, server_tasks, server, state.time_step)
                for server, server_tasks in state.server_tasks.items()
            }

            # Environment step using the pricing actions to get the next state, rewards, done and info
            next_state, rewards, done, info = training_env.step(auction_prices)

            # Update the server_auction_observations and auction_trajectories variables with the new next_state info
            for server in state.server_tasks.keys():
                # Generate the current agent's state
                current_agent_state = AgentState(state.auction_task, state.server_tasks[server], server,
                                                 state.time_step)

                if server_auction_agent_states[server]:  # If a server auction observation exists
                    # Get the last time steps agent state, action and if the server won the auction
                    previous_agent_state, previous_action, previous_auction_win = server_auction_agent_states[server]

                    # If the server won the auction in the last time step then add the info to the auction trajectories
                    if previous_auction_win:
                        successful_auction_agent_states.append(
                            (previous_agent_state, previous_action, current_agent_state))
                    else:
                        # Else add the agent state to the agent's replay buffer as a failed auction bid
                        # Else add the observation as a failure to the task pricing rl_agents
                        server_task_pricing[server].failed_auction_bid(previous_agent_state, previous_action,
                                                                       current_agent_state)

                # Update the server auction agent states with the current agent state
                server_auction_agent_states[server] = (current_agent_state, auction_prices[server], server in rewards)
        else:  # Else the environment is at resource allocation stage
            # For each server and each server task calculate its relative weighting
            resource_weighting_actions = {
                server: {
                    task: server_resource_allocation[server].weight(task, server_tasks, server, state.time_step)
                    for task in server_tasks
                }
                for server, server_tasks in state.server_tasks.items()
            }

            # Environment step using the resource weighting actions to get the next state, rewards, done and info
            next_state, finished_server_tasks, done, info = training_env.step(resource_weighting_actions)

            # For each server, there are may be finished tasks due to the resource allocation
            #    therefore add the task pricing auction agent states with the finished tasks
            for server, finished_tasks in finished_server_tasks.items():
                for finished_task in finished_tasks:
                    # Get the successful auction agent state from the list of successful auction agent states
                    successful_auction = next(auction_agent_state
                                              for auction_agent_state in successful_auction_agent_states
                                              if auction_agent_state[0].task == finished_task)
                    # Remove the successful auction agent state
                    successful_auction_agent_states.remove(successful_auction)

                    # Unwrap the successful auction agent state tuple
                    auction_agent_state, bid_action, next_agent_state = successful_auction

                    # Add the winning auction bid info to the agent
                    server_task_pricing[server].winning_auction_bid(auction_agent_state, bid_action,
                                                                    finished_task, next_agent_state)

            # Add the agent states for resource allocation
            for server, tasks in state.server_tasks.items():
                # If the number of tasks allocated to the server is greater than 1
                #       (otherwise the weight defaults to 1 as no point in getting a weighting)
                if len(tasks) > 1:
                    # Get the agent state for each task
                    for weighted_task in tasks:
                        # Get the last agent state that generated the weighting
                        last_agent_state = AgentState(weighted_task, state.server_tasks[server], server,
                                                      state.time_step)
                        last_action = resource_weighting_actions[server][weighted_task]

                        # Get the modified task in the next state, the task may be missing if the task is finished
                        updated_task = next((next_task for next_task in next_state.server_tasks[server]
                                             if weighted_task == next_task), None)

                        # If the task wasn't finished
                        if updated_task:
                            # Check if the next state contains other tasks than the updated task
                            if len(next_state.server_tasks[server]) > 1:
                                # Get the next observation (imagining that no new tasks were auctioned)
                                next_agent_state = AgentState(updated_task, next_state.server_tasks[server], server,
                                                              next_state.time_step)

                                # Add the task observation with the rewards of other tasks completed
                                server_resource_allocation[server].allocation_obs(last_agent_state, last_action,
                                                                                  next_agent_state,
                                                                                  finished_server_tasks[server])
                            else:
                                # Add the task observation but without the next observations
                                server_resource_allocation[server].allocation_obs(last_agent_state, last_action, None,
                                                                                  finished_server_tasks[server])
                        else:
                            # The weighted task was finished so using the finished task in the finished_server_tasks dictionary
                            finished_task = next(finished_task for finished_task in finished_server_tasks[server]
                                                 if finished_task == weighted_task)

                            # Update the resource allocation with teh finished task observation
                            server_resource_allocation[server].finished_task_obs(last_agent_state, last_action,
                                                                                 finished_task,
                                                                                 finished_server_tasks[server])

        # Update the state with the next state
        state = next_state


def set_policy(task_pricing_agents: List[TaskPricingRLAgent],
               resource_weighting_agents: List[ResourceWeightingRLAgent], policy: bool):
    for task_pricing_agent in task_pricing_agents:
        task_pricing_agent.eval_policy = policy
    for resource_weighting_agent in resource_weighting_agents:
        resource_weighting_agent.eval_policy = policy


def run_training(training_env: OnlineFlexibleResourceAllocationEnv, eval_envs: List[str],
                 total_episodes: int, task_pricing_agents: List[TaskPricingRLAgent],
                 resource_weighting_agents: List[ResourceWeightingRLAgent], eval_frequency: int):
    """
    Runs the training of the agents for a fixed number of episodes

    Args:
        training_env: The training environments
        eval_envs: The evaluation environment filenames
        total_episodes: The total number of episodes
        task_pricing_agents: The task pricing agents
        resource_weighting_agents: The resource weighting agents
        eval_frequency: The evaluation frequency

    """
    # Loop over the episodes
    for episode in range(total_episodes):
        set_policy(task_pricing_agents, resource_weighting_agents, False)
        train_agent(training_env, task_pricing_agents, resource_weighting_agents)

        # Every eval_frequency episodes, the agents are evaluated
        if episode % eval_frequency == 0:
            set_policy(task_pricing_agents, resource_weighting_agents, True)
            eval_agent(eval_envs, episode, task_pricing_agents, resource_weighting_agents)


def generate_eval_envs(eval_env: OnlineFlexibleResourceAllocationEnv, num_evals: int, folder: str,
                       overwrite: bool = False) -> List[str]:
    """
    Generates and saves the evaluation environment used for evaluating training of the agents

    Args:
        eval_env: The evaluation environment used to generate the files
        num_evals: The number of environments generated
        folder: The folder where the environments are generated
        overwrite: If to overwrite previous environments saved

    Returns: A list of environment file paths

    """

    eval_files = []
    for eval_num in range(num_evals):
        eval_file = f'{folder}/eval_{eval_num}.json'
        eval_files.append(eval_file)
        if overwrite or not os.path.exists(eval_file):
            eval_env.reset()
            eval_env.save(eval_file)

    return eval_files


def setup_tensorboard(folder: str):
    """
    Setups the tensorboard for the training and evaluation results

    Args:
        folder: The folder for the tensorboard

    """
    tf.summary.create_file_writer(folder)
