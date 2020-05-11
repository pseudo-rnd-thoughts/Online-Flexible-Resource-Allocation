from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

"""
Eval for Algorithms
    c51 - 05-08_01-15-40
    ddpg - 05-08_01-15-42
    ddqn - 05-03_22-38-42
    dqn - 05-07_14-50-10
    dueling dqn - 05-03_22-50-42
    td3 - 05-08_01-16-31
    td3 central critic - 05-08_02-32-54
    resource_weighting_c51 - 05-10_21-43-21
    
Eval for Environment settings and number of agents
    multi agents multi envs - 05-02_18-20-28
    multi agents single env - 05-02_18-20-28
    single agent multi envs - 05-02_18-20-28
    single agent single env - 05-02_18-20-27
    
    multi agents multi envs single env - 05-02_18-20-28
    multi agents single env single env - 05-02_18-20-28
    single agent multi envs single env - 05-02_18-20-28
    single agent single env single env - 05-02_18-20-27
    
Eval for Network architectures 
    bidirectional - 05-03_04-37-31
    gru - 05-03_22-38-42
    lstm - 05-03_03-21-05
    rnn - 05-03_22-38-42
    seq2seq - 05-07_15-54-14
"""


def load_csv(filename, name):
    df = pd.read_csv(filename)
    df['name'] = name.replace('_', ' ').upper()
    return df


def graph_results(graph_name: str, save_filename: str, names: Sequence[str], window=3):
    df = {
        name: load_csv(f'{graph_name}/{name}.csv', name)
        for name in names
    }

    plt.figure(figsize=(8, 4))
    for name, _df in df.items():
        plt.plot(_df['Step'], _df['Value'].rolling(window=window).mean(), label=name.replace('_', ' ').title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../final_report/figures/5_evaluation_figs/{save_filename}.png', )
    plt.show()


if __name__ == "__main__":
    # Algo
    algo_names = ['dqn', 'double_dqn', 'dueling_dqn', 'categorical_dqn', 'resource_weighting_c51',
                  'ddpg', 'td3', 'td3_central_critic']
    for results_name in ['num_completed_tasks', 'num_failed_tasks', 'percent_tasks']:
        graph_results(f'algorithms/{results_name}', f'algo_training_fig/{results_name}', algo_names)

    # Multi-env
    multi_env_names = ['multi_agents_multi_envs', 'multi_agents_single_env',
                       'single_agent_multi_envs', 'single_agent_single_env']
    for results_name in ['num_completed_tasks', 'num_failed_tasks', 'percent_tasks']:
        graph_results(f'env_agent_num/{results_name}', f'env_agent_num_training_fig/{results_name}', multi_env_names)

    # Single env
    single_env_names = ['multi_agents_multi_envs_single_env', 'multi_agents_single_env_single_env',
                        'single_agent_multi_envs_single_env', 'single_agent_single_env_single_env']
    for results_name in ['num_completed_tasks', 'num_failed_tasks', 'percent_tasks']:
        graph_results(f'env_agent_num/{results_name}', f'env_agent_num_training_fig/single_env_{results_name}', multi_env_names)

    # Network arch
    network_names = ['bidirectional', 'gru', 'lstm', 'rnn', 'seq2seq']
    for results_name in ['num_completed_tasks', 'num_failed_tasks', 'percent_tasks']:
        graph_results(f'network_arch/{results_name}', f'net_arch_training_fig/{results_name}', network_names)
