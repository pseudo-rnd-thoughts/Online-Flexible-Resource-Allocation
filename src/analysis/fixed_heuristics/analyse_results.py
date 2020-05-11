"""
Analyse fixed vs flexible results
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

agent_completed_tasks = [
    90,  73,  55,  43,  41,  59,  80,  87,  120, 46,  166, 174,  # 420261
    30,  123, 165, 120, 56,  81,  44,  113, 89,  81,  119, 105, 84,  30,  69,  57,  121, 166, 126, 42,   # 420264
    40,  44,  134, 116, 96,  111, 96,  47,  109, 60,  58,  95,  39,  92,  176, 125, 53,  48,  101, 65,   # 420266
    49,  97,  128, 78,  49,  106, 92,  92,  55,  51,  89,  49,  95,  179, 153, 45,  111, 107, 126, 107,  # 420267
    96,  73,  52,  167, 63,  96,  100, 62,  66,  70,  127, 107, 47,  136, 88,  54,  51,  44,  50,  81,   # 420286
    101, 120, 51,  73,  107, 105, 73,  166, 104, 51,  105, 92,  194, 61,  88,  117, 78,  91,  35,  63,   # 420269
    41,  50,  120, 60,  121, 51,  39,  48,  74,  93,  47,  48,  171, 101, 155, 85,  111, 113, 75,  119,  # 420270
    133, 76,  82,  64,  62,  92,  114, 37,  36,  119, 82,  115, 129, 64,  98,  111, 41,  92,  68,  135,  # 420284
    59,  108, 195, 118, 94,  146, 48,  40,  158, 58,  84,  44,  167, 106, 55,  63,  73,  81,  119, 116,  # 420285
    43,  79,  74,  91,  151, 104, 170, 122, 41,  176, 102, 44,  122, 108, 137, 59,  69,  143, 57,  58,   # 420286
    53,  114, 91,  60,  93,  166, 84,  72,  60,  53,  47,  113, 107, 42,  139, 80,  139, 89,  103, 118,  # 420287
]

agent_failed_tasks = [
    3, 0,  1,  0,  17, 3,  1,  6,  1,  3,  24, 33,  # 420261
    1, 4,  18, 0,  3,  1,  1,  4,  11, 5,  19, 27, 0,  0,  4,  7,  4,  24, 22, 1,   # 420264
    2, 1,  15, 3,  10, 7,  34, 12, 35, 2,  2,  0,  1,  44, 19, 3,  2,  1,  2,  15,  # 420266
    6, 31, 36, 13, 1,  3,  47, 4,  2,  1,  9,  12, 1,  10, 17, 9,  0,  33, 18, 2,   # 420267
    2, 7,  1,  20, 1,  5,  0,  5,  15, 8,  1,  41, 1,  18, 1,  1,  3,  2,  0,  4,   # 420286
    6, 26, 0,  2,  48, 1,  3,  21, 5,  0,  15, 25, 26, 1,  4,  30, 52, 66, 0,  11,  # 420269
    4, 1,  17, 0,  27, 0,  0,  1,  4,  6,  1,  7,  28, 49, 19, 10, 41, 31, 4,  20,  # 420270
    6, 13, 2,  9,  1,  2,  2,  0,  0,  5,  11, 26, 1,  1,  5,  1,  1,  1,  10, 2,   # 420284
    2, 1,  15, 0,  1,  37, 4,  1,  23, 2,  48, 0,  37, 1,  6,  10, 32, 8,  4,  18,  # 420285
    1, 6,  1,  40, 26, 0,  20, 17, 2,  34, 1,  10, 2,  2,  5,  4,  8,  43, 24,  0,  # 420286
    6, 25, 6,  0,  2,  17, 0,  0,  4,  1,  0,  0,  10, 8,  27, 2,  8,  11, 35, 26,  # 420287
]

agent_attempted_tasks = [
    143, 140, 148, 131, 148, 152, 234, 248, 202, 146, 214, 250,  # 420261
    131, 243, 205, 250, 127, 234, 153, 205, 126, 133, 204, 248, 199, 141, 126, 143, 229, 216, 214, 131,  # 420264
    146, 142, 250, 207, 137, 145, 222, 122, 214, 146, 142, 222, 150, 246, 234, 214, 156, 143, 218, 133,  # 420266
    135, 203, 201, 124, 154, 213, 227, 200, 157, 138, 210, 138, 190, 202, 202, 123, 223, 220, 210, 203,  # 420267
    207, 136, 142, 234, 140, 135, 203, 155, 127, 132, 224, 228, 122, 204, 239, 146, 152, 131, 152, 136,  # 420286
    239, 204, 155, 125, 203, 200, 140, 213, 138, 154, 240, 221, 243, 142, 125, 206, 239, 236, 139, 132,  # 420269
    131, 137, 237, 147, 249, 122, 124, 130, 232, 208, 159, 155, 242, 243, 218, 205, 232, 214, 120, 204,  # 420270
    215, 134, 232, 127, 153, 239, 246, 140, 142, 219, 128, 246, 247, 138, 130, 210, 132, 196, 150, 225,  # 420284
    132, 221, 220, 203, 239, 214, 127, 152, 201, 146, 249, 147, 248, 220, 139, 141, 224, 149, 228, 204,  # 420285
    141, 194, 122, 224, 239, 224, 222, 238, 137, 237, 128, 128, 215, 232, 233, 123, 133, 240, 149, 133,  # 420286
    158, 242, 221, 128, 225, 210, 210, 195, 134, 157, 159, 207, 227, 137, 232, 130, 224, 126, 243, 224,  # 420287
]

fixed_completed_tasks = [
    34, 25, 25, 15, 19, 25, 18, 42, 39,  16, -1, -1,  # 420261
    14, 45, 79, 40, 22, 26, 20, 42, 33,  34,  61,  55,  13,  9,   29, 20, 45,  82, 69, 18,  # 420264
    14, 23, 62, 41, 36, 42, 54, 22, 61,  25,  23,  31,  20,  62, -1,  43, 26,  23, 36, 29,  # 420266
    22, 53, 68, 35, 18, 35, 56, 31, 26,  23,  34,  21,  39, -1,   73, 19, 41,  57, 59, 44,  # 420267
    35, 32, 16, 82, 28, 38, 30, 26, 29,  30,  47,  59,  18,  73,  29, 22, 23,  19, 21, 32,  # 420286
    34, 61, 15, 28, 66, 32, 25, 82, 43,  17,  44,  55, -1,   22,  37, 61, 54,  59, 11, 33,  # 420269
    22, 16, 55, 22, 70, 18, 18, 20, 27,  30,  23,  26,  84,  61,  71, 43, 62,  64, 32, 58,  # 420270
    53, 32, 18, 26, 28, 34, 51, 17, 15,  46,  37,  60,  47,  24,  38, 37, 18,  33, 31, 49,  # 420284
    24, 43, 85, 41, 27, 82, 19, 20, 82,  25,  57,  15, -1,   44,  19, 25, 46,  29, 43, 58,  # 420285
    16, 20, 27, 56, 69, 42, 84, 69, 20, -1,   37,  20,  46,  42,  54, 20, 27, -1,  30, 25,  # 420286
    25, 60, 37, 24, 34, 74, 24, 20, 27,  26,  25,  46,  43,  18,  69, 32, 48,  35, 57, 66,  # 420287
]


def graph_results():
    plt.figure(figsize=(8, 3))
    plt.hist(agent_completed_tasks, label='Flexible Resource Allocation')
    plt.axvline(agent_completed_tasks.mean(), linewidth=1, color='blue')
    plt.hist(fixed_completed_tasks, label='Fixed Resource Allocation')
    plt.axvline(fixed_completed_tasks.mean(), linewidth=1, color='orange')
    plt.legend()
    plt.xlabel('Number of completed tasks')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig('../../../final_report/figures/5_evaluation_figs/fixed_flexible_completed_tasks.png')
    plt.show()

    task_difference = np.subtract(agent_completed_tasks, fixed_completed_tasks)
    plt.figure(figsize=(8, 3))
    plt.hist(task_difference)
    plt.xlabel('Difference in Tasks completed')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.axvline(task_difference.mean(), linewidth=1, color='blue')

    plt.savefig('../../../final_report/figures/5_evaluation_figs/fixed_flexible_tasks_difference.png')
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.hist(agent_failed_tasks, 50)
    plt.xlabel('Number of tasks failed')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.axvline(agent_failed_tasks.mean(), linewidth=1, color='blue')

    plt.savefig('../../../final_report/figures/5_evaluation_figs/flexible_failed_tasks.png')
    plt.show()

    print(f'Percent difference: {np.divide(agent_failed_tasks, agent_completed_tasks).mean()}')


def statistical_results():
    zscore_agent = scipy.stats.zscore(agent_completed_tasks)
    zscore_fixed = scipy.stats.zscore(fixed_completed_tasks)

    fixed_ks = scipy.stats.kstest(agent_completed_tasks, 'norm')
    agent_ks = scipy.stats.kstest(fixed_completed_tasks, 'norm')
    print(f'KS - fixed: {fixed_ks}, agent: {agent_ks}')

    f_test = scipy.stats.f_oneway(zscore_agent, zscore_fixed)
    print(f'f test: {f_test}')


if __name__ == "__main__":
    assert len(agent_completed_tasks) == len(agent_failed_tasks) == len(agent_attempted_tasks) == len(fixed_completed_tasks)

    valid_pos = [pos for pos, value in enumerate(fixed_completed_tasks) if value > -1]
    agent_completed_tasks = np.array([value for pos, value in enumerate(agent_completed_tasks) if pos in valid_pos])
    agent_failed_tasks = np.array([value for pos, value in enumerate(agent_failed_tasks) if pos in valid_pos])
    agent_attempted_tasks = np.array([value for pos, value in enumerate(agent_attempted_tasks) if pos in valid_pos])
    fixed_completed_tasks = np.array([value for pos, value in enumerate(fixed_completed_tasks) if pos in valid_pos])

    assert len(agent_completed_tasks) == len(agent_failed_tasks) == len(agent_attempted_tasks) == len(
        fixed_completed_tasks)

    graph_results()
    statistical_results()
