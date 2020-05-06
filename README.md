# Online Flexible Resource Allocation

Project is the Dissertation work of Mark Towers for the University of Southampton on 
Online Flexible Resource Allocation in Mobile Edge Computing. The research can be 
found in the final_report folder while the code can be found in src and tests folder. 

> Mobile Edge clouds enable computational tasks to be completed at the edge of the network, without relying on access to
remote data centres. A key challenge in these settings is that servers have limited computational resources that often
need to be allocated to many self-interested users. Existing resource allocation approaches usually assume that tasks
have inelastic resource requirements (i.e., a fixed amount of computation, bandwidth and storage), which may result in
inefficient resource use and even bottlenecks. In this project, an elastic resource requirement mechanism is expanded
upon to an online setting, such that tasks arrive over time with the prices and resource allocation determined by
agents trained using reinforcement learning.

Within the project, a reinforcement learning environment is developed using the OpenAI
gym environment specification in src/env. Numerous reinforcement learning algorithms are
also implemented using Tensorflow 2: DQN, Double DQN, Dueling DQN, DDPG, TD3 and Seq2seq DDPG. 

