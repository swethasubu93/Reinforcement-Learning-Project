# Lunar Landing with Reinforcement Learning

### Overview
This is an implementation of a lunar landing experiment using reinforcement learning. The goal of the project is to train an agent to autonomously land a spacecraft on the lunar surface by learning from its environment through reinforcement learning techniques.

### What is Reinforcement learning?
![1__JJT1mCIcRXy1G8rtPstwA](https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/5ec5cabe-0120-43a6-b933-d5b602d3fd23)
Reinforcement learning (RL) is a type of machine learning where an agent learns to make optimal decisions through trial-and-error interactions with an environment. The agent receives feedback in the form of rewards and penalties, and it learns to choose actions that maximize its cumulative reward.

### Features
- **Reinforcement Learning Algorithm:** Utilizes Deep Q-Learning algorithm for training the lunar landing agent. 
- **Environment Simulation:** The gym package provides a realistic lunar landing environment in which the agent can learn based on rewards.  

### Deep Q-Learning Network (DQN)
![1_uLtcNBImDEo1qcUaOK5niw](https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/bad119c9-bec8-4169-b5bd-718cedae22a3)
The important components of DQN are:
- **Experience Replay Buffer:** A memory store that holds past experiences (state, action, reward, next state) encountered by the agent  during its interactions with the environment. This buffer is crucial for DQN as it allows the agent to learn from its past experiences and break the correlation between consecutive samples, leading to more stable and efficient learning.
- **Q-Network (Neural Network):** A neural network that approximates the Q-function, denoted as Q(s, a), where "s" is the state and "a" is the action and estimates the Q-value, the expected future reward for taking a particular action in a given state. The network takes the current state of the environment as input and outputs a vector of Q-values, one for each possible action. The agent chooses the action with the highest Q-value, indicating the action that is expected to lead to the highest cumulative reward.
- **Target Q-Network:** It is a copy of the Q-network that is used to stabilize the learning process. This network is updated less frequently than the Q-network, typically every few updates of the Q-network. This slower update rate helps to smooth out the changes in the Q-values and reduces oscillations in the training process.
- **Loss Function:** The loss function quantifies the difference between the predicted Q-values (from Q-Network) and the target Q-values (from Target Q-Network). The loss function is used to update the weights of the Q-network during training. The goal is to minimize this loss during training.
- **Epsilon-Greedy Exploration:** Also known exploration-exploitation tradeoff, this is a fundamental challenge in reinforcement learning. Exploration refers to taking actions that are not yet well-known to gather more information, while exploitation refers to taking actions that are known to be good to maximize immediate rewards. With probability ε, the agent selects a random action (exploration), and with probability 1-ε, it selects the action with the highest Q-value (exploitation).
- **Optimization Algorithm:** An optimization algorithm, such as stochastic gradient descent (SGD) or variants like Adam, used to minimize the loss and update the Q-network parameters. In this case, I have used Adam optimizer.
- **Hyperparameters:**
  - *Seed:* The random seed that helps maintain reproducibility in the experiment.
  - *State size:* It is the number of features or dimensions that represent the current state of the environment. It provides the agent with information about the environment's configuration and helps it make informed decisions. 
  - *Action size:* It denotes the distinct number of actions the agent can take in a given state in the environment. 
  - *Number of Episodes:* It determines the overall length of the training process, i.e. total number of times the agent will interact with the environment during training. A higher number of training episodes leads to better performance, but it also comes at the cost of increased training time and computational resources.
  - *Steps per Episode:* It controls the number of actions the agent can take before the episode ends and a new one begins. This parameter is often used to prevent the agent from getting stuck in loops or exploring overly long sequences of actions.
  - *Buffer size:* This determines the amount of memory allocated to Experience Replay Buffer. 
  - *Batch size:* This determines the size of each training batch. 
  - *Gamma (Discount):* A hyperparameter that determines the weight given to future rewards. The discount factor is typically a value between 0 and 1, with higher values indicating a greater emphasis on long-term rewards. The discount factor plays a crucial role in shaping the agent's decision-making and its ability to plan for long-term goals.
  - *Learning rate:* It determines the magnitude of weight updates during training. It controls how quickly the DQN learns from its experiences and adapts to changes in the environment. A high learning rate can lead to faster convergence but also increased instability whereas a low learning rate can lead to slower convergence but more stable and consistent learning.
  - *Epsilon (ε) values:* There are three epsilon parameters - epsilon_start, epsilon_end and epsilon_decay. The epsilon_start determines the initial probability of taking random actions during training (higher epsilon_start lead to more explorataion). The epsilon_end represents the lower bound for the epsilon-greedy exploration policy, i.e., minimum value of epsilon (ensures that agent still occasionally takes random actions during exploitation, preventing the agent from becoming overly fixed on its current policy). The epsilon_decay controls the rate at which epsilon decreases over time - determining how quickly the agent transitions from exploration to exploitation (higher eps_decay value leads to a slower decrease in epsilon, allowing for more exploration throughout the training process. Conversely, a lower eps_decay value results in a faster decline of epsilon, leading to more exploitation and reliance on the learned policy).
  - *Target Q-Value Update Rate (Tau):* The target Q-network parameters are updated periodically (soft update) to the parameters of the Q-network at this rate. This stability-enhancing technique helps in more stable and efficient training.
  - *Network update frequency (Update_Every):* This defines the frequency with which the target Q-network is updated with the weights of the local Q-network.
Network update frequency (Update_every) determines how often the update occurs, while Target Q-Value Update Rate (Tau) determines the smoothness and rate of the update. A balance between these parameters is essential to ensure stable and efficient learning.
The DQN algorithm is built such that if average of last 100 scores (score is total reward from each episode) is above a certain value then training stops and this means agent has solved the environment.

### Experiment architecture
I have trained two agents in the lunar environment, with difference only in three hyperparameters - batch size, learning rate and seed. 
The common hyperparameters are:
- Action size = 4 (based on environment - lunar)
- State size = 8 (based on environment - lunar)
- Number of training epiodes = 1000
- Number of steps per episode = 1000
- epsilon_start = 1
- epsilon_end = 0.01
- epsilon_decay = 0.995
- Average of last 100 scores >= 200 (score limit to solve environment)

The varying parameters are:
**Agent 1**
- batch size = 64
- learning rate = 5e-4
- seed = 0
**Agent 2**
- batch size = 32
- learning rate = 1e-4
- seed = 1

### Results
**Agent 1**
With the given hyperparameters, the agent solved the environment within 565 episodes with average score of 201. The learning progress graph for Agent 1 is shown below. 
![Agent1](https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/eb5fc8ff-c1af-44bf-aa49-7775f78481e1)

Below is the video of lunar landing with Agent 1.

https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/f13f7389-8ad4-4147-a031-b8cc4d74ec89

**Agent 2**
With the given hyperparameters, the agent could not solve the environment and the average score was still below 0, even at 1000th episode.The learning progress graph for Agent 2 is shown below.
![Agent2](https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/9c8c8d12-8c63-4165-b00f-1cff139a5e84)

Below is the video of lunar landing with Agent 2.

https://github.com/swethasubu93/Reinforcement-Learning-Project/assets/109064336/c8997fb0-9632-4e42-92f8-55642b53f0ef

It can be observed that average score of Agent 2 was not decreasing after a point. This could be due to the decrease in learning rate value and the change of seed value when compared to Agent 1.

