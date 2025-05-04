# RL-practice


This repository documents my journey of studying reinforcement learning (RL) while serving in the military. Due to many limitations including available software and security issues, I use solely Google Collaboratory as the platform and Python as the language. I use PyTorch and gymnasium as libraries for the model and game environment, respectively.

---

## Goals

The primary goal of this project is to gain in-depth knowledge of RL by implementing various algorithms from scratch, debugging through trial and error, and adding skills to enhance the performances. 
The secondary goal is to provide a simple, yet intuitive path to study RL from scratch for any beginners such as me, and sharing & maintaining the insights that I have gained during the project. 
The last goal is to motivate myself by recording the progress of the study. I am mainly using the after dark times from 22:00-24:00 and it is often tough to devote myself into the study after a fatiguing day of work. However, by keeping the record and being able to look back at the progress I made, it will help me take the steady steps to mastering RL.

---

## References

For the fundamental theory, I mainly referrenced Berkeley's CS 285 lectures by Professor Sergey Levine, and for implementing the codes, I gained help from multiple sources including github repos, gen AI, Stack Overflow, etc.

---

## Timeline & Progress

| Date         | Project / Notebook                     | Description & Key Learnings |
|--------------|----------------------------------------|-----------------------------|
| **Apr 14, 2025** | `gymnasium_tutorial/`                  | An introduction to the Gymnasium environment following the tutorial from https://gymnasium.farama.org/ |

| **Apr 14, 2025** | `sarsa_qlearn_comparison.ipynb`        | Implemented SARSA and Q-learning, the most fundamental on-policy and off-policy algorithms. |

| **Apr 15, 2025** | `DQN_cartpole.ipynb`                   | Implemented DQN for cartpole, first time using MLPs for training. Had to refresh basic MLP knowledge/syntax. Understanding how to 
 use MLPs to predict certain functions and how to train it. In the DQN, it predicts the Q value from current state, action and compares it with reward + gamma * max(Q(next_state)). |
 
| **Apr 15, 2025** | `DDQN_cartpole.ipynb`                  | Added a MLP to the DQN to implement DDQN. DQN overestimates the general values of Q function since it takes the max value of the Q vector. Therefore, DDQN adds another MLP Q' that estimates Q. The predicted Q value is estimated using Q'. The new target value is calculated as reward + gamma * Q(next_state, argmax(Q'(next_state))) which doesn't take the max value of the Q estimates. This prevents overestimating the Q value. Empirically found better results compared to the DQN. |

| **Apr 15, 2025** | `Dueling_DQN_cartpole.ipynb`   ✅      | Implemented Dueling DQN which inherits DDQN's structure but adds an advantage stream. The Q vector is calculated as value + advantage - advantage.mean() while the value and advantage is estimated through distinct streams. This separates the benefit of being in a good state and the benefit of taking a good action. In many real life situations, it's the state that maters, not the action. In these cases, the Dueling DQN's network can just flatten the advantage and learn the value, which is a function of the state only. |

| **Apr 16, 2025** | `A2C_bipedal.ipynb` (unfinished)       | First try implementing A2C. Tried it on Bipedal game to experiment a continuous game. A2C is an algorithm that has two seperate players, the Actor and the Critic. The critic's goal is to estimate the best value, similar to the Deep Q networs but caclulating the value function instead. The actor's goal is to optimize the policy function(a function that returns a probability distribution of actions in a given state). The basic A2C computes advantage by subtracting estimated value from gamma decayed reward. Then, the policy updates by increasing the probability of the selected actions if the advantage is positive, and vice-versa otherwise. The critic updates to minimize the MSE(advantage, 0). Couldn't finish debugging the code. |
| **Apr 18, 2025** | `A2C_bipedal.ipynb` (unfinished)       | Switched to n-step A2C(aka. Vanilla A2C). n-step A2C is developed to update more frequently than REINFORCE which has to wait a whole episode to end. n-step A2C computes and stores the advantages, log probs, and values for n steps (or untii the trajectory is done), and refreshes them after updating the actor and critic. The model wasn't getting trained at all. Printed out multiple statistics such as values, returns, entropies for debugging. Found out gradient was to big and applied gradient clipping. Also added normalization layers. Couldn't get it working. |
| **Apr 19, 2025** | `A2C_bipedal.ipynb` (unfinished)       | Referrenced RL Baselines3 Zoo for hyperparameters. |
| **May 3, 2025**  | `A2C_cartpole.ipynb`, `RL_A2C.ipynb`   | Reinforced A2C structure; added GAE-style advantages. Investigated variance vs bias trade-off in actor-critic methods. |





