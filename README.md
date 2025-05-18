# RL-practice

This repository documents my journey of studying reinforcement learning (RL) while serving in the military. Due to many limitations including available software and security issues, I use solely Google Collaboratory as the platform and Python as the language. I use PyTorch and Gymnasium as libraries for the model and game environment, respectively.

---

## Goals

The primary goal of this project is to gain in-depth knowledge of RL by implementing various algorithms from scratch, debugging through trial and error, and adding skills to enhance the performances.  
The secondary goal is to provide a simple, yet intuitive path to study RL from scratch for any beginners like me, and to share & maintain the insights I have gained during the project.  
The last goal is to motivate myself by recording the progress of my study. I am mainly using the late-night hours from 22:00–24:00, and it is often tough to devote myself to studying after a fatiguing day of work. By recording my progress here and looking back at the progress I made, it helps me take steady steps toward mastering RL.

---

## Algorithms Covered

- Q-Learning (Tabular)
- SARSA
- DQN
- Double DQN
- Dueling DQN
- A2C (Monte Carlo and n-step variants)
- Generalized Advantage Estimation (GAE)
- Vectorized Environments
  
---

## References

For the fundamental theory, I mainly referenced Berkeley's CS 285 lectures by Professor Sergey Levine. For implementing the code, I gained help from multiple sources including GitHub repos, generative AI, Stack Overflow, etc.

---

## Timeline & Progress

| Date             | Project / Notebook                  | Description & Key Learnings |
|--------------|----------------------------------------|-----------------------------|
| **Apr 14, 2025** | `gymnasium_tutorial/`  ✅              | An introduction to the Gymnasium environment following the tutorial from https://gymnasium.farama.org/ |
| **Apr 14, 2025** | `sarsa_qlearn_comparison.ipynb`  ✅    | Implemented SARSA and Q-learning, the most fundamental on-policy and off-policy algorithms. |
| **Apr 15, 2025** | `DQN_cartpole.ipynb`  ✅               | Implemented DQN for CartPole, first time using MLPs for training. Had to refresh basic MLP knowledge/syntax. Learned how to use MLPs to predict certain functions and how to train them. In the DQN, it predicts the Q value from the current state and action and compares it with reward + gamma * max(Q(next_state)). |
| **Apr 15, 2025** | `DDQN_cartpole.ipynb`  ✅              | Added an MLP to the DQN to implement DDQN. DQN overestimates the general values of the Q function since it takes the max value of the Q vector. Therefore, DDQN adds another MLP Q' that estimates Q. The predicted Q value is estimated using Q'. The new target value is calculated as reward + gamma * Q(next_state, argmax(Q'(next_state))), which doesn't take the max value directly. This prevents overestimation of the Q value. Empirically found better results compared to DQN. |
| **Apr 15, 2025** | `Dueling_DQN_cartpole.ipynb`   ✅      | Implemented Dueling DQN which inherits DDQN's structure but adds an advantage stream. The Q vector is calculated as value + advantage - advantage.mean(), while the value and advantage are estimated through distinct streams. This separates the benefit of being in a good state from the benefit of taking a good action. In many real-life situations, it's the state that matters, not the action. In these cases, Dueling DQN's network can just flatten the advantage and learn the value, which is a function of the state only. |
| **Apr 16, 2025** | `A2C_bipedal.ipynb`  🛠️                | First try implementing Monte Carlo A2C. Tried it on BipedalWalker to experiment with a continuous environment. A2C is an algorithm that has two separate players, the Actor and the Critic. The Critic's goal is to estimate the value function, similar to Deep Q Networks, but estimating value instead. The Actor's goal is to optimize the policy function (a function that returns a probability distribution over actions in a given state). The basic A2C computes advantage by subtracting the estimated value from gamma-discounted reward. Then, the policy updates by increasing the probability of the selected actions if the advantage is positive, and vice versa otherwise. The Critic updates to minimize MSE(advantage, 0). Couldn't finish debugging the code. |
| **Apr 18, 2025** | `A2C_bipedal.ipynb` 🛠️                 | Switched to n-step A2C (aka Vanilla A2C). n-step A2C is developed to update more frequently than Monte Carlo A2C, which has to wait for the whole episode to end. n-step A2C computes and stores the advantages, log probs, and values for n steps (or until the trajectory ends), and refreshes them after updating the Actor and Critic. The model wasn't training at all. Printed out multiple statistics such as values, returns, entropies for debugging. Found out the gradient was too big and applied gradient clipping. Also added normalization layers. Still couldn’t get it working. |
| **Apr 19, 2025** | `A2C_bipedal.ipynb` 🛠️                 | Tried using RL Baselines3 Zoo's hyperparameters and implemented Generalized Advantage Estimation (GAE). Still wouldn’t work. |
| **Apr 20, 2025** | `A2C_cartpole.ipynb`, `RL_A2C.ipynb` 🛠️ | Came back to CartPole to try implementing n-step A2C in a simpler environment. Downloaded the following Jupyter Notebook and succeeded in training ([source](https://github.com/ayeenp/deep-rl-a2c-cartpole/blob/main/RL_A2C.ipynb)). Made sure that the structure of the MLP and hyperparameters were not the problem. |
| **Apr 24–May 1, 2025** | Vacation!! | Visited Tokyo to meet up with T@B friends, came back to Seoul and celebrated my brother's birthday. |
| **May 3, 2025** | `A2C_cartpole.ipynb`, `RL_A2C.ipynb`  ✅ | Thought n-step was complicating the problem and started from the beginning by implementing Monte Carlo A2C. Finally got it to work. Then, tried n-step A2C again. Using a bigger n than max_step makes n-step A2C behave equally to Monte Carlo A2C. However, my model did not train correctly when using the n-step methods instead. Though, this helped my pinpoint that the issue is in the method compute_n_step_returns and fount the critical bug: forgetting to mask the bootstrapped value with (1 - done) caused the agent to learn from terminal states incorrectly. After fixing this, got both n-step and Monte Carlo A2C working. |
| **May 5, 2025** | `A2C_bipedal.ipynb` 🛠️ | Re-implemented n-step A2C in a continuous space using 'A2C_cartpole's code. Though it would work since the bug was fixed, but did not train well. Switched environment in to LunarLanderContinuous for easier train. Then, bounded action by using tanh function, and initialized std with a smaller value. Still didn't train well enough. |
| **May 6, 2025** | `A2C_lunar_lander.ipynb` 🛠️ | Made a new file to treat Lunar Lander in both discrete and continuous. For discrete Lunar Lander, I expected it to train well when using the same code for cartpole. However, it didn't, and printed out multiple parameters to find the problem. First, the gradients were too big since the calculated returns were all very large due to the crash (-100 reward). Therefore, implemented advantage normalization. However, this caused a problem of vanishing actor_loss since log_prob is pretty much uniform when initialized, and when taken dot product with the normlized advantage vector, the result should mathematically converge to 0. Lastly, tried normalizing n-step return instead of advantage. Showed it is training, but not efficiently. |
| **May 12, 2025** | `A2C_cartpole_vec_env.ipynb` 🛠️ | Practicing/analyzing usage of vectorized environments for faster training. Used A2C_cartpole file to execute. Found some interesting results. First, using multiple environments (8) took much more time than a single environment even when training same episodes on each environment. 00:30 to train 1000 episodes in single env, 12:31 to train 1000 episodes in 8 envs. Second, when same number of total episodes (num_envs * episodes) were explored, multiple environments showed much better reward. When 1000 episodes trained in single env, average reward was 119.2 and when 125 episodes were trained in 8 environments, average reward was 500. When only 10 episodes were trained each in 8 environments, average reward was 119.3.| 
| **May 14, 2025** | `A2C_cartpole_vec_env.ipynb` 🛠️ | Found bug that every single episode (even the first training episodes) explored max_steps for vectorized environments which accounted for longer training time, better result. It was because np.all(done) always returned False since done was not permanently True after state being terminated. Resolved using done_mask vector. | 
| **May 15, 2025** | `A2C_cartpole_vec_env.ipynb` ✅ | Successfully trained using multiple environments. Multiple environments had both significant time/reward advantage over single environment. Compared async/sync vectorization methods and async trained better, sync trained faster. This was unexpected, and more analyzed in the file's comments. | 
| **May 16, 2025** | `A2C_lunar_lander.ipynb` 🛠️ | Found this repo -- https://github.com/nikhilbarhate99/Actor-Critic-PyTorch/blob/master/train.py. The algorithm was mostly alike except some futile differences, except the way loss was calculated. The refernce model took a stepwise sum of logprob * advantage, and my model took the mean of the whole batch. Researched difference between sum vs. mean reduction. Besides magnitude of gradient, not much varies. | 
| **May 17, 2025** | `A2C_lunar_lander.ipynb` 🛠️ | Found critical bug in my calculate_rewards method. The dimensions of the tensors logprobs, state_values, and rewards were all different --0 [n,1], [n, 1, 1], [n]. This led to a very weird way of broadcasting and messed up the loss completely. Learned two lessons - be careful with dimensions of tensors, don't ignore the warnings the cell outputs are telling you. | 
| **May 18, 2025** | `A2C_lunar_lander.ipynb` 'A2C_bipedal.ipynb' 🛠️ | Ran the lunar_lander code after fixing the critical bugs. Took hour and half training sync & async, then found out I was updating every n steps but calculating returns in monte carlo. Will try a gain tomorrow with MC first. Also, polished code for continuous environments in 'A2C_bipedal.ipynb. | 
