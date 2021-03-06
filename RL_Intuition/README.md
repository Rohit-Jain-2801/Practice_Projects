# RL_Intuition
Reinforcement Learning

<br/>

## ⮚ Features-
### ⮩ Description:
| Process / Equation Name | Equation | Attributes |
| :---: | :---: | :---: |
| Bellman Equation <br/> (Deterministic) | ![BellmanEquation](https://latex.codecogs.com/svg.latex?V%28s%29%20%3D%20%5Cunderset%7Ba%7D%7Bmax%7D%20%28R%28s%2C%20a%29%20&plus;%20%5Cgamma%20V%28%7Bs%7D%27%29%29) | ![BellmanAttributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20V%5C%21%5C%21%3A%20value%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20action%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20factor%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state%20) |
| Markov Decision Process <br/> (Non-Deterministic) | ![MarkovEquation](https://latex.codecogs.com/svg.latex?V%28s%29%20%3D%20%5Cunderset%7Ba%7D%7Bmax%7D%20%28R%28s%2C%20a%29%20&plus;%20%5Cgamma%20%5Csum_%7B%7Bs%7D%27%7D%20%28P%28s%2C%20a%2C%20%7Bs%7D%27%29%20*%20V%28%7Bs%7D%27%29%29%29) | ![MarkovAttributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20V%5C%21%5C%21%3A%20value%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20action%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20factor%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state%2C%20%5Cnewline%20P%5C%21%5C%21%3A%20probability) |
| Q-Learning <br/> (Non-Deterministic) | ![Q-Equation](https://latex.codecogs.com/gif.latex?Q%28s%2C%20a%29%20%3D%20R%28s%2C%20a%29%20&plus;%20%5Cgamma%20%5Csum_%7B%7Bs%7D%27%7D%20%28P%28s%2C%20a%2C%20%7Bs%7D%27%29%20*%20%5Cunderset%7B%7Ba%7D%27%7D%7Bmax%7D%20%28Q%28%7Bs%7D%27%2C%20%7Ba%7D%27%29%29%29) | ![Q-Attributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20Q%5C%21%5C%21%3A%20action%20%5C%2C%20value%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20current%20%5C%2C%20action%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%7Ba%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20action%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state%2C%20%5Cnewline%20P%5C%21%5C%21%3A%20probability%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20%5C%2C%20factor%20%5Cnewline%20) |
| Temporal Difference Learning <br/> (Deterministic) | ![TD-Equation](https://latex.codecogs.com/svg.latex?Q_%7Bt%7D%28s%2C%20a%29%20%3D%20Q_%7Bt-1%7D%28s%2C%20a%29%20&plus;%20%5Calpha%20%28%5Cunderset%7BTD_%7Bt%7D%28a%2C%20s%29%7D%7B%5Cunderbrace%7BR%28s%2C%20a%29%20&plus;%20%5Cgamma%20*%20%5Cunderset%7B%7Ba%7D%27%7D%7Bmax%7DQ_%7Bt%7D%28%7Bs%7D%27%2C%20%7Ba%7D%27%29%20-%20Q_%7Bt-1%7D%28s%2C%20a%29%7D%7D%29) | ![TD-Attributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20TD%5C%21%5C%21%3A%20temporal%20%5C%2C%20difference%2C%20%5Cnewline%20Q%5C%21%5C%21%3A%20action%20%5C%2C%20value%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20current%20%5C%2C%20action%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%7Ba%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20action%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state%2C%20%5Cnewline%20%5Calpha%5C%21%5C%21%3A%20learning%20%5C%2C%20rate%2C%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20%5C%2C%20factor) |


### ⮩ References:
#### ➞ Papers:
* [Reinforcement Learning I: Introduction](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=947C1C78AE35225DD1928F05EB6F010B?doi=10.1.1.32.7692&rep=rep1&type=pdf)
* [The Theory of Dynamic Programming](https://www.rand.org/pubs/papers/P550.html)
* [A Survey of Applications of Markov Decision Processes](http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications3.pdf)
* [Markov Decision Processes: Concepts and Algorithms](https://www.cs.vu.nl/~annette/SIKS2009/material/SIKS-RLIntro.pdf)
* [Learning to predict by the methods of temporal differences](https://link.springer.com/article/10.1007/BF00115009)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Adaptive ε-greedy Exploration in Reinforcement Learning Based on Value Differences](http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf)

#### ➞ Blogs & Docs:
* [The Complete Reinforcement Learning Dictionary](https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e)
* [Reinforcement Learning - Medium.com - Mohammad Ashraf](https://medium.com/@m.elsersy96)
* [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
* [Reinforcement learning: Temporal-Difference, SARSA, Q-Learning & Expected SARSA in python](https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e)
* [Introduction to Various Reinforcement Learning Algorithms. Part I (Q-Learning, SARSA, DQN, DDPG)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)
* [Introduction to Various Reinforcement Learning Algorithms. Part II (TRPO, PPO)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-part-ii-trpo-ppo-87f2c5919bb9)
* [YouTube.com - Reinforcement Learning - Edureka (Escape_Room)](https://www.youtube.com/watch?v=LzaWrmKL1Z4)
* [YouTube.com - Reinforcement Learning - Deeplizard (FrozenLake-v0, CartPole-v0)](https://www.youtube.com/playlist?list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv)
* [Reinforcement Learning - GitHub.com - DennyBritz](https://github.com/dennybritz/reinforcement-learning)
