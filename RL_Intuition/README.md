# RL_Intuition
Reinforcement Learning

<br/>

## ⮚ Features-
### ⮩ Description:
| Equation Name | Equation | Attributes |
| :---: | :---: | :---: |
| Bellman Equation | ![BellmanEquation](https://latex.codecogs.com/svg.latex?V%28s%29%20%3D%20max_%7Ba%7D%28R%28s%2C%20a%29%20&plus;%20%5Cgamma%20V%28%7Bs%7D%27%29%29) | ![BellmanAttributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20V%5C%21%5C%21%3A%20value%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20action%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20factor%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state) |
| Markov Decision Process | ![MarkovEquation](https://latex.codecogs.com/svg.latex?V%28s%29%20%3D%20max_%7Ba%7D%28R%28s%2C%20a%29%20&plus;%20%5Cgamma%20%5Csum_%7B%7Bs%7D%27%7D%20%28P%28s%2C%20a%2C%20%7Bs%7D%27%29%20*%20V%28%7Bs%7D%27%29%29%29%20%5C%3B) | ![MarkovAttributes](https://latex.codecogs.com/svg.latex?%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%5C%21%20V%5C%21%5C%21%3A%20value%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20R%5C%21%5C%21%3A%20reward%2C%20%5Cnewline%20a%5C%21%5C%21%3A%20action%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5Cgamma%5C%21%5C%21%3A%20discounting%20factor%2C%20%5Cnewline%20s%5C%21%5C%21%3A%20current%20%5C%2C%20state%2C%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20%7Bs%7D%27%5C%21%5C%21%3A%20next%20%5C%2C%20state%2C%20%5Cnewline%20P%5C%21%5C%21%3A%20probability) |


### ⮩ References:
#### ➞ Papers:
* [Reinforcement Learning I: Introduction](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=947C1C78AE35225DD1928F05EB6F010B?doi=10.1.1.32.7692&rep=rep1&type=pdf)
* [The Theory of Dynamic Programming](https://www.rand.org/pubs/papers/P550.html)
* [A Survey of Applications of Markov Decision Processes](http://www.it.uu.se/edu/course/homepage/aism/st11/MDPApplications3.pdf)

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
