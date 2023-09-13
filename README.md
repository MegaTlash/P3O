## Abstract

The Proximal Policy Optimization (PPO) is a policy gradient approach providing state-of-the-art performance in many domains through the ``surrogate'' objective function using stochastic gradient ascent. While PPO is an appealing approach in reinforcement learning, it does not consider the importance of states (a frequently seen state in a successful trajectory) in policy/value function updates. In this work, we introduce Preferential Proximal Policy Optimization (P3O)  which incorporates the importance of these states into parameter updates. First, we determine the importance of each state based on the variance of the action probabilities given a particular state multiplied by the value function, normalized and smoothed using the Exponentially Weighted Moving Average. Then, we incorporate the state's importance in the surrogate objective function. That is, we redefine value and advantage estimation objectives functions in the PPO approach. Unlike other related approaches, we select the importance of states automatically which can be used for any algorithm utilizing a value function.  Empirical evaluations across six Atari environments demonstrate that our approach significantly outperforms the baseline (vanilla PPO) across different tested environments, highlighting the value of our proposed method in learning complex environments. 



## Installation

- Clone this repo: 

```
git clone https://github.com/MegaTlash/P3O.git
```

- Install Python Dependencies and Pytorch Dependencies from http://pytorch.org

```
pip install -r requirements.txt
```


## To run code (same format as CleanRL https://github.com/vwxyzjn/cleanrl)

To train P3O model run (can change parameters in file or command line):

```
python p3o_envpool_atari.py
```


To train PPO model run (can change parameters in file or command line):

```
python ppo_envpool_atari.py
```


To test models run:

```
python test_model.py
```