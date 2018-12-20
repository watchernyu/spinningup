# Soft Actor-Critic Pytorch Implementation
Soft Actor-Critic Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This is a minimal, easy-to-learn and well-commented Pytorch implementation, and recommended to be studied along with the OpenAI Spinup Doc. This SAC implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. Target audience of this repo is Pytorch users (especially NYU students) who are learning Soft Actor-Critic algorithm. 

## Setup environment:
To use the code you should first install spinup: 
https://spinningup.openai.com/en/latest/user/installation.html

The Pytorch version used is: 0.4.1, install pytorch:
https://pytorch.org/

If you want to run Mujoco environments, you need to also install Mujoco and get a liscence. For how to install and run Mujoco on NYU's hpc cluster, check out my other tutorial: https://github.com/watchernyu/hpc_setup

## Run experiment
The SAC implementation can be found under `spinup/algos/sac_pytorch/`

Run experiments with pytorch sac: 

In the sac_pytorch folder, run the SAC code with `python sac_pytorch`

Or you can use a spinup experiment grid: a sample grid is given under `spinningup/experiments/`, you can run it with `python sample_grid.py`

The program structure, though in Pytorch has been made to be as close to spinup tensorflow code as possible so readers who are familiar with other algorithm code in spinup will find this one easier to work with. I also referenced rlkit's SAC pytorch implementation, especially for the policy and value models part, but did a lot of simplification. 

## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

## Acknowledgement 
Great thanks to Joshua Achiam, the author of OpenAI Spinning Up. I think the Spinning Up documentation/code is an incredibly good resource for learning DRL and it made my learning much more effective. And also huge thanks for helping me with some Spinup coding issues!

Below are original readme

==================================

**Status:** Active (under active development, breaking changes may occur)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!
