# Reinforcement Learning Playground 🚀

Hey there! I'm super stoked to kick off a new project in this repository. My plan is to implement a bunch of awesome Reinforcement Learning (RL) algorithms using Python, OpenAI Gym environments, and Pytorch.

![WorldOfChaldeaChaldeaGIF (2)](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/assets/47561760/982589d0-d7a8-4b6f-b4b8-6e2b717ccace)


## Table of Contents
- [Introduction](#introduction)
- [OpenAI Gym](#openai-gym)
- [Algorithms](#algorithms)
  - [DQN (Deep Q-Network)](#dqn-deep-q-network)
  - [Policy Gradient (Implemented✅)](#policy-gradient)
  - [Actor-Critic (Implemented✅)](#actor-critic)
  - [Proximal Policy Optimization (TODO⛔)](#ppo-proximal-policy-optimization)


## Introduction

Reinforcement Learning (RL) is a fascinating field of artificial intelligence where agents learn to make decisions by interacting with their environment. This playground provides an organized collection of popular RL algorithms to help you understand, implement, and compare their performance on the classic OpenAI Gym environment - CartPole.

## OpenAI Gym

OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It provides a variety of environments ranging from classic control tasks to Atari 2600 games. Here are some of the environments we'll be working with:

| CartPole-v1 | Pendulum-v0     | MountainCar-v0                |
| :-------- | :------- | :------------------------- |
|   ![images/mountain car.gif](https://gymnasium.farama.org/_images/cart_pole.gif) |   ![Pendulum-v0](https://gymnasium.farama.org/_images/pendulum.gif) | ![mountain-car](https://gymnasium.farama.org/_images/mountain_car.gif) |


## Algorithms

### Policy Gradient

Policy Gradient methods focus on directly learning the policy and the agent's strategy to make decisions. Implement and experiment with policy gradient algorithms like REINFORCE, A2C (Advantage Actor-Critic), and more.

#### Results

![cartpole](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/assets/47561760/868e2d4c-32bf-4fc5-8fc3-3031cfc05fec)![be227de0-9bf6-4c2e-95db-66e4255a740d](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/assets/47561760/0a1281e4-04b5-45b6-b171-55e78da53b59)




### DQN (Deep Q-Network)

DQN is a fundamental RL algorithm that uses a deep neural network to approximate the Q-value function. Experience the power of Q-learning and deep neural networks in training agents to balance the CartPole.

### Actor-Critic (Check here for my implementation [Actor-Critic](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/tree/main/ActorCritic).)

Actor-critic algorithms combine the benefits of value-based and policy-based methods by maintaining both a policy (the actor) and a value function (the critic). Explore algorithms such as A3C (Asynchronous Advantage Actor-Critic) and A2C to enhance your understanding of actor-critic approaches.

### PPO (Proximal Policy Optimization)

PPO is a state-of-the-art policy optimization algorithm known for its stability and sample efficiency. Dive into the world of PPO and see how it outperforms other policy gradient methods.
