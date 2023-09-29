# 🚀 Actor-Critic Reinforcement Learning 🎮

This repository contains the implementation of the Actor-Critic reinforcement learning method trained on two different environments: OpenAI Gym's Continuous Car environment and a custom environment created using OpenAI Gym and Pygame. 🤖

## 🔍 Overview

In this project, I have implemented the Actor-Critic reinforcement learning algorithm to train agents in two distinct environments:

1. **Continuous Car Environment (OpenAI Gym)** 🚗
    - In this environment, the agent learns to control a car to navigate a continuous state space.
    - The agent receives observations of the car's position along the x-axis.
    - A negative reward of -0.1 * action2 is received at each timestep to penalize for taking actions of large magnitude. If the mountain car reaches the goal, a positive reward of +100 is added to the negative reward for that timestep.

2. **Custom Environment (Pygame + OpenAI Gym)** 🎮
    - In this custom environment, an agent interacts with an opponent to find and approach it.
    - Observations include the agents' and opponents' positions (x, y).
    - The reward is the negative distance between the agent and the opponent.


## 📁 Repository Structure

Let's navigate this repository like seasoned adventurers:

- `Actor_Critic.py`: Contains the implementation of the Actor-Critic algorithm.
- `actor_critic_mountain_car.py`: Implementation of the OpenAI Gym Continuous Car environment.
- `actor_critic_custom.py`: Implementation of the custom environment with Pygame and OpenAI Gym.
- `utils.py`: Training helper functions.
- `README.md`: You are here!

## Results
### Custom Environment
| Random | Trained 1000 episodes|
| :-------- | :------- |
|![random](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/assets/47561760/1c11fd1d-5c2f-4944-a1c1-7b72ed40eb0d) | ![actor_critic_custom](https://github.com/sobhanshukueian/Reinforcement-Learning-Playground/assets/47561760/d166b1cc-3ee7-4a71-ab6e-5bf22d3fa223) |



