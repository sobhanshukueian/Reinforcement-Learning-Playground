# Actor-Critic Reinforcement Learning

This repository contains the implementation of the Actor-Critic reinforcement learning method trained on two different environments: OpenAI Gym's Continuous Car environment and a custom environment created using OpenAI Gym and Pygame.

## Overview

In this project, I have implemented the Actor-Critic reinforcement learning algorithm to train agents in two distinct environments:

1. **Continuous Car Environment (OpenAI Gym)**:
    - In this environment, the agent learns to control a car to navigate a continuous state space.
    - The agent receives observations consisting of the car's position (x, y) and velocity (x_dot, y_dot).
    - The reward is defined as the negative distance between the car and a target point.

2. **Custom Environment (Pygame + OpenAI Gym)**:
    - In this custom environment, an agent interacts with an opponent to find and approach it.
    - Observations include the positions (x, y) of both the agent and the opponent.
    - The reward is defined as the negative distance between the agent and the opponent.

## Repository Structure

Here is an overview of the repository structure:

- `Actor_Critic.py`: Contains the implementation of the Actor-Critic algorithm.
- `actor_critic_mountain_car.py`: Implementation of the OpenAI Gym Continuous Car environment.
- `actor_critic_custom.py`: Implementation of the custom environment with Pygame and OpenAI Gym.
- `utils.py`: Training helper functions.
- `README.md`: You are here!

## Getting Started

### Results