import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import get_optimizer, save_checkpoint, load, save



class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.layer1 = nn.Linear(*args.state_dim, args.latent1)
        self.layer2 = nn.Linear(args.latent1, args.latent2)
        self.fc_mu = nn.Linear(args.latent2, args.n_actions)
        self.fc_sigma = nn.Linear(args.latent2, args.n_actions)
        self.to(args.device)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        fc_mu = self.fc_mu(x)
        fc_sigma = self.fc_sigma(x)

        return fc_mu, fc_sigma


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.args = args
        self.layer1 = nn.Linear(*args.state_dim, args.latent1)
        self.layer2 = nn.Linear(args.latent1, args.latent2)
        self.out = nn.Linear(args.latent2, 1)
        self.to(args.device)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        x = self.out(x)

        return x


# Actor Critic Agent Class for training 
class Agent:
    def __init__(self, args, model_name="actor") -> None:
        self.args=args
        print("Initializing Agent...")

        # Initialize actor network
        self.actor = ActorNetwork(self.args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.000005, weight_decay=self.args.wd)
        if self.args.resume:
            self.actor, self.actor_optimizer, _ = self.load_actor(model_name=model_name)


        # Initialize critic network
        self.critic = CriticNetwork(self.args)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00001, weight_decay=self.args.wd)


    # Get action (sample it) from the policy distribution
    def get_action(self, obs) -> None:
        # Guassian Distribution
        mean, sigma  = self.actor(obs)
        action_probs = torch.distributions.Normal(mean, torch.exp(sigma))
        # Sample from the distribution
        probs = action_probs.sample()
        # Log probability of taking the action
        self.log_probs = action_probs.log_prob(probs).to(self.args.device)
        # Limit the action to be within [-1, 1]
        action = torch.tanh(probs)

        return action.detach().cpu().numpy()

    
    def train_step(self, obs, reward, new_obs, done) -> None:

        value = self.critic(obs)
        next_value = self.critic(new_obs)
        delta = torch.tensor(reward) + self.args.gamma * (1 - int(done)) * next_value - value
        
        actor_loss = (-self.log_probs * delta).mean()
        critic_loss = delta**2
        
        loss = actor_loss + critic_loss

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward() 
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def save_actor(self, model_name="actor") -> None:
        save(self.args.results_dir, model=self.actor, epoch=0, optimizer=self.actor_optimizer, model_name=model_name)

    def load_actor(self, model_name="actor") -> None:
        return load(self.actor, optimizer=self.actor_optimizer, resume=self.args.resume, model_name=model_name) 