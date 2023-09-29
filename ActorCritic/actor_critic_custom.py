from gym import Env
from gym import spaces

import argparse
import pygame
import sys
import imageio


import numpy as np
import torch

from actor_critic import Agent
from utils import *

# @title Arguments
parser = argparse.ArgumentParser(description='Actor Critic')
parser.add_argument('--mode', default="train", type=str, help='Mode of the run, whether train or test.')

parser.add_argument('--episodes', default=1000, type=int, metavar='N', help='Number of episodes for training agent.')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', default=0.0001, type=float, help='Weight decay for training optimizer')
parser.add_argument('--seed', default=3, type=int, help='Seed for reproducibility')
parser.add_argument('--model-name', default="actor", type=str, help='Model name for saving model.')
parser.add_argument('--gamma', default=0.99, type=float, metavar='N', help='The discount factor as mentioned in the previous section')
parser.add_argument('--show-freq', default=5, type=int, metavar='N', help='Show Environment Frequency') 

# Model
parser.add_argument("--latent1", default=256, required=False, help="Latent Space Size for first layer of network.")
parser.add_argument("--latent2", default=256, required=False, help="Latent Space Size for second layer of network.")

# Env Properties
parser.add_argument("--state_dim", default=(4,), required=False, help="State Dimension")
parser.add_argument("--n-actions", default=2, required=False, help="Actions Count for each state")


# utils
parser.add_argument('--resume', default="", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

# args = parser.parse_args()  # running in command line
args = parser.parse_args()  # running in ipynb

# set command line arguments here when running in ipynb
if args.save_dir == '':
    args.save_dir = "./ActorCritic/custom"

args.results_dir = args.save_dir

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(args)

        
WHITE = (255, 255, 255)
MAGIC_BLUE = (50, 153, 204)
MAGIC_RED = (255, 51, 102)

class ReachOpponentEnv(Env):
    def __init__(self):
        super(ReachOpponentEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))  # Continuous actions for X and Y movement
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,))  # State: [agent_x, agent_y, opponent_x, opponent_y]
        self.agent_position = np.array([0.0, 0.0])
        self.opponent_position = np.array([8.0, 8.0])
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0

    def reset(self):
        # Reset the environment
        self.agent_position = np.array([np.random.uniform(0, 8), np.random.uniform(0, 8)])
        self.opponent_position = np.array([np.random.uniform(0, 8), np.random.uniform(0, 8)])
        self.current_step = 0
        return self._next_observation()

    def step(self, action):
        # Move the agent based on the action (X, Y movement)
        self.agent_position += action

        # Clip agent's position to be within the environment bounds
        self.agent_position = np.clip(self.agent_position, 0, 10)

        # Compute distance to opponent
        distance_to_opponent = np.linalg.norm(self.agent_position - self.opponent_position)

        # Reward: Negative distance to encourage reaching the opponent
        reward =- distance_to_opponent

        # Check if the agent has reached the opponent
        if distance_to_opponent < 1:
            reward += 1000
            done = True
        else: done = False 

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        # Generate the next observation
        observation = self._next_observation()

        return observation, reward, done, {}

    def _next_observation(self):
        # Concatenate agent's position and opponent's position
        return np.concatenate([self.agent_position, self.opponent_position])


class PygameUI:
    def __init__(self, env):
        self.env = env
        pygame.init()
        self.screen_width, self.screen_height = 500, 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

    def capture_frame(self):
        """Capture the current Pygame screen as an image."""
        data = pygame.surfarray.array3d(pygame.display.get_surface())
        return data
    
    def reset(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

    def render(self, agent_pos, opponent_pos):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # # Fantasy-like background gradient
        self.screen.fill(WHITE)

        # Draw the agent (a star)
        pygame.draw.circle(self.screen, MAGIC_BLUE, (int(agent_pos[0] * 50), int(agent_pos[1] * 50)), 10)
        
        # Draw the opponent (a crescent moon)
        pygame.draw.circle(self.screen, MAGIC_RED, (int(opponent_pos[0] * 50), int(opponent_pos[1] * 50)), 15)

        pygame.display.flip()
        self.clock.tick(30)

def train(args)->None:
    # Set reproducibility
    reproducibility(args.seed)
    # Set results directory
    args.results_dir = prepare_save_dir(args)
    agent = Agent(args)
    env = ReachOpponentEnv()

    best_score = -np.inf
    print("Start Training")
    print(('%20s' * 2) % ('Episode', 'Score'))
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=args.device)
        done = False
        score = 0

        while not done:
            # Sample action from policy distribution
            action = agent.get_action(state)
            # Step the environment
            n_state, reward, done, info = env.step(action)
            score+=reward
            n_state = torch.tensor(n_state, dtype=torch.float32, device=args.device)
            # Train the agent
            agent.train_step(state, reward, n_state, done)
            state = n_state
        
        # Save the best actor model
        if score > best_score:
            agent.save_actor()
            best_score = score
        print(('%20s' * 3) % (episode , "  ", score))
    
    # Save the last actor model
    agent.save_actor(model_name="last")
    print("Best Score: ", best_score)
    return None

#  Test the trained actor model using pygame visualization
def test(args) -> None:
    frames = []
    reproducibility(args.seed)
    agent = Agent(args, "last")
    env = ReachOpponentEnv()
    pygame_ui = PygameUI(env)

    print("Start Training")
    print(('%20s' * 3) % ('Episode' , 'Loss', 'Score'))
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        state = torch.tensor(state).float().to(args.device)
        done = False
        while not done:
            pygame_ui.render(state[:2], state[2:])
            frame = pygame_ui.capture_frame() 
            # print(frame)
            frames.append(frame)
            action = agent.get_action(state)
            n_state, reward, done, info = env.step(action)
            n_state = torch.tensor(n_state).float().to(args.device)
            state = n_state

    pygame.quit()
    imageio.mimsave('./random.gif', frames, duration=0.1)
    return None


if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    else:
        test(args)


