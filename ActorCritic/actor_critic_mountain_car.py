import gym
import argparse
import numpy as np
import random
import torch

from Actor_Critic import Agent
from utils import *
        
# @title Arguments
parser = argparse.ArgumentParser(description='Actor Critic')
parser.add_argument('--mode', default="train", type=str, help='Mode of the run, whether train or test.')

parser.add_argument('--episodes', default=100, type=int, metavar='N', help='Number of episodes for training agent.')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', default=0.0001, type=float, help='Weight decay for training optimizer')
parser.add_argument('--seed', default=3, type=int, help='Seed for reproducibility')
parser.add_argument('--model-name', default="policy_net", type=str, help='Model name for saving model.')
parser.add_argument('--gamma', default=0.99, type=float, metavar='N', help='The discount factor as mentioned in the previous section')

# Model
parser.add_argument("--latent1", default=256, required=False, help="Latent Space Size for first layer of network.")
parser.add_argument("--latent2", default=256, required=False, help="Latent Space Size for second layer of network.")

# Env Properties
parser.add_argument("--state_dim", default=(2,), required=False, help="State Dimension")
parser.add_argument("--n-actions", default=1, required=False, help="Actions Count for each state")


# utils
parser.add_argument('--resume', default="", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

# args = parser.parse_args()  # running in command line
args = parser.parse_args()  # running in ipynb

# set command line arguments here when running in ipynb
if args.save_dir == '':
    args.save_dir = "./ActorCriticRuns/mountain"

args.results_dir = args.save_dir

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(args)

def train(args)->None:
    
    reproducibility(args.seed)
    args.results_dir = prepare_save_dir(args)
    agent = Agent(args)
    env = gym.make("MountainCarContinuous-v0")

    best_score = -np.inf
    print("Start Training")
    print(('%20s' * 3) % ('Episode', "   ", 'Score'))
    for episode in range(1, args.episodes + 1):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=args.device)
        done = False
        score = 0
        time_steps = 0
        
        while not done:
            time_steps += 1
            # print("State Shape: {}".format(state.size()))
            action = agent.get_action(state)
            n_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score+=reward
            
            n_state = torch.tensor(n_state, dtype=torch.float32, device=args.device)
            agent.train_step(state, reward, n_state, done)
            state = n_state
            done = done or time_steps >= 1000

        if score > best_score:
            agent.save_actor()
            best_score = score

        print(('%20s' * 3) % (episode, "   ", score))

    
    # Save the last actor model
    agent.save_actor(model_name="last")
    print("Best Score: ", best_score)
    return None


#  Test the trained actor model using pygame visualization
def test(args) -> None:
    reproducibility(args.seed)
    args.results_dir = prepare_save_dir(args)
    agent = Agent(args, "actor")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    print("Start Testing")
    for episode in range(1, args.episodes + 1):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=args.device)
        done = False
        while not done:
            action = agent.get_action(state)
            n_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = torch.tensor(n_state, dtype=torch.float32, device=args.device)
    return None

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    else:
        test(args)
