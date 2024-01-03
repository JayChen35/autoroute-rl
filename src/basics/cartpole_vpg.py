import argparse
import os, datetime, random
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# Vanilla policy gradient on the cartpole environment, with simple value function
class Agent(nn.Sequential):
    def __init__(self, env: gym.Env):
        # For CartPole, the observation space is a (4)-dimension tuple such that:
        # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
        w = args.dense_width
        super(Agent, self).__init__(
            layer_init(nn.Linear(env.observation_space.shape[0], w)),
            nn.Tanh(),
            layer_init(nn.Linear(w, w)),
            nn.Tanh(),
            layer_init(nn.Linear(w, env.action_space.n), std=1.0)
        )        
        
    # Get the current parameterized policy (pi_theta)
    def get_pi(self, obs: Tensor):
        logits = self.__call__(obs)
        return torch.distributions.Categorical(logits=logits)
    
    # Sample an action with the current policy. 
    def get_action(self, obs: Tensor):
        return self.get_pi(obs).sample().item()
    
    # Compute loss. For the right inputs, loss = policy gradient.
    def compute_loss(self, obs: Tensor, actions: Tensor, weights: Tensor):
        log_p = self.get_pi(obs).log_prob(actions)
        return -(log_p * weights).mean()
        
        
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def train_single_epoch(args, env, agent: Agent, optim):
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # Reset episode-specific variables
    obs, _ = env.reset()                    # first obs comes from starting distribution
    terminated, truncated = False, False    # signal from environment that episode is over
    ep_rews = []                            # list for rewards accrued throughout episode
    
    # Policy rollout, i.e. collect experience with the current policy
    while True:
        # Agent will act based on current observation, and get an associated reward
        batch_obs.append(obs.copy())
        action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        batch_acts.append(action)
        ep_rews.append(reward)
        # [observation, action, reward] for this step in the episode is collected
        
        if terminated or truncated:
            # Now, since the episode has ended, consolidate episode-specific values
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            
            # We now want the weight for each logprob(a|s), which is R(tau). Say the current
            # episode lasted n steps, and thus there are n observations/actions in batch_obs
            # and batch_acts. We therefore will need to assign n weights to each logprob(a|s):
            batch_weights += [ep_ret] * ep_len
            # This gives each (a|s) a weight equal to the total return at the end of the episode.
            
            # Reset episode-specific variables since the episode has ended
            obs, _ = env.reset()
            terminated, truncated, ep_rews = False, False, []
                            
            # End experience loop if we have enough of it (batch size reached)
            if len(batch_obs) > args.batch_size:
                break 
            
    # Now that we've collected `batch_size` worth of [obs, act, weight] triplets, update policy
    optim.zero_grad()
    batch_loss = agent.compute_loss(
        obs=torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32),
        actions=torch.as_tensor(np.asarray(batch_acts), dtype=torch.int32),
        weights=torch.as_tensor(np.asarray(batch_weights), dtype=torch.float32)
    )
    batch_loss.backward()
    optim.step()
    return batch_loss, batch_rets, batch_lens
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_id', type=str, default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument('-r', '--learning_rate', type=float, default=5e-3,
                        help="learning rate of the optimizer")
    parser.add_argument('-d', '--dense_width', type=int, default=64,
                        help="width of the largest dense layer in our agent")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed")
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help="total epochs of training (each with `batch_size` state/action pairs)")
    parser.add_argument('-b', '--batch_size', type=int, default=5000,
                        help="batch size, i.e. number of observations we make per policy update")
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True,
                        nargs='?', const=True, help="sets `torch.backends.cudnn.deterministic`")
    parser.add_argument('-c', '--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?',
                        const=True, help="toggles whether or not cuda is enabled by default")
    parser.add_argument('-v', '--video', type=lambda x: bool(strtobool(x)), default=False, nargs='?',
                        const=True, help="sets whether to record video of the environment")
    return parser.parse_args()

    
if __name__ == "__main__":
    # Hyperparameter and Tensorboard setup
    args = parse_args()
    dt_str = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    run_name = f"{args.gym_id}__{args.seed}__{dt_str}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"{key}|{value}" for key, value in vars(args).items()]))
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # Setting up the environment
    env = gym.make(args.gym_id, render_mode="rgb_array")
    if args.video:
        # One episode = time between when the env returns terminated or truncated
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", step_trigger=lambda t: t%50000==0)
    
    # Create the agent
    agent = Agent(env)  
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
    for i in range(args.epochs):
        batch_loss, batch_rets, batch_lens = train_single_epoch(args, env, agent, optimizer)
        print('Epoch: %3d | Loss: %.3f | Average Return: %.3f | Average Episode Length: %.3f'% 
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        writer.add_scalar("batch_loss", batch_loss, global_step=i)
        writer.add_scalar("avg_batch_return", np.mean(batch_rets), global_step=i)
        writer.add_scalar("avg_batch_ep_len", np.mean(batch_lens), global_step=i)
