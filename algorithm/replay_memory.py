import torch
import numpy as np
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    """Simple replay memory that contains trajectories for each task
    in a Markov chain
    Args:
        args (argparse): Python argparse that contains arguments
    Refs:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, args):
        self.memory = deque([], maxlen=args.memory_capacity)
        self.transition = namedtuple(
            'transition', ("obs", "peer_latent", "actions", "reward", "next_obs", "next_peer_latent"))
        self.args = args

    def __len__(self):
        return len(self.memory)

    def push(self, obs, peer_latent, actions, reward, next_obs, next_peer_latent):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        peer_latent = peer_latent.clone().detach()
        actions = torch.tensor(np.array(actions, dtype=np.int64).reshape(1, -1), dtype=torch.int64, device=device)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward, dtype=np.float32)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(1, -1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        next_peer_latent = next_peer_latent.clone().detach()
        self.memory.append(self.transition(*(obs, peer_latent, actions, reward, next_obs, next_peer_latent)))

    def sample(self, mode):
        if mode == "random":
            indices = np.random.randint(0, len(self), size=self.args.batch_size)
        elif mode == "all":
            if len(self) < self.args.batch_size:
                indices = np.arange(len(self))
            else:
                indices = np.arange(len(self) - self.args.batch_size, len(self))
        elif mode == "recent":
            indices = [len(self) - 1]
        else:
            raise ValueError("Invalid sampling mode")

        batch = self.transition(*zip(*[self.memory[index] for index in indices]))
        obs = torch.cat(batch.obs, dim=0)
        peer_latent = torch.cat(batch.peer_latent, dim=0)
        actions = torch.cat(batch.actions, dim=0)
        reward = torch.cat(batch.reward, dim=0)
        next_obs = torch.cat(batch.next_obs, dim=0)
        next_peer_latent = torch.cat(batch.next_peer_latent, dim=0)

        return obs, peer_latent, actions, reward, next_obs, next_peer_latent
