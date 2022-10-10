import gym
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_env_dim(env):
    """Get observation and action dimensions
    Args:
        env (gym env): Gym environment from gym_env folder
    """
    if isinstance(env.observation_space, gym.spaces.Box):
        env_n_obs = env.observation_space.shape[0]
    else:
        env_n_obs = env.observation_space.n

    if isinstance(env.action_space, gym.spaces.Box):
        env_n_action = env.action_space.shape[0]
    else:
        env_n_action = env.action_space.n

    return env_n_obs, env_n_action


def get_env_action_type(env):
    """Get action space type (whether discrete or continuous space)
    Args:
        env (gym env): Gym environment from gym_env folder
    """
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete_action = True
        action_dtype = np.int64
    else:
        is_discrete_action = False
        action_dtype = np.float32

    return is_discrete_action, action_dtype


def to_transition(obs, actions, reward, next_obs, agent, args):
    """Concatenate and transform transition into vector
    Args:
        obs (np.ndarray): Observation
        actions (list): List of np.arrays that represents joint action
        reward (np.ndarray): Reward
        next_obs (np.ndarray): Next observation
        agent (algorithm.*): Agent class in algorithm folder
        args (argparse): Python argparse that contains arguments
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)

    if not isinstance(actions, torch.Tensor):
        actions = np.array(actions, dtype=np.int64).reshape(1, -1)
        actions = torch.tensor(actions, dtype=torch.int64, device=agent.device)
    actions_onehot = [
        to_onehot(actions[..., i_agent], dim=agent.env_n_action)
        for i_agent in range(args.n_agent)]
    actions_onehot = torch.cat(actions_onehot, dim=-1).float()

    if not isinstance(reward, torch.Tensor):
        reward = torch.tensor(reward, dtype=torch.float32, device=agent.device).unsqueeze(1)

    if not isinstance(next_obs, torch.Tensor):
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent.device)

    return torch.cat([obs, actions_onehot, reward, next_obs], dim=-1)


def to_onehot(value, dim):
    """Convert batch of tensor numbers to onehot
    Args:
        value (numpy.ndarray or torch.Tensor): Batch of numbers to convert to onehot
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray or torch.Tensor): Converted onehot
    References:
        https://gist.github.com/NegatioN/acbd8bb6be866ce1831b2d073fd7c450
    """
    if isinstance(value, np.ndarray):
        assert len(value.shape) == 1, "Shape must be (batch,)"
        onehot = np.eye(dim, dtype=np.float32)[value]
        assert onehot.shape == (value.shape[0], dim), "Shape must be: (batch, dim)"
    elif isinstance(value, torch.Tensor):
        scatter_dim = len(value.size())
        y_tensor = value.view(*value.size(), -1)
        zeros = torch.zeros(*value.size(), dim, dtype=value.dtype, device=value.device)
        onehot = zeros.scatter(scatter_dim, y_tensor, 1)
    else:
        raise ValueError("Not supported data type")

    return onehot


def reparameterization(mean, logvar):
    """Perform sampling based on reparameterization
    Args:
        mean (torch.Tensor): Mean of normal distribution
        logvar (torch.Tensor): Log variance of normal distribution
    Returns:
        z (torch.Tensor): Sampled logit based on reparameterization
    References:
        https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
    """
    var = torch.exp(0.5 * logvar)
    distribution = torch.distributions.Normal(mean, var)
    z = distribution.rsample()
    return z
