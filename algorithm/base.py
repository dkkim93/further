import abc
import torch
import numpy as np
import torch.nn.functional as F
from algorithm.replay_memory import ReplayMemory
from misc.rl_utils import get_env_dim, get_env_action_type
from misc.torch_utils import to_numpy
from torch.distributions import Categorical


class Base(metaclass=abc.ABCMeta):
    """Base class for all algorithms
    Args:
        env (gym env): Gym environment from gym_env folder
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
    """
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(Base, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(i_agent)
        self.i_agent = i_agent

        # Call base functions
        self._set_device()
        self._set_env_dim()
        self._set_action_type()
        self._set_memory()

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def act(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def add_transition(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_loss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    def _set_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_env_dim(self) -> None:
        self.env_n_obs, self.env_n_action = get_env_dim(self.env)
        self.log[self.args.log_name].info("[{}] Env input dim: {}".format(
            self.name, self.env_n_obs))
        self.log[self.args.log_name].info("[{}] Env output dim: {}".format(
            self.name, self.env_n_action))

    def _set_action_type(self) -> None:
        self.is_discrete_action, self.action_dtype = get_env_action_type(self.env)
        self.log[self.args.log_name].info("[{}] Discrete action space: {} with dtype {}".format(
            self.name, self.is_discrete_action, self.action_dtype))

    def _set_memory(self) -> None:
        self.memory = ReplayMemory(self.args)

    def _set_epsilon(self) -> None:
        self.epsilon = float(self.args.epsilon)

    def _select_action(self, logit, action_selection):
        if action_selection == "epsilon":
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.env_n_action, size=(1,))
            else:
                action = torch.argmax(logit).unsqueeze(0)
                action = to_numpy(action, dtype=self.action_dtype)
        elif action_selection == "softmax":
            prob = F.softmax(logit, dim=-1)
            distribution = Categorical(probs=prob)
            action = distribution.sample().flatten()
            action = to_numpy(action, dtype=self.action_dtype)
        elif action_selection == "greedy":
            action = torch.argmax(logit).unsqueeze(0)
            action = to_numpy(action, dtype=self.action_dtype)
        else:
            raise ValueError("Invalid action selection method")

        return action
