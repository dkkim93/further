import abc
import gym
import numpy as np
from misc.rl_utils import to_onehot


class Base(gym.Env, metaclass=abc.ABCMeta):
    """Base class for two agent iterated matrix game
    Possible actions for each agent are (A) and (B)
    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        super(Base, self).__init__()
        assert len(args.agent_types) == 2, "Only two agents are supported in this domain"

        self.args = args

    @abc.abstractmethod
    def _set_payoff_matrix(self, *args, **kwargs):
        pass

    def _action_to_state(self, actions):
        assert actions[0].shape == actions[1].shape
        action0, action1 = actions
        state = 2 * action0 + action1
        return state

    def reset(self):
        self.timestep = 0
        obs = np.zeros(1, dtype=np.int32)
        obs = to_onehot(obs, dim=5)
        return obs

    def step(self, actions):
        # Update timestep
        self.timestep += 1

        # Get observation
        state = self._action_to_state(actions)
        assert state.shape == (1,), "Shape should be (1,)"
        obs = self.states[state]
        obs = to_onehot(obs, dim=5)

        # Get reward
        rewards = []
        for i_agent in range(2):
            rewards.append(self.payoff_matrix[i_agent][state])

        # Get done
        done = True if self.timestep >= self.args.ep_horizon else False

        return obs, rewards, done, {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()
