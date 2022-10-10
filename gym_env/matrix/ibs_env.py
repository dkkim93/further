import numpy as np
from gym_env.matrix.base import Base
from gym.spaces import Discrete


class IBSEnv(Base):
    """Two agent Bach or Stravinsky game
    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        super(IBSEnv, self).__init__(args)

        self.observation_space = Discrete(5)
        self.states = np.arange(start=1, stop=5, step=1, dtype=np.int32)
        self.action_space = Discrete(2)
        self._set_payoff_matrix()

    def _set_payoff_matrix(self):
        self.payoff_matrix = [
            np.array([+2, +0, +0, +1], dtype=np.float32),
            np.array([+1, +0, +0, +2], dtype=np.float32)]
