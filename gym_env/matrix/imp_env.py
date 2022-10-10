import numpy as np
from gym_env.matrix.base import Base
from gym.spaces import Discrete


class IMPEnv(Base):
    """Two agent matching pennies game
    Args:
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, args):
        super(IMPEnv, self).__init__(args)

        self.observation_space = Discrete(5)
        self.states = np.arange(start=1, stop=5, step=1, dtype=np.int32)
        self.action_space = Discrete(2)
        self._set_payoff_matrix()

    def _set_payoff_matrix(self):
        self.payoff_matrix = [
            np.array([+1, -1, -1, +1], dtype=np.float32),
            np.array([-1, +1, +1, -1], dtype=np.float32)]
