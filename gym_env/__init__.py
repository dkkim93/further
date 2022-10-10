import gym
from gym.envs.registration import register


register(
    id='IBS-v0',
    entry_point='gym_env.matrix.ibs_env:IBSEnv',
    kwargs={'args': None},
    max_episode_steps=150
)


register(
    id='IC-v0',
    entry_point='gym_env.matrix.ic_env:ICEnv',
    kwargs={'args': None},
    max_episode_steps=150
)


register(
    id='IMP-v0',
    entry_point='gym_env.matrix.imp_env:IMPEnv',
    kwargs={'args': None},
    max_episode_steps=150
)


def make_env(args):
    """Load gym environment
    Args:
        args (argparse): Python argparse that contains arguments
    """
    env = gym.make(args.env_name, args=args)
    env._max_episode_steps = args.ep_horizon
    return env
