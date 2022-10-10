import os
import argparse
import random
import torch
import numpy as np
from gym_env import make_env
from trainer import train
from misc.utils import load_config, set_log
from tensorboardX import SummaryWriter
from algorithm import get_agent

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Set logging
    if not os.path.exists("./log"):
        os.makedirs("./log")

    log = set_log(args)
    tb_writer = SummaryWriter('./log/tb_{0}'.format(args.log_name))

    # Set env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)

    # Set agents
    agents = []
    for i_agent, agent_type in enumerate(args.agent_types):
        agent = get_agent(
            env=env, log=log, tb_writer=tb_writer,
            args=args, agent_type=agent_type, i_agent=i_agent)
        agents.append(agent)

    # Begin train
    train(agents, env, log, tb_writer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="meta-mapg")

    # Algorithm
    parser.add_argument(
        "--agent-types", nargs='+', default=["", ""],
        help="Agent types (algorithms) to play with")
    parser.add_argument(
        "--actor-lr", type=float, default=0.0005,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", type=float, default=0.001,
        help="Learning rate for critic")
    parser.add_argument(
        "--gain-lr", type=float, default=0.001,
        help="Learning rate for gain")
    parser.add_argument(
        "--inference-lr", type=float, default=0.001,
        help="Learning rate for inference")
    parser.add_argument(
        "--discount", type=float, default=0.99,
        help="Discount factor in reinforcement learning")
    parser.add_argument(
        "--n-hidden", type=int, default=16,
        help="Number of neurons for hidden network")
    parser.add_argument(
        "--n-latent", type=int, default=5,
        help="Number of latent variables")
    parser.add_argument(
        "--max-grad-clip", type=float, default=10.,
        help="Max norm gradient clipping value in optimization")
    parser.add_argument(
        "--kl-weight", type=float, default=0.01,
        help="Weight of KL loss in inference optimization")
    parser.add_argument(
        "--entropy-weight", type=float, default=0.1,
        help="Weight for entropy")
    parser.add_argument(
        "--polyak", type=float, default=0.995,
        help="Soft update for target network")
    parser.add_argument(
        "--max-timestep", type=int, default=int(1e4),
        help="Max timestep to train agents")

    # Replay buffer
    parser.add_argument(
        "--memory-capacity", type=int, default=int(1e4),
        help="Max number of transitions to save in memory")
    parser.add_argument(
        "--sampling-mode", type=str, default="random",
        help="Sampling method for training actor and critic [random, recent, now]")
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of samples to use for each train iteration")

    # Env
    parser.add_argument(
        "--env-name", type=str, default="",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-horizon", type=int, default=150,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n-agent", type=int, default=2,
        help="Number of agents in a shared environment")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config that replaces default params with experiment specific params")

    args = parser.parse_args()

    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = \
        "env::%s_agent_types::%s_seed::%s_actor_lr::%s_critic_lr::%s_gain_lr::%s_inference_lr::%s_" \
        "entropy_weight::%s_n_latent::%s_prefix::%s_log" % (
            args.env_name, args.agent_types, args.seed, args.actor_lr, args.critic_lr, args.gain_lr,
            args.inference_lr, args.entropy_weight, args.n_latent, args.prefix)

    main(args=args)
