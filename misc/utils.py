import yaml
import logging
import git
import torch
import numpy as np


def load_config(args, path="."):
    """Loads and replaces default parameters with experiment specific parameters
    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    """Sets python logger
    Args:
        logger_name (str): Specifies logging name
        log_file (str): Specifies path to save logging
        level (int): Logging when above specified level. Default: logging.INFO
    """
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    """Sets python logging
    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to get Git repository. Default: "."
    Returns:
        log (dict): Dictionary that contains python logging
    """
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./log/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    # Log git information
    repo = git.Repo(path)
    try:
        log[args.log_name].info("Branch: {}".format(repo.active_branch))
    except TypeError:
        pass
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    # Log device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log[args.log_name].info("Device: {}".format(device))

    return log


def reset_debug(args, debug):
    """Reset debug
    Args:
        args (argparse): Python argparse that contains arguments
        debug (dict): Dictionary that contains debugging topics
    """
    if debug is None:
        debug = {
            "rewards": np.zeros((args.n_agent,)),
            "accumulate_rewards": np.zeros((args.n_agent,))}
    else:
        debug["rewards"] = np.zeros((args.n_agent,))

    return debug


def update_debug(debug, args, **kwargs):
    """Update debug with information
    Args:
        args (argparse): Python argparse that contains arguments
        debug (dict): Dictionary that contains debugging topics
    """
    for key, value in kwargs.items():
        debug["rewards"] += np.array(value).flatten()
        debug["accumulate_rewards"] += np.array(value).flatten()


def log_debug(debug, log, tb_writer, args, timestep):
    """Log training performance
    Args:
        debug (dict): Dictionary that contains debugging topics
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        timestep (int): timestep of training
    """
    for i_agent in range(args.n_agent):
        reward = debug["rewards"][i_agent] / args.ep_horizon
        log[args.log_name].info("Episodic reward: {:.3f} for agent {} at timestep {}".format(
            reward, i_agent, timestep))
        tb_writer.add_scalars("debug/reward", {"agent" + str(i_agent): reward}, timestep)

    if "IMP-v0" in args.env_name:
        for i_agent in range(args.n_agent):
            accumulate_reward = debug["accumulate_rewards"][i_agent]
            log[args.log_name].info("Accumulate reward: {:.3f} for agent {} at timestep {}".format(
                accumulate_reward, i_agent, timestep))
            tb_writer.add_scalars("debug/accumulate_reward", {"agent" + str(i_agent): accumulate_reward}, timestep)
