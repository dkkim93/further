def get_agent(env, log, tb_writer, args, agent_type, i_agent):
    """Load corresponding agent (algorithm)
    Args:
        env (gym env): Gym environment from gym_env folder
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        agent_type (str): Specifies agent's learning algorithm
        i_agent (int): Agent index among the agents in the shared environment
    """
    if agent_type == "further":
        from algorithm.further import FURTHER as Algorithm
    elif agent_type == "lili":
        from algorithm.lili import LILI as Algorithm
    elif agent_type == "tabular_q":
        from algorithm.tabular_q import TabularQ as Algorithm
    else:
        raise ValueError("Not supported algorithm: {}".format(agent_type))

    agent = Algorithm(env=env, log=log, tb_writer=tb_writer, args=args, name=agent_type, i_agent=i_agent)

    return agent
