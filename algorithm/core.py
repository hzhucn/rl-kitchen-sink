"""
core.py

Defines class switcher, for picking the appropriate learning algorithm.
"""
from algorithm.reinforce import REINFORCE

def get_algorithm(algorithm, policy_module, environments, args, visdom=None):
    """
    Get and initialize the appropriate learning algorithm, using the provided policy, set of environments, and
    command line arguments.

    :param algorithm: Algorithm string identifier - one of <reinforce>
    :param policy_module: Torch nn module that parameterizes the agent policy
    :param environments: Distributed environment wrapper, for running environments
    :param args: Command line arguments, with algorithm specific parameters (e.g. learning rate, entropy penalty, etc.)
    :param visdom: Visdom wrapper, if provided (default None)

    :return: Initialized algorithm engine, with functionality for training/executing
    """
    if algorithm == 'reinforce':
        learner = REINFORCE(policy_module, environments, visdom)
    else:
        raise NotImplementedError("Algorithm %s not supported! Try one of <reinforce | a2c | ppo>" % algorithm)

    return learner