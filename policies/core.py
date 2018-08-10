"""
core.py

Defines class switcher, for picking the appropriate policy.
"""
from policies.mlp import MLP
from policies.configs.deepmind_atari import deepmind_atari

# User Config Registry - Add imports, dictionary entries
user_registry = {}


def get_policy(prefix, env_metadata, config='deepmind_atari'):
    """
    Creates the desired

    :param prefix: Type of Policy to use (e.g. mlp, cnn, lstm, etc.)
    :param env_metadata: Environment metadata, for use in creating input/output dimensions.
    :param config: Configuration for layer sizes, defined in policies/configs -> Dictionary containing mappings
    :return: PyTorch nn.Module defining policy.
    """
    check_policy_compatibility(prefix, env_metadata)

    # DeepMind Atari
    if config == 'deepmind_atari':
        # Fill in Input/Output fields, using the env_metadata
        cfg = deepmind_atari[prefix]
        cfg[0], cfg[-1] = ["In", *env_metadata['obs_shape']], ["Out", env_metadata['action_shape']]

        # Switch on Policy Prefix
        if prefix == 'mlp':
            return MLP(cfg)
        else:
            raise NotImplementedError # TODO

    elif config in user_registry:
        pass # TODO User Registry Config
    else:
        raise NotImplementedError("Config %s does not exist!" % config)



def check_policy_compatibility(prefix, metadata):
    """
    Helper function to check whether the choice of policy is compatible with the given environment.

    :param prefix: Policy prefix (e.g. mlp, cnn, etc.)
    :param metadata: Environment metadata dictionary
    :return: None, less an exception is raised
    """
    if metadata['env-type'] == 'classic' and prefix != 'mlp':
        raise NotImplementedError("Classic Control Environments cannot utilize %s policies" % prefix)