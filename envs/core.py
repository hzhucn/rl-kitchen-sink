"""
envs/core.py

Defines class switcher, for picking the appropriate environment spec. Provides a simple string based registry for
grabbing OpenAI Gym (or user-specified) environments, and folding them into a lazy-evaluated environment callable.
"""
from envs.gym import GymThunk
import gym


# Create Environment Registry (from OpenAI Gym + User)
gym_registry = gym.envs.registry.env_specs.keys()
user_registry = []


def build_env(env_id):
    """
    Function for looking up environment from registry, and packing it with the appropriate pre-processors (i.e.

    :param env_id: Environment ID to look-up in registry.
    :return:
    """
    # Check if Env is User Defined
    if env_id in user_registry:
        raise NotImplementedError  # TODO - Add User Defined Environment Example!

    # Go through Gym Configurations
    elif env_id in gym_registry:
        # Create Environment Thunk
        thunk = GymThunk(env_id)
        return thunk

    else:
        raise KeyError("Given Environment cannot be found in User Registry, or Gym Registry!")