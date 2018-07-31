"""
core.py

Core definition file for environments. Provides a simple string based registry for grabbing OpenAI Gym (or
user-specified) environments, and folding them into a repo specific base class.
"""
import gym


# Create Environment Registry (from OpenAI Gym + User)
gym_registry = gym.envs.registry.env_specs.keys()
user_registry = []


class GymThunk:
    def __init__(self, env_id):
        """
        Initialize a callable object (lazily-evaluated), for generating RL Environments. The purpose of a GymThunk
         is to define an easy wrapper around OpenAI Gym environment generation, for applying wrappers, setting random
         seeds, and provisioning simulators without incurring the cost of creation.

        :param env_id: String of the OpenAI Gym environment to create.
        """
        self.env_id = env_id

        # Create empty metadata dictionary, for checking compatibility downstream (i.e. with policy)
        self.metadata = {}

    def __call__(self, seed=None):
        """
        Provisions the environment, applying any wrappers as necessary.

        :param seed: Random seed for setting environment behavior (when multiple environments). If not specified, chosen
                     randomly.

        :return: Returns provisioned environment.
        """
        env = gym.make(self.env_id)

        if hasattr(env, 'env'):
            # Classic Control Environment
            if 'classic_control' in str(env.env.__class__):
                self.metadata = {'mlp', 'classic'}
                return env

            # Atari Environment
            elif 'atari' in str(env.env.__class__):
                # Atari Environment from Pixels
                if 'ram' not in str(env.env.__class__):
                    self.metadata = {'cnn', 'atari'}
                    raise NotImplementedError

                # Atari Environment from RAM
                else:
                    self.metadata = {'mlp', 'atari-ram'}
                    raise NotImplementedError

            # Others... TODO
            else:
                raise NotImplementedError

        # Some weird environments don't define internal env... TODO
        else:
            raise NotImplementedError




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
        thunk = GymThunk(env_id)()

    else:
        raise KeyError("Given Environment cannot be found in User Registry, or Gym Registry!")



