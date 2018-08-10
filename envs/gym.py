"""
gym.py

Core definition file for OpenAI Gym environments. Given an environment ID, build a lazy-evaluated callable object
for creating an environment, and lifting any necessary metadata.
"""
import gym


class GymThunk(object):
    def __init__(self, env_id):
        """
        Initialize a callable object (lazily-evaluated), for generating RL Environments. The purpose of a GymThunk
         is to define an easy wrapper around OpenAI Gym environment generation, for applying wrappers, setting random
         seeds, and provisioning simulators without incurring the cost of creation.

        :param env_id: String of the OpenAI Gym environment to create.
        """
        self.env_id = env_id

    def __call__(self, seed=None):
        """
        Provisions the environment, applying any wrappers as necessary.

        :param seed: Random seed for setting environment behavior (when multiple environments). If not specified, chosen
                     randomly.

        :return: Returns provisioned environment.
        """
        env, metadata = gym.make(self.env_id), {}
        env.seed(seed=seed)

        if hasattr(env, 'env'):
            # Classic Control Environment
            if 'classic_control' in str(env.env.__class__):
                metadata = {'policy': ['mlp'], 'env-type': 'classic',
                            'obs_shape': env.observation_space.shape,
                            'action_type': env.action_space.__class__.__name__,
                            'action_shape': env.action_space.n if env.action_space.__class__.__name__ == 'Discrete'
                                                               else env.action_space.shape, # TODO!
                            'max_episode_length': env.spec.max_episode_steps}

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

        return env, metadata
