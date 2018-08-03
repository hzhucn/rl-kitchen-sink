"""
ray.py

Class defining Ray distributed environment wrapper. Defines both the global scheduler, and the individual workers.
"""
import ray
import numpy as np


class RayEnvironment(object):
    def __init__(self, env_thunk, num_envs):
        """
        Initialize a Ray Environment scheduler, with the given thunk, and the target number of environments.

        :param env_thunk: Thunk for quickly spinning up environments.
        :param num_envs: Number of environments to spin up.
        """
        self.thunk, self.num_envs = env_thunk, num_envs

        # Create Environments
        self.envs = [RayActor.remote(env_thunk) for _ in range(self.num_envs)]

    def get_metadata(self):
        """
        Get environment metadata - information about policy compatibility, env-type, observation and action shapes, etc.

        :return: Environment Metadata Dictionary
        """
        return ray.get(self.envs[0].get_metadata.remote())

@ray.remote
class RayActor(object):
    def __init__(self, env_thunk):
        """
        Initialize a single distributed environment, as an independent Ray Worker node.

        :param env_thunk: Thunk for spinning up environments.
        """
        self.env, self.metadata = env_thunk(seed=np.random.random_integers(0, high=10000))
        self.env.reset()

    def get_env(self):
        return self.env

    def get_metadata(self):
        return self.metadata