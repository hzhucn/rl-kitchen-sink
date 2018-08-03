"""
backend.py

Core class for defining the Distributed Env Backend -> A wrapper for holding multiple environments, and facilitating
communication of states/actions/rewards.
"""
from distribute.ray import RayEnvironment


class DistributedEnv(object):
    def __init__(self, env_thunk, num_envs, backend='ray'):
        """
        Create a Distributed Environment, with the specified number of parallel processes, and the given backend.

        :param env_thunk: Thunk for quickly spinning up environments -> passed to backend to run in own process..
        :param num_envs: Number of environments to spin up.
        :param backend: Distributed backend (default `ray`, options: <ray | subprocess>)
        """
        self.thunk, self.num_envs, self.backend = env_thunk, num_envs, backend

        # Create Core, by switching off of backend
        if self.backend == 'ray':
            import ray ; ray.init()
            self.core = RayEnvironment(self.thunk, self.num_envs)
        elif self.backend == 'subprocess':
            raise NotImplementedError # TODO
        else:
            raise NotImplementedError("%s currently not supported. Try one of <ray | subprocess>" % self.backend)

