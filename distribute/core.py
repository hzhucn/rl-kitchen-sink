"""
core.py

Defines class switcher, for picking the appropriate distributed backend.
"""
from distribute.ray import RayEnvironment

import numpy as np

def get_distributed_backend(env_thunk, num_envs, backend='ray'):
    """
    Create a Distributed Environment, with the specified number of parallel processes, and the given backend.

    :param env_thunk: Thunk for quickly spinning up environments -> passed to backend to run in own process..
    :param num_envs: Number of environments to spin up.
    :param backend: Distributed backend (default `ray`, options: <ray | subprocess>)
    """
    if backend == 'ray':
        import ray ; ray.init()
        d = RayEnvironment(env_thunk, num_envs, list(np.random.randint(0, 1000, size=[num_envs])))

    elif backend == 'subprocess':
        raise NotImplementedError  # TODO

    else:
        raise NotImplementedError("%s currently not supported. Try one of <ray | subprocess>" % backend)

    return d