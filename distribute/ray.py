"""
ray.py

Class defining Ray distributed environment wrapper. Defines both the global scheduler, and the individual workers.
"""
import ray
import numpy as np


class RayEnvironment(object):
    def __init__(self, env_thunk, num_envs, seeds):
        """
        Initialize a Ray Environment scheduler, with the given thunk, and the target number of environments.

        :param env_thunk: Thunk for quickly spinning up environments
        :param num_envs: Number of environments to spin up
        :param seeds: Random seeds for environments
        """
        self.thunk, self.num_envs = env_thunk, num_envs

        # Create Environments
        self.envs = [RayActor.remote(env_thunk, seeds[i]) for i in range(self.num_envs)]

    def get_metadata(self):
        """
        Get environment metadata - information about policy compatibility, env-type, observation and action shapes, etc.

        :return: Environment Metadata Dictionary
        """
        return ray.get(self.envs[0].get_metadata.remote())

    def reset(self):
        """
        Reset all environments, and return the starting observations, as a single tensor

        :return: Tensor object containing initial states for each environment after reset
        """
        return np.vstack(ray.get([env.reset.remote() for env in self.envs]))

    def act(self, actions, log_probabilities):
        """
        Execute actions on each environment, with each environment logging both the executed actions, as well as the
        action log probability.

        :param actions: 1-D Numpy Array representing actions to take for each environment.
        :param log_probabilities: 1-D Torch Tensor representing action log probabilities
        :return: Tuple of ([bsz, obs_shape] states, [bsz] rewards, [bsz] done)
        """
        update = ray.get([self.envs[i].act.remote(actions[i], log_probabilities[i]) for i in range(len(self.envs))])
        states, rewards, done = map(lambda x: np.array(x), zip(*update))
        return states, rewards, done

@ray.remote
class RayActor(object):
    def __init__(self, env_thunk, seed):
        """
        Initialize a single distributed environment, as an independent Ray Worker node.

        :param env_thunk: Thunk for spinning up environments.
        :param seed: Random seed for environment
        """
        self.env, self.metadata = env_thunk(seed=seed)

        # Create buffers for logging actions, rewards, and action log probabilities (Torch tensor for gradients)
        self.actions, self.rewards, self.log_probabilities = [], [], []

        # State variables for tracking episode termination, and current state
        self.state, self.reward, self.done = None, None, False

    def get_env(self):
        return self.env

    def get_metadata(self):
        return self.metadata

    def reset(self):
        self.state, self.done = self.env.reset(), False
        return self.state

    def act(self, action, log_probability):
        # Only execute action if not done!
        if not self.done:
            self.actions.append(action)
            self.log_probabilities.append(log_probability)

            # Execute action in environment
            self.state, self.reward, self.done, _ = self.env.step(action)

            # Append reward to rewards
            self.rewards.append(self.reward)

        # Return new state, reward, done
        return self.state, self.reward, self.done