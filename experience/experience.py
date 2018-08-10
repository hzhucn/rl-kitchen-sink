"""
experience.py

Class definition for the Experience buffer, storing all states/actions/rewards, as well as the torch log probabilities,
for performing gradient updates. Note that the reason Experience is centralized is so that the corresponding tensors
can be easily placed on GPU, for compute efficiency.

Credit: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/storage.py
"""
import torch

class Experience(object):
    def __init__(self, timesteps, num_envs, obs_shape, action_type, action_shape):
        """
        Initialize an Experience buffer, with the given capacity (timesteps x num_envs x [obs/action])

        :param timesteps: Number of states observed (one more than the number of actions predicted!)
        :param num_envs: Number of parallel environments.
        :param obs_shape: Shape of observations (e.g. C x H x W for images)
        :param action_type: Type of actions (e.g. 'Discrete')
        :param action_shape: Shape of actions (scalar values for Discrete actions)
        """
        self.timesteps, self.num_envs, self.obs_shape = timesteps, num_envs, obs_shape
        self.action_type, self.action_type = action_type, action_shape

        # Create Buffers for observations, actions, action_probabilities, rewards
        self.observations = None
        import IPython
        IPython.embed()