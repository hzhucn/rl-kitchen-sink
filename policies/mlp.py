"""
mlp.py

Definition for Feed-Forward (Multi-layer Perceptron) policy, for the OpenAI Gym Classic Control + Atari Tasks (CartPole,
MountainCar, Pong, Breakout, etc.).
"""
from policies.base import Policy
from torch.distributions import Categorical

import torch
import torch.nn as nn


class MLP(Policy):
    def __init__(self, config, action_type='Discrete', activation=nn.Tanh, initializer=None):
        """
        Initialize a MLP (Multi-layer Perceptron) Policy, with the given config.

        :param config: List containing parameter mappings.
        :param action_type: Action type, for setting final layer
        :param activation: Activation function for linear layers
        """
        super(MLP, self).__init__()
        self.config, self.action_type, self.activation = config, action_type, activation

        # Setup Variable Initializer
        if not initializer:
            # Via https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/utils.py#L26
            def init_normc(weight, gain=1):
                weight.normal_(0, 1)
                weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

            self.initializer = lambda m: self.weight_bias_initializer(m,
                                                                      weight_init=init_normc,
                                                                      bias_init=lambda x: nn.init.constant_(x, 0))

        # Create Actor
        self.actor = self.compile_actor(self.config, self.initializer, self.activation, self.action_type)

    def forward(self, states):
        """
        Implement the forward pass of the model - a call to the self.actor method

        :param states: Tensor of [bsz, obs_shape] representing states to feed through network
        :return: 2-D Tensor of [bsz, action_shape], representing distribution over actions to take for each state
        """
        return self.actor(states)

    def act(self, states):
        """
        Feed the given states through the policy network, and sample from the resulting distribution to obtain actions.

        :param states: Tensor of [bsz, obs_shape] representing states to feed through network
        :return: Tuple of ([bsz] Torch Tensor, [bsz] Torch Tensor) representing actions and log probabilities
        """
        action_distribution = self.forward(states)

        # If action_type is Discrete, sample from Categorical Distribution
        if self.action_type == 'Discrete':
            action_distribution = Categorical(action_distribution)
        else:
            raise NotImplementedError # TODO!

        # Get actions and log of action probabilities
        actions = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(actions)

        return actions, action_log_prob
