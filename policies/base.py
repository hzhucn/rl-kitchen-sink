"""
policy.py

Base class for policy modules. Defines helper methods for compiling networks from the specified configuration format
    (see policies.configs.deepmind_atari.py).
"""
import torch.nn as nn


def config2layer(spec, prev_shape, initializer, activation):
    """
    Helper function for taking an individual layer specification (list item in config), and turning it into the given
    Torch module.

    :param spec: Individual layer spec (e.g. ["Linear", (64,)])
    :param prev_shape: Shape of previous layer output.
    :param initializer: Layer initializer (for linear/conv/rnn layers).
    :param activation: Activation function.

    :return: Initialized Torch nn.Module for given layer.
    """
    if 'Activation' in spec:
        return activation(), prev_shape

    elif 'Linear' or 'Out' in spec:
        return initializer(nn.Linear(prev_shape, spec[1])), spec[1]

    else:
        raise NotImplementedError("Layer Mapping for spec %s not implemented!" % str(spec))


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    @staticmethod
    def compile_actor(config, initializer, activation=nn.Tanh, action_type='Discrete'):
        """
        Compile a neural network from a configuration (vis-a-vis the format outlined in policies.configs.deepmind_atari)

        :param config: Configuration (network specification)
        :param initializer: Joint weight/bias initialization function (see staticmethod below).
        :param activation: Activation function

        :return: A Torch Sequential module, representing the actor network (for generating policies)
        """
        input_dim = config.pop(0)[1]

        # Assemble layers in loop
        layers, prev_shape = [None for _ in range(len(config))], input_dim
        for i, spec in enumerate(config):
            layers[i], prev_shape = config2layer(spec, prev_shape, initializer, activation)

        # Append Softmax if action_type == 'Discrete'
        if action_type == 'Discrete':
            layers.append(nn.Softmax(dim=1))

        # Return Sequential
        return nn.Sequential(*layers)


    @staticmethod
    def weight_bias_initializer(mod, weight_init, bias_init, gain=1):
        """
        Helper function for quick functional initializiation of weights/biases.

        Reference: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/utils.py#L26

        :param mod: Torch Module on which to apply the given initializers
        :param weight_init: Weight initializer function
        :param bias_init: Bias initializer function
        :param gain: Default gain (activation-function specific) for initialization.

        :return: Initialized Torch Module
        """
        weight_init(mod.weight.data, gain=gain)
        bias_init(mod.bias.data)
        return mod

    def forward(self, states):
        raise NotImplementedError

    def act(self, states):
        raise NotImplementedError