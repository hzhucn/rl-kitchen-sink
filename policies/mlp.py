"""
classic_mlp.py

Definition for Feed-Forward (Multi-layer Perceptron) policy, for the OpenAI Gym Classic Control Tasks (CartPole,
MountainCar, etc.).
"""
import torch.nn as nn


class ClassicMLP(nn.Module):
    def __init__(self):
        super(ClassicMLP, self).__init__()

    def forward(self, states):
        pass