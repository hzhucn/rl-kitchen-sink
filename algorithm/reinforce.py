"""
reinforce.py

Implementation of the REINFORCE policy gradient algorithm (Williams, 1992). Supports functionality for clipping
gradients, adding an entropy penalty, adding simple baseline functions for variance reduction (e.g. subtract mean),
as well as support for multiple workers (summing gradients over workers).
"""
import torch


class REINFORCE(object):
    def __init__(self, policy, envs, visdom=None):
        """
        Initialize a REINFORCE learner, with the given policy, set of distributed environments, and necessary
        hyperparameters.

        :param policy: Torch nn module parameterizing agent policy
        :param envs: Distributed environments for executing actions and observing rewards
        :param visdom: Visdom visualization environment (default: None)
        """
        self.policy, self.envs, self.viz = policy, envs, visdom

        # Print Message to Screen, with hyperparameters and ckpt file path
        print("[*] Initializing REINFORCE Agent with following Hyperparameters!")

        # Set current processed frames
        self.processed_frames = 0

    def train(self, num_frames):
        """
        Train the REINFORCE Agent for the given number of frames (divided across all workers). Note that as a design
        choice, all transfer to/from numpy/torch tensors happens in this function => environments expect numpy,
        policy expects torch!

        :param num_frames: Number of frames to train on
        """
        # Reset Environments, and get current states (cast to Torch tensor)
        states = torch.from_numpy(self.envs.reset()).float()

        # Collect actions via a synchronous pass through Policy
        actions, action_log_probs = self.policy.act(states)
        actions = actions.data.numpy()

        # Perform actions in environments (update states)
        states, rewards, done = self.envs.act(actions, action_log_probs)


