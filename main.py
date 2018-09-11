"""
main.py
"""
from algorithm import get_algorithm
from distribute import get_distributed_backend
from envs import build_env
from experience import Experience
from policies import get_policy

import argparse
import numpy as np
import torch

# Set Random Seeds
np.random.seed(21)
torch.manual_seed(21)


def parse_args():
    p = argparse.ArgumentParser(description="RL-Kitchen-Sink")
    p.add_argument("-a", "--algorithm", default='reinforce', help='RL algorithm to use: <reinforce>')
    p.add_argument("-e", "--env", default="CartPole-v1", help='OpenAI Gym Environment to run on')
    p.add_argument("-r", "--render", default=False, action='store_true', help='Render environment via Visdom')
    p.add_argument("-s", "--statistics", default=False, action='store_true', help='Plot summary statistics via Visdom')

    p.add_argument("-d", "--distributed_backend", default="ray", help='Distributed Backend: <ray | subprocess>')
    p.add_argument("--num_processes", default=2, help='Number of environments to run in parallel')

    p.add_argument("-p", "--policy", default='mlp', help='Type of policy to use for actor: <mlp | cnn | rnn>')

    p.add_argument("-n", "--num_frames", default=1000000, help='Number of frames to train for (default 1m)')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Command Line Arguments
    args = parse_args()

    # Start Visdom
    viz = None
    if args.render or args.statistics:
        from visdom import Visdom
        viz = Visdom()

    # Build Environment Template -> Lazy Evaluated Callable, for spawning environments
    env_template = build_env(args.env)

    # Build Distributed Environments
    envs = get_distributed_backend(env_template, args.num_processes, backend=args.distributed_backend)

    # Obtain Environment metadata
    metadata = envs.get_metadata()

    # Instantiate Policy
    policy = get_policy(args.policy, metadata)

    # Create agent, with the given training algorithm
    agent = get_algorithm(args.algorithm, policy, envs, args, visdom=viz)

    # Create Experience Buffer, with the environment metadata
    experience = Experience(metadata['max_episode_length'], args.num_processes, metadata['obs_shape'],
                            metadata['action_type'], metadata['action_shape'])

    # Train agent
    agent.train(num_frames=args.num_frames)

    import IPython
    IPython.embed()
