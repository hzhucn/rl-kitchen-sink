"""
main.py
"""
from envs.core import build_env
from visdom import Visdom
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="RL-Kitchen-Sink")
    p.add_argument("--algorithm", default='reinforce', help='RL algorithm to use: <reinforce>')
    p.add_argument("--env", default="CartPole-v1", help='OpenAI Gym Environment to run on')
    p.add_argument("--mode", default="train", help='Run model in train or evaluation')

    p.add_argument("--num_processes", default=4, help='Number of environments to run in parallel')

    return p.parse_args()

if __name__ == "__main__":
    # Parse Command Line Arguments
    args = parse_args()

    # Start Visdom
    viz = Visdom()

    # Build Environment Template
    env = build_env(args.env)

    # Build Distributed Environments
    envs = DistributedEnv(env, args.num_processes)


    import IPython
    IPython.embed()