"""
main.py
"""
from envs.core import build_env
from visdom import Visdom
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="RL-Kitchen-Sink")
    p.add_argument("-a", "--algorithm", default='reinforce', help='RL algorithm to use: <reinforce>')
    p.add_argument("-e", "--env", default="CartPole-v1", help='OpenAI Gym Environment to run on')
    p.add_argument("-d", "--distributed_backend", default="ray", help='Distributed Backend: <ray | subprocess>')
    p.add_argument("-r", "--render", default=False, action='store_true', help='Render environment via Visdom')
    p.add_argument("-p", "--plot", default=False, action='store_true', help='Plot summary statistics via Visdom')

    p.add_argument("--num_processes", default=4, help='Number of environments to run in parallel')

    return p.parse_args()

if __name__ == "__main__":
    # Parse Command Line Arguments
    args = parse_args()

    # Start Visdom
    viz = Visdom()

    # Build Environment Template -> Lazy Evaluated Callable, for spawning environments
    env = build_env(args.env)

    # Build Distributed Environments
    envs = DistributedEnv(env, args.num_processes)


    import IPython
    IPython.embed()