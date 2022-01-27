#!/usr/bin/env python3
# coding: utf-8

import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sorted', action="store_true", help='sort the list of Gym environments alphabetically')
args = parser.parse_args()

if args.sorted:
    envs = sorted(gym.envs.registry.all(), key = lambda x : x.id)
else:
    envs = gym.envs.registry.all()
    
for env in envs:
    print(f'{env.id:<50s} reward_threshold={env.reward_threshold} max_episode_steps={env.max_episode_steps}')
               