#!/usr/bin/env python3
# coding: utf-8

import gym
import cv2
import torch
import numpy
import stable_baselines3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action="store_true", help='print out additional package information')
args = parser.parse_args()

if args.verbose:
    print(cv2.getBuildInformation())

print(f'cv2 version {cv2.__version__}')    
print(f'gym version {gym.__version__}')
print(f'pytorch version {torch.__version__}')
print(f'numpy version {numpy.__version__}')
print(f'stable_baselines3 version {stable_baselines3.__version__}')
