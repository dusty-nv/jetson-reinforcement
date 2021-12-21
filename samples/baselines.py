#!/usr/bin/env python3
# coding: utf-8

# xvfb-run -a -s "-screen 0 1400x900x24" python debug-gym.py

#from pyvirtualdisplay import Display
#virtual_display = Display(visible=0, size=(200, 300))
#virtual_display.start()

# https://githubmemory.com/repo/pyglet/pyglet/issues/367
import pyglet
pyglet.options['headless'] = True

import sys
import argparse

import gym
import cv2
import torch
import stable_baselines3
#import matplotlib.pyplot as plt

from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

parser = argparse.ArgumentParser()

parser.add_argument('--env', '--environment', type=str, default='CartPole-v1')
parser.add_argument('--list-envs', action='store_true', help='list the available environments and exit')
parser.add_argument('--observations', choices=['state', 'pixels'], nargs="+", default=['state'], help='observation spaces to use')
parser.add_argument('--pixel-res', type=int, default=224, help='the NxN pixel resolution used if --observation=pixels is specified')
parser.add_argument('--pixel-gray', action='store_true', help='convert RGB pixels to grayscale if --observation=pixels is specified')
parser.add_argument('--disable-display', action='store_true', help='disable visualization of environment')
parser.add_argument('--display-freq', type=int, default=2, help='display every N-th frame and skip rendering the others')
parser.add_argument('--display-scale', type=float, default=0.5, help='downscaling factor for rendering the frames')

args = parser.parse_args()
print(args)

print(f'gym version {gym.__version__}')
print(f'pytorch version {torch.__version__}')
print(f'stable_baselines3 version {stable_baselines3.__version__}')

if args.list_envs:
    for env in gym.envs.registry.all(): #sorted(gym.envs.registry.all(), key = lambda x : x.id):
        print(f'{env.id:<50s} reward_threshold={env.reward_threshold} max_episode_steps={env.max_episode_steps}')
    sys.exit(0)
    
class TrainingMonitor(BaseCallback):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    """
    def __init__(self, verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        self.frame_counter = 0
        
    def _on_step(self) -> bool:
        if self.frame_counter % args.display_freq == 0 and not args.disable_display:
            img = self.training_env.render(mode='rgb_array')
            #print(img.shape)
            #plt.figure()
            #plt.imshow(img)
            #plt.show()
            if args.display_scale != 1.0:
                img = cv2.resize(img, (int(img.shape[1] * args.display_scale), int(img.shape[0] * args.display_scale)), interpolation=cv2.INTER_NEAREST) # INTER_LINEAR
                
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'{args.env} Training', img)  
            cv2.waitKey(1) 
            
        self.frame_counter += 1
        return True
        
        
env = gym.make(args.env)
env.reset()

if 'pixels' in args.observations:
    env = PixelObservationWrapper(env, pixels_only='state' not in args.observations)
    
    print(f'Wrapping the env in a PixelObservation wrapper with pixels_only={env._pixels_only}')
    
    #if 'state' not in args.observations:  # TODO make a custom wrapper that handles the observation dict when both state + pixels are used
    #    env = ResizeObservation(env, args.pixel_res)
    #    print(f'Wrapping the env in a ResizeObservation wrapper with {args.pixel_res}x{args.pixel_res} resolution')
    #    
    #    if args.pixel_gray:
    #        env = GrayScaleObservation(env)
    #        print(f'Wrapping the env in a GrayScaleObservation wrapper')
    
print('Observation space:  ' + str(env.observation_space))
print('Action space:  ' + str(env.action_space))

model = A2C('MultiInputPolicy' if 'pixels' in args.observations else 'MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, callback=TrainingMonitor())

print('done training')

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    plt.figure(3)
    plt.clf()
    img = env.render(mode='rgb_array')
    print(img.shape)
    plt.imshow(img)
    plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.axis('off')
    if done:
      obs = env.reset()