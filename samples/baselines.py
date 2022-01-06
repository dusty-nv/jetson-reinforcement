#!/usr/bin/env python3
# coding: utf-8

# xvfb-run -a -s "-screen 0 1400x900x24" python debug-gym.py

#from pyvirtualdisplay import Display
#virtual_display = Display(visible=0, size=(200, 300))
#virtual_display.start()

# https://githubmemory.com/repo/pyglet/pyglet/issues/367
import pyglet
pyglet.options['headless'] = True

import os
import sys
import argparse

import gym
import cv2
import torch
import stable_baselines3
#import matplotlib.pyplot as plt

from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper

from stable_baselines3 import A2C, DDPG, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, choices=['A2C', 'DDPG', 'DQN', 'PPO'], default='A2C', help='the RL algorithm to use')
parser.add_argument('--env', '--environment', type=str, default='CartPole-v1')
parser.add_argument('--list-envs', action='store_true', help='list the available environments and exit')

parser.add_argument('--observations', choices=['state', 'pixels'], nargs="+", default=['state'], help='observation spaces to use')
parser.add_argument('--pixel-res', type=int, default=224, help='the NxN pixel resolution used if --observation=pixels is specified')
parser.add_argument('--pixel-gray', action='store_true', help='convert RGB pixels to grayscale if --observation=pixels is specified')

parser.add_argument('--disable-display', action='store_true', help='disable visualization of environment')
parser.add_argument('--display-freq', type=int, default=2, help='display every N-th frame and skip rendering the others')
parser.add_argument('--display-scale', type=float, default=0.5, help='downscaling factor for rendering the frames')

parser.add_argument('--train-target', type=str, choices=['timesteps', 'reward', 'reward_mean'], default='timesteps', help='condition used to terminate training')
parser.add_argument('--train-steps', type=int, default=10000, help='number of timesteps to train for when --train-target=timesteps')
parser.add_argument('--eval-steps', type=int, default=1000, help='number of timesteps to evaluate after training is complete')
parser.add_argument('--reward-threshold', type=float, default=None, help='target reward to train for when --train-target=reward')

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
    def __init__(self, env_monitor, verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        self.env_monitor = env_monitor
        self.frame_counter = 0
        
    def _on_step(self) -> bool:
        if self.frame_counter % args.display_freq == 0 and not args.disable_display:
            img = self.training_env.render(mode='rgb_array')

            if args.display_scale != 1.0:
                img = cv2.resize(img, (int(img.shape[1] * args.display_scale), int(img.shape[0] * args.display_scale)), interpolation=cv2.INTER_NEAREST) # INTER_LINEAR
                
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'{args.env} Training', img)  
            cv2.waitKey(1) 
            
        self.frame_counter += 1
        
        # check to see if training target has been reached
        if args.train_target == 'reward':
            if len(self.env_monitor.ep_rewards) == 0:
                return True
              
            continue_training = bool(self.env_monitor.ep_rewards[-1] < args.reward_threshold)
            
            if not continue_training:
                print("Stopping training because the last episode reward {:.2f} is above the threshold {}".format(self.env_monitor.ep_rewards[-1], args.reward_threshold))

            return continue_training
            
        elif args.train_target == 'reward_mean':
            if self.env_monitor.ep_reward_mean is None:
                return True
            
            continue_training = bool(self.env_monitor.ep_reward_mean < args.reward_threshold)
            
            if not continue_training:
                print("Stopping training because the mean episode reward {:.2f} is above the threshold {}".format(self.env_monitor.ep_reward_mean, args.reward_threshold))

            return continue_training
        else:
            return True   # baselines will automatically stop when timesteps are reached
  
  
class EnvMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    """
    def __init__(self, env):
        super(EnvMonitor, self).__init__(env=env)
        self.clear()
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        
        if done:
            self.ep_rewards.append(sum(self.rewards))
            self.rewards = []
            
            if len(self.ep_rewards) > self.ep_reward_mean_n:
                self.ep_rewards = self.ep_rewards[-self.ep_reward_mean_n:]
                
            self.ep_reward_mean = sum(self.ep_rewards) / len(self.ep_rewards)
            print(f'episode {self.num_episodes}  steps {self.timesteps}  ep_reward {self.ep_rewards[-1]:.2f}  ep_reward_mean {self.ep_reward_mean:.2f}')
            self.num_episodes += 1
            
        self.timesteps += 1
        return observation, reward, done, info
      
    def clear(self):
        self.rewards = []
        self.ep_rewards = []
        self.ep_reward_mean = None
        self.ep_reward_mean_n = 100
        self.num_episodes = 0
        self.timesteps = 0
        
# Create and wrap the environment
env = EnvMonitor(gym.make(args.env))
env_monitor = env
env.reset()

if args.train_target == 'reward' and args.reward_threshold is None:
    if env.spec.reward_threshold is None:
        raise ValueError(f"the environment {args.env} doesn't have a default reward_threshold.  Please specify one with --reward-threshold or use --train-mode=timesteps instead")    
    args.reward_threshold = env.spec.reward_threshold
    args.train_steps = 10000000
    
if args.train_target == 'reward':
    print(f'training until episode reward reaches {args.reward_threshold}')
elif args.train_target == 'reward_mean':
    print(f'training until mean episode reward reaches {args.reward_threshold}')
else:
    print(f'training for {args.train_steps} timesteps')
    

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

# create model
policy = 'MultiInputPolicy' if 'pixels' in args.observations else 'MlpPolicy' # TODO:  CnnPolicy

if args.model == 'A2C':
    model = A2C(policy, env, verbose=1)  
elif args.model == 'DDPG':
    model = DDPG(policy, env, verbose=1) 
elif args.model == 'DQN':
    model = DQN(policy, env, verbose=1) 
elif args.model == 'PPO':
    model = PPO(policy, env, verbose=1) 
    
print('Model:  ' + args.model)
print('Policy: ' + policy)

# training 
model.learn(total_timesteps=args.train_steps, callback=TrainingMonitor(env_monitor))

print('done training')
cv2.destroyAllWindows()
env_monitor.clear()

obs = env.reset()
for i in range(args.eval_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    img = env.render(mode='rgb_array')

    if args.display_scale != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * args.display_scale), int(img.shape[0] * args.display_scale)), interpolation=cv2.INTER_NEAREST) # INTER_LINEAR
        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{args.env} Eval', img)  
    cv2.waitKey(1) 
            
    if done:
        obs = env.reset()
               