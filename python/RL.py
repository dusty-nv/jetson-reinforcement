import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
#parser.add_argument('--width', type=int, default=64, metavar='N', help='width of virtual screen')
#parser.add_argument('--height', type=int, default=64, metavar='N', help='height of virtual screen')
parser.add_argument('--inputs', type=int, default=64, metavar='N', help='number of data inputs to the neural network')
parser.add_argument('--actions', type=int, defulat=2, metavar='N', help='number of output actions from the neural network')
#parser.add_argument('--env', metavar='N', default='CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# print(gym.envs.registry.all())


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

print('use_cuda: ' + str(use_cuda))
#print('use_env:  ' + args.env)
#env = gym.make(args.env)
#env.seed(args.seed)
torch.manual_seed(args.seed)

num_inputs  = args.inputs
num_actions = args.actions
print('num inputs:  ' + str(num_inputs))
print('num actions: ' + str(num_actions))

num_episodes = 0
max_episodes = 10000

#print('observation space: ' + str(env.observation_space.shape))
#print(env.observation_space.low)
#print(env.observation_space.high)
#num_inputs = env.observation_space.shape[0]
#print('observation dimensions: ' + str(num_inputs))


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, num_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


policy = Policy()

if use_cuda:
    print('detected CUDA support')
    policy.cuda()

optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state, save):			# use DNN to select action from current state
	#state = torch.from_numpy(state).float().unsqueeze(0)
	state = state.unsqueeze(0)
	state = state.cuda()
	probs = policy(Variable(state))
	action = probs.multinomial()

	if save:
		policy.saved_actions.append(action)

	return action.data


def next_action(state):					# inference only
	action = select_action(state, False)
	return action


def finish_episode():					# training at the end of an episode
	print('finish_episode({:d})'.format(num_episodes))
	R = 0
	rewards = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		rewards.insert(0, R)
	rewards = torch.Tensor(rewards)
	rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
	for action, r in zip(policy.saved_actions, rewards):
		action.reinforce(r)
	optimizer.zero_grad()
	autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_actions[:]


def next_reward(state, reward, new_episode):	# next reward is available
	print('reward = ' + str(reward))
	print('new_episode = ' + str(new_episode))

	if new_episode:					# if this is the first frame of a new episode, complete last episode
		if num_episodes != 0:
			policy.rewards.append(reward)	# append the previous action's reward
			finish_episode()			# finish training last episode

		num_episodes += 1				# keep track of the current number of episodes have being run

	action = select_action(state, True)
	return action


#running_reward = 10
#for i_episode in count(1):
#    state = env.reset()
#    for t in range(10000): # Don't infinite loop while learning
#        action = select_action(state)
#        state, reward, done, _ = env.step(action[0,0])
#        if args.render:
#            env.render()
#        policy.rewards.append(reward)
#        if done:
#            break
#
#    running_reward = running_reward * 0.99 + t * 0.01
#    finish_episode()
#    if i_episode % args.log_interval == 0:
#        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
#            i_episode, t, running_reward))
#    if running_reward > 200:
#        print("Solved! Running reward is now {} and "
#              "the last episode runs to {} time steps!".format(running_reward, t))
#        break
