# -*- coding: utf-8 -*-

import argparse
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

if use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


# parse command line
parser = argparse.ArgumentParser(description='PyTorch DQN runtime')
parser.add_argument('--width', type=int, default=64, metavar='N', help='width of virtual screen')
parser.add_argument('--height', type=int, default=64, metavar='N', help='height of virtual screen')
parser.add_argument('--channels', type=int, default=3, metavar='N', help='channels in the input image')
parser.add_argument('--actions', type=int, default=3, metavar='N', help='number of output actions from the neural network')
parser.add_argument('--optimizer', default='RMSprop', help='Optimizer of choice')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N', help='optimizer learning rate')
parser.add_argument('--replay_mem', type=int, default=10000, metavar='N', help='replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--gamma', type=float, default=0.9, metavar='N', help='discount factor for present rewards vs. future rewards')
parser.add_argument('--epsilon_start', type=float, default=0.9, metavar='N', help='epsilon_start of random actions')
parser.add_argument('--epsilon_end', type=float, default=0.05, metavar='N', help='epsilon_end of random actions')
parser.add_argument('--epsilon_decay', type=float, default=200, metavar='N', help='exponential decay of random actions')
parser.add_argument('--allow_random', type=int, default=1, metavar='N', help='Allow DQN to select random actions')
parser.add_argument('--debug_mode', type=int, default=0, metavar='N', help='debug mode')
parser.add_argument('--use_lstm', type=int, default=1, metavar='N', help='use LSTM layers in network')
parser.add_argument('--lstm_size', type=int, default=256, metavar='N', help='number of inputs to LSTM')

args = parser.parse_args()

input_width    = args.width
input_height   = args.height
input_channels = args.channels
num_actions    = args.actions
optimizer 	= args.optimizer
learning_rate 	= args.learning_rate
replay_mem 	= args.replay_mem
batch_size 	= args.batch_size
gamma 		= args.gamma
epsilon_start 	= args.epsilon_start
epsilon_end 	= args.epsilon_end
epsilon_decay 	= args.epsilon_decay
allow_random 	= args.allow_random
debug_mode 	= args.debug_mode
use_lstm		= args.use_lstm
lstm_size      = args.lstm_size


print('[deepRL]  use_cuda:       ' + str(use_cuda))
print('[deepRL]  use_lstm:       ' + str(use_lstm))
print('[deepRL]  lstm_size:      ' + str(lstm_size))
print('[deepRL]  input_width:    ' + str(input_width))
print('[deepRL]  input_height:   ' + str(input_height))
print('[deepRL]  input_channels: ' + str(input_channels))
print('[deepRL]  num_actions:    ' + str(num_actions))
print('[deepRL]  optimizer:      ' + str(optimizer))
print('[deepRL]  learning rate:  ' + str(learning_rate))
print('[deepRL]  replay_memory:  ' + str(replay_mem))
print('[deepRL]  batch_size:     ' + str(batch_size))
print('[deepRL]  gamma:          ' + str(gamma))
print('[deepRL]  epsilon_start:  ' + str(epsilon_start))
print('[deepRL]  epsilon_end:    ' + str(epsilon_end))
print('[deepRL]  epsilon_decay:  ' + str(epsilon_decay))
print('[deepRL]  allow_random:   ' + str(allow_random))
print('[deepRL]  debug_mode:     ' + str(debug_mode))




######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta) 
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *quality* of
# taking each action given the current input.
#

class DQN(nn.Module):

	def __init__(self):
		print('[deepRL]  DQN::__init__()')
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		
		#if input_width >= 128 and input_height >= 128:
		#	self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		#	self.bn4 = nn.BatchNorm2d(32)
		#	self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		#	self.bn5 = nn.BatchNorm2d(32)

		self.head = None

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		#if input_width >= 128 and input_height >= 128:
		#	x = F.relu(self.bn4(self.conv4(x)))
		#	x = F.relu(self.bn5(self.conv5(x)))

		y = x.view(x.size(0), -1)

		if self.head is None:
			print('[deepRL]  nn.Conv2d() output size = ' + str(y.size(1)))
			self.head = nn.Linear(y.size(1), num_actions)

			if use_cuda:
				self.head.cuda()

		return self.head(y)


#
# Deep Recurrent Q-Network (with LSTM)
#
class DRQN(nn.Module):

	def __init__(self):
		print('[deepRL]  DRQN::__init__()')
		super(DRQN, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		self.lstm = None
		self.head = None

	def forward(self, inputs):
		x, (hx, cx) = inputs

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		y = x.view(x.size(0), -1)

		if self.lstm is None:
			print('[deepRL]  nn.Conv2d() output size = ' + str(y.size(1)))
			self.lstm = nn.LSTMCell(y.size(1), lstm_size)
	
			if use_cuda:
				self.lstm.cuda()

		if self.head is None:
			self.head = nn.Linear(lstm_size, num_actions)

			if use_cuda:
				self.head.cuda()

		hx, cx = self.lstm(y, (hx, cx))
		y = hx

		return self.head(y), (hx, cx)

	def init_states(self, batch_dim):
		hx = Variable(torch.zeros(batch_dim, lstm_size))
		cx = Variable(torch.zeros(batch_dim, lstm_size))
		return hx, cx

	def reset_states(self, hx, cx):
		hx[:, :] = 0
		cx[:, :] = 0
		return hx.detach(), cx.detach()

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#


print('[deepRL]  creating DQN model instance')

lstm_actor_hx = lstm_actor_cx = None
lstm_batch_hx = lstm_batch_cx = None
lstm_final_hx = lstm_final_cx = None

if use_lstm:
	model = DRQN()

	lstm_actor_hx, lstm_actor_cx = model.init_states(1)
	lstm_batch_hx, lstm_batch_cx = model.init_states(batch_size)
	lstm_final_hx, lstm_final_cx = model.init_states(batch_size)

	print('[deepRL]  LSTM (hx, cx) size = ' + str(lstm_actor_hx.size(1)))
else:
	model = DQN()

print('[deepRL]  DQN model instance created')

if use_cuda:
    model.cuda()

def load_model(filename):
	print('[deepRL]  loading model checkpoint from ' + filename)
	model.load_state_dict(torch.load(filename))

def save_model(filename):
	print('[deepRL]  saving model checkpoint to ' + filename)
	torch.save(model.state_dict(), filename)

if (optimizer == 'Adam'):
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

elif (optimizer == 'RMSprop'):
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

else:
	print('Optimizer Error. Make sure you have choosen the right optimizer and learning rate')
	sys.exit()

memory = ReplayMemory(replay_mem)

steps_done = 0

def select_action(state, allow_rand):
	global steps_done
	global lstm_actor_hx
	global lstm_actor_cx

	sample = random.random()
	eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
		math.exp(-1. * steps_done / epsilon_decay)
	steps_done += 1
	if not allow_rand or sample > eps_threshold:
		if use_lstm:
			action, (lstm_actor_hx, lstm_actor_cx) = model(
				(Variable(state, volatile=True).type(FloatTensor), (lstm_actor_hx, lstm_actor_cx)))
			action = action.data.max(1)[1].unsqueeze(0)
		else:
			action = model(
				Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].unsqueeze(0)
		#print('select_action = ' + str(action))
		return action
	else:
#		print('[deepRL]  DQN selected exploratory random action')
		return LongTensor([[random.randrange(num_actions)]])

episode_durations = []

print('[deepRL]  DQN script done init')


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0


def optimize_model():
	global last_sync
	global lstm_batch_hx
	global lstm_batch_cx
	global lstm_final_hx
	global lstm_final_cx

	if use_lstm:
		lstm_batch_hx, lstm_batch_cx = model.reset_states(lstm_batch_hx, lstm_batch_cx)
		lstm_final_hx, lstm_final_cx = model.reset_states(lstm_final_hx, lstm_final_cx)

	# sample a batch of transitions from the replay buffer
	if len(memory) < batch_size:
		return

	transitions = memory.sample(batch_size)

	# Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
		                                batch.next_state)))

	# We don't want to backprop through the expected action values and volatile
	# will save us on temporarily changing the model parameters'
	# requires_grad to False!
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state
		                                      if s is not None]),
		                           volatile=True)
	#print(non_final_next_states)
	state_batch = Variable(torch.cat(batch.state))
	action_batch = Variable(torch.cat(batch.action))
	reward_batch = Variable(torch.cat(batch.reward))

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken
	if use_lstm:
		model_batch, (lstm_batch_hx, lstm_batch_cx) = model((state_batch, (lstm_batch_hx, lstm_batch_cx)))
	else:
		model_batch = model(state_batch)

	state_action_values = model_batch.gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	next_state_values = Variable(torch.zeros(batch_size).type(Tensor))

	if use_lstm:
		final_batch, (lstm_final_hx, lstm_final_cx) = model((non_final_next_states, (lstm_final_hx, lstm_final_cx)))
		next_state_values[non_final_mask] = final_batch.max(1)[0]
	else:
		next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

	# Now, we don't want to mess up the loss with a volatile flag, so let's
	# clear it. After this, we'll just end up with a Variable that has
	# requires_grad=False
	next_state_values.volatile = False
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * gamma) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in model.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

last_action = None
last_state = None
curr_state = None
last_diff = None
curr_diff = None

def next_action(state_in):
	global last_state
	global curr_state
	global last_action
	global curr_diff
	global last_diff

	#print('state = ' + str(state.size()))
	state = state_in.clone().unsqueeze(0)
	#print('state = ' + str(state.size()))
	
	if curr_state is not None:
		last_state = curr_state.clone()

	if curr_diff is not None:
		last_diff = curr_diff.clone()

	curr_state = state.clone()

	last_action = select_action(curr_state, allow_random)

	if last_state is not None:
		curr_diff = state - last_state
		#print('curr_diff = ' + str(curr_diff.abs().sum()) + ' ' + str(curr_diff.max()) + ' ' + str(curr_diff.min()))
		#last_action = select_action(curr_diff, allow_random)
		
	#else:
	#	curr_state = None
	#	curr_diff = None
	#	last_action = None

	if last_action is not None:
		#print('ret action = ' + str(last_action[0][0]))
		return last_action[0][0]
	else:
		#print('invalid action')
		return -1


def next_reward(reward, end_episode):
	global last_state
	global curr_state
	global last_action
	global curr_diff
	global last_diff
	global lstm_actor_hx
	global lstm_actor_cx

	#print('reward = ' + str(reward))
	reward = Tensor([reward])
	
	if last_diff is not None and curr_diff is not None and last_action is not None:	
		# store the transition in memory
		#memory.push(last_diff, last_action, curr_diff, reward)
		memory.push(last_state, last_action, curr_state, reward)

		#if end_episode:
		#	memory.push(curr_diff, last_action, None, reward)

		# perform one step of optimization on the target network
		optimize_model()

	if end_episode:
		last_state = None
		curr_state = None
		last_action = None
		curr_diff = None
		last_diff = None

		if use_lstm:
			lstm_actor_hx, lstm_actor_cx = model.reset_states(lstm_actor_hx, lstm_actor_cx)


