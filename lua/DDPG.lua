if 1 then		-- change to zero if your environment (i.e. th REPL)
    arg = {}	-- sets the command line to lua 'arg' global
end

require 'torch'

cmd = torch.CmdLine()

cmd:text()
cmd:text('Options')

-- general options:
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 4, 'number of threads')	-- 8

-- gpu
cmd:option('-cuda', 0, 'cuda')

-- experiment
cmd:option('-action_size', 1, 'action dimensionality')
cmd:option('-action_lim', 10, 'action clipping')
cmd:option('-action_repeat', 3, 'frames in history')
cmd:option('-exploration_noise', 0.3, 'random exploration noise')
cmd:option('-episodes', 1000, 'number of episodes')
cmd:option('-episode_length', 30, 'frames of each episode')

-- model
cmd:option('-gamma', 0.99, 'discount factor')
cmd:option('-q_tau', 1e-2, 'soft target updates')
cmd:option('-dt', 0.1, 'time interval')
cmd:option('-img_size', 64, 'image size')
cmd:option('-batch_size', 16, 'batch training size')
cmd:option('-replay_buffer', 1e+6, 'replay buffer size')

-- training
cmd:option('-epochs_per_step', 1, 'epochs per step')
cmd:option('-l2_critic', 1e-2, 'l2 weight decay')
cmd:option('-l2_actor', 0, 'l2 weight decay')
cmd:option('-learningRate_actor', 1e-4, 'actor learning rate')
cmd:option('-learningRate_critic', 1e-3, 'critic learning rate')

-- get current path
require 'sys'
dname, fname = sys.fpath()
cmd:option('-save', dname, 'save path')
cmd:option('-load', 0, 'load pretrained model')

cmd:option('-plot', 0, 'display images')
cmd:text()

opt = cmd:parse(arg)

print('hello from DDPG')

require 'torch'
require 'nngraph'
require 'optim'
require 'nn'

print('packages loaded')

log = require 'DDPG/log'
tools = require 'DDPG/base'

require 'DDPG/LinearO'
require 'DDPG/GradientClipping'
require 'DDPG/GradientNormalization'

print('modules loaded')

torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

print('Torch platform defaults set')

-- CUDA initialization
if opt.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.cuda)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(opt.cuda))
else
    opt.dtype = 'torch.FloatTensor'
end

print('DDPG script initialized')

