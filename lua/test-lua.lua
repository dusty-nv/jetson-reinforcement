--
-- deepRL - environment verification script
--

print('[deepRL]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')
print('[deepRL]  loading Lua packages...')

print('[deepRL]  loading torch...')
require 'torch'

print('[deepRL]  loading cutorch...')
require 'cutorch'

print('[deepRL]  loading nn...')
require 'nn'

print('[deepRL]  loading cudnn...')
require 'cudnn'

print('[deepRL]  loading math...')
require 'math'

print('[deepRL]  loading nnx...')
require 'nnx'

print('[deepRL]  loading optim...')
require 'optim'

print('[deepRL]  done loading packages. (time=' .. os.clock() .. ')')
