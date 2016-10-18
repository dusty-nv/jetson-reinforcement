--
-- deepRL - environment verification script
--

print('[deepRL]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')
print('[deepRL]  loading Lua packages...')

print('[deepRL]  loading torch...')
require 'torch'

print('[deepRL]  loading nn...')
require 'nn'

print('[deepRL]  done loading packages. (time=' .. os.clock() .. ')')

print('[deepRL]  engaging torch test. (time=' .. os.clock() .. ')')
torch.test() 
print('[deepRL]  engaging torch test. (time=' .. os.clock() .. ')')

print('[deepRL]  engaging nn test. (time=' .. os.clock() .. ')')
nn.test() 
print('[deepRL]  engaging nn test. (time=' .. os.clock() .. ')')


