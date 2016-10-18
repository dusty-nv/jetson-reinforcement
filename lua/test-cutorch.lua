--
-- deepRL - environment verification script
--

print('[deepRL]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')
print('[deepRL]  loading Lua packages...')

print('[deepRL]  loading torch...')
require 'torch'

print('[deepRL]  loading nn...')
require 'nn'

print('[deepRL]  loading cutorch...')
require 'cutorch'

print('cutorch.hasHalf == ' .. tostring(cutorch.hasHalf) )

if cutorch.hasHalf then
	print('half tensor test...')
	a = torch.CudaHalfTensor(3)
	b = torch.CudaHalfTensor(3)
	print(torch.cmul(a,b))
end

print('[deepRL]  loading cudnn...')
require 'cudnn'

print('[deepRL]  done loading packages. (time=' .. os.clock() .. ')')

print('[deepRL]  engaging torch test. (time=' .. os.clock() .. ')')
torch.test() 
print('[deepRL]  engaging torch test. (time=' .. os.clock() .. ')')

print('[deepRL]  engaging nn test. (time=' .. os.clock() .. ')')
nn.test() 
print('[deepRL]  engaging nn test. (time=' .. os.clock() .. ')')

print('[deepRL]  engaging cutorch test. (time=' .. os.clock() .. ')')
cutorch.test()
print('[deepRL]  finished cutorch test. (time=' .. os.clock() .. ')')

