
print('[deepRL]  test-DQN.lua is running (time=' .. os.clock() .. ')')

print('[deepRL]  loading cuTorch...')
require 'cutorch'

print('[deepRL]  loading DQN package...')
local Brain = require 'DQN'


--
-- initialize a network with a specified number of input states and output actions
--
function init_network( num_inputs, num_actions )

	print('[deepRL]  init_network(' .. num_inputs .. ', ' .. num_actions .. ') (time=' .. os.clock() .. ')')

	Brain.init(num_inputs, num_actions)

	print('[deepRL]  network initialization complete (time=' .. os.clock() .. ')')

end


--
-- given the input state, compute the best action to follow.
--
function forward( input_tensor, action_tensor )
	-- state_tensor.size=' .. input_tensor.size(1) .. '  
	--print('[deepRL]  forward()  (time=' .. os.clock() .. ')')
	--print(input_tensor)
	--print('calling brain.forward()')
	input_array = {}
	
	for i=1, input_tensor:size(1) do
		input_array[i] = input_tensor[i]
	end
	
	action = Brain.forward(input_array)

	print('[deepRL]  done forward() action=' .. action .. ' (time=' .. os.clock() .. ')')

	
	--print(action_tensor[1])
	--action_tensor[1] = action
	
	print('exiting forward function')
	return action
	--collectgarbage()
end


--
-- apply a reward and learning
--
function backward( reward )

	print('inside backward function')
	
	--reward = reward_tensor[1]
	
	print('[deepRL]  backward( ' .. reward .. ' )  (time=' .. os.clock() .. ')')

	--Brain.backward(reward)
	Brain.backward(0)
	print('[deepRL]  done backward()  (time=' .. os.clock() .. ')')

end

