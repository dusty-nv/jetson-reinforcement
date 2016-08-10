require 'math'
require 'nnx'
require 'os'
require 'optim'


math.randomseed( os.time() )
torch.setdefaulttensortype('torch.FloatTensor')

local Brain = {}

--[[  HELPER FUNCTIONS --]]
 
function randf(s, e) 
  return (math.random(0,(e-s)*9999)/10000) + s;
end

-- new methods for table

function table.merge(t1, t2)
   local t = t1
    for i = 1, #t2 do
        t[#t+1] = t2[i]
    end
    return t
end

function table.copy(t)
  local u = { }
  for k, v in pairs(t) do u[k] = v end
  return setmetatable(u, getmetatable(t))
end

function table.length(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- BRAIN

function Brain.init(num_states, num_actions)
   -- Number of past state/action pairs input to the network. 0 = agent lives in-the-moment :)
   Brain.temporal_window = 0  
   -- Maximum number of experiences that we will save for training
   Brain.experience_size = 30000     
   -- experience necessary to start learning
   Brain.start_learn_threshold = 300
   -- gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
   -- Determines the amount of weight placed on the utility of the state resulting from an action.
   Brain.gamma = 0.9;
   -- number of steps we will learn for
   Brain.learning_steps_total = 100000
   -- how many steps of the above to perform only random actions (in the beginning)?
   Brain.learning_steps_burnin = 300;
   -- controls exploration exploitation tradeoff. Will decay over time
   -- a higher epsilon means we are more likely to choose random actions
   Brain.epsilon = 1.0
   -- what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
   Brain.epsilon_min = 0.05;
   -- what epsilon to use when learning is turned off. This is for testing
   Brain.epsilon_test_time = 0.01;

    --[[ states and actions that go into neural net:
    	 (state0,action0),(state1,action1), ... , (stateN)
     	this variable controls the size of that temporal window.
     --]]
   Brain.net_inputs = (num_states + num_actions) * Brain.temporal_window + num_states;
   Brain.hidden_nodes = 16
   Brain.num_states = num_states;
   Brain.num_actions = num_actions;
   Brain.net_outputs = Brain.num_actions;
   
    --[[ Window size dictates the number of states, actions, rewards, and net inputs that we
    	save. The temporal window size is the number of time states/actions that are input
    	to the network and must be smaller than or equal to window_size
	--]]
   Brain.window_size = math.max(Brain.temporal_window, 2);

   -- advanced feature. Sometimes a random action should be biased towards some values
   -- for example in flappy bird, we may want to choose to not flap more often
   Brain.random_action_distribution = {};
    if(table.length(Brain.random_action_distribution) > 0) then
      -- this better sum to 1 by the way, and be of length this.num_actions
      if(table.length(Brain.random_action_distribution) ~= Brain.num_actions) then
        print('TROUBLE. random_action_distribution should be same length as num_actions.');
      end
      
      local s = 0.0;
      
      for k = 1, table.length(Brain.random_action_distribution) do
         s = s + Brain.random_action_distribution[k]
      end
      
      if(math.abs(s - 1.0) > 0.0001) then
         print('TROUBLE. random_action_distribution should sum to 1!');
      end
    end
    

   -- define architecture
   Brain.net = nn.Sequential()

   Brain.net:add(nn.Linear(Brain.net_inputs, Brain.hidden_nodes))
   Brain.net:add(nn.Threshold(0,0))

   Brain.net:add(nn.Linear(Brain.hidden_nodes, Brain.hidden_nodes))
   Brain.net:add(nn.Threshold(0,0))

   Brain.net:add(nn.Linear(Brain.hidden_nodes, Brain.net_outputs))
   
   Brain.criterion = nn.MSECriterion()
   
   
   -- other learning parameters
   Brain.learning_rate = 0.01;
   Brain.learning_rate_decay = 5e-7
   Brain.batch_size = 16;
   Brain.momentum = 0.9;
      
   -- various housekeeping variables
   Brain.age = 0; -- incremented every backward()
   
   -- number of times we've called forward - lets us know when our input temporal
   -- window is filled up
   Brain.forward_passes = 0;
   Brain.learning = true;

	-- coefficients for regression
	Brain.coefL1 = 0.001
	Brain.coefL2 = 0.001

	-- parameters for optim.sgd
	Brain.parameters, Brain.gradParameters = Brain.net:getParameters()
	
	local exp_table_size = (Brain.experience_size + 1) * (Brain.net_inputs * 2 + 2)
	io.write(string.format('\nAllocating %.2f GB for experience table...\n\n', (4 * exp_table_size)/(1024^3)))
	-- experience table
	Brain.experience = torch.Tensor(exp_table_size)
	-- tracks number of experiences input into the experience table
	Brain.eCount = 0
	-- These windows track old experiences, states, actions, rewards, and net inputs
	-- over time. They should all start out as empty with a fixed size.
	-- This is a first in, last out data structure that is shifted along time
   Brain.state_window = {}
   Brain.action_window = {}
   Brain.reward_window = {}
   Brain.net_window = {}
   for i = 1, Brain.window_size do  
      Brain.state_window[i] = {}
      Brain.action_window[i] = {}
      Brain.reward_window[i] = {}
      Brain.net_window[i] = {}
   end
end

   -- a bit of a helper function. It returns a random action
   -- we are abstracting this away because in future we may want to 
   -- do more sophisticated things. For example some actions could be more
   -- or less likely at "rest"/default state.
function Brain.random_action()
	-- if we don't have a random action distribution defined then sample evenly
   if(table.length(Brain.random_action_distribution) == 0) then
   	return (torch.random() % Brain.net_outputs) + 1
   	
      -- okay, lets do some fancier sampling:
   else 
      local p = randf(0, 1);
      local cumprob = 0.0;

      for k= 1, Brain.num_actions do
        cumprob = cumprob + Brain.random_action_distribution[k];
        
        if(p < cumprob) then
         return k
        end
      end
   end
end

  -- compute the value of doing any action in this state
  -- and return the argmax action and its value
function Brain.policy(state)   
  local action_values = Brain.net:forward(state);
  
  local maxval = action_values[1]
  local max_index = 1
 
 -- find maximum output and note its index and value
  for i = 2, Brain.net_outputs do
  	if action_values[i] > maxval then
  		maxval = action_values[i]
  		max_index = i
  	end
  end
  
  return {action = max_index, value = maxval};
end
    
-- This function assembles the input to the network by concatenating
-- old (state, chosen_action) pairs along with the current state      
  -- return s = (x,a,x,a,x,a,xt) state vector. 
function Brain.getNetInput(xt) 
  local w = {};
  w = table.merge(w, xt); -- start with current state
  
  -- and now go backwards and append states and actions from history temporal_window times
  local n = Brain.window_size + 1; 
  for k = 1, Brain.temporal_window do
    -- state
    w = table.merge(w, Brain.state_window[n-k]);
    -- action, encoded as 1-of-k indicator vector. We scale it up a bit because
    -- we dont want weight regularization to undervalue this information, as it only exists once
    local action1ofk = {};
    for i = 1, Brain.num_actions do
      action1ofk[i] = 0
    end

   -- assign action taken for current state to be 1, all others are 0
    action1ofk[Brain.action_window[n-k]] = 1.0*Brain.num_states;
      
    w = table.merge(w, action1ofk);
  end
  
  return w;
end
    
--[[ This function computes an action by either:
	1. Giving the current state and past (state, action) pairs to the network
		and letting it choose the best acction
	2. Choosing a random action
--]]
function Brain.forward(input_array) 
  Brain.forward_passes = Brain.forward_passes + 1;
  
  local action, net_input;
  
  --print('brain.forward()  2')
  --print(input_tensor2)
  --print(input_tensor2:size())
  --print('calling totable')
 
  --print('done input_tensor.totable()')
  
    -- if we have enough (state, action) pairs in our memory to fill up
    -- our network input then we'll proceed to let our network choose the action
  if(Brain.forward_passes > Brain.temporal_window ) then
    net_input = Brain.getNetInput(input_array);
	--print('calling torch.Tensor()')
    net_input = torch.Tensor(net_input)
    --print('done calling torch.Tensor()')
	
    -- if learning is turned on then epsilon should be decaying
    if(Brain.learning) then
      -- compute (decaying) epsilon for the epsilon-greedy policy
      local new_epsilon = 1.0 - (Brain.age - Brain.learning_steps_burnin)/(Brain.learning_steps_total - Brain.learning_steps_burnin)
      
      -- don't let epsilon go above 1.0
      Brain.epsilon = math.min(1.0, math.max(Brain.epsilon_min, new_epsilon)); 
    else
    	-- if learning is turned off then use the epsilon we've specified for testing        
      Brain.epsilon = Brain.epsilon_test_time;
    end
    
    -- use epsilon probability to choose whether we use network action or random action
    if(randf(0, 1) < Brain.epsilon) then
	  --print('calling Brain.random_action()')
      action = Brain.random_action();
	  --print('done calling Brain.random_action()')
    else
      -- otherwise use our policy to make decision
	  print('calling Brain.policy')
      local best_action = Brain.policy(net_input);
	  print('done calling Brain.policy')
      action = best_action.action; -- this is the action number
     end
  else
    -- pathological case that happens first few iterations when we can't
    -- fill up our network inputs. Just default to random action in this case
    net_input = {};
	--print('calling pathological Brain.random_action()')
    action = Brain.random_action();
	--print('done calling pathological Brain.random_action()')
  end
  
  --print('shifting table 0')
  -- shift the network input, state, and action chosen into our windows
  table.remove( Brain.net_window, 1)
  table.insert( Brain.net_window, net_input) 
	--print('shifting table 1')
  table.remove( Brain.state_window, 1)
  table.insert( Brain.state_window, input_array)      
	--print('shifting table 2')
  table.remove( Brain.action_window, 1)
  table.insert( Brain.action_window, action)
  --print('shifting table 3')
  return action;
end    
    
--[[ 
	This function trains the network using the reward resulting from the last action
	It will save this past experience which consists of:
		the state, action chosen, whether a reward was obtained, and the
	 	state that resulted from the action
	After that, it will train the network (using a batch of experiences) using a 
	random sampling of our entire experience history.
--]]
function Brain.backward(reward)
		-- add reward to our history 
	  print('Brain.backward()  reward=' .. reward )
	  
      table.remove( Brain.reward_window, 1)
      table.insert( Brain.reward_window, reward)
      
	
      -- if learning is turned off then don't do anything
      if(not Brain.learning) then 
         return; 
      end
      
	-- sizes of tensors
	local e_size
	local state0_size
	local action0_size
	local reward0_size
	local state1_size
      
      Brain.age = Brain.age + 1;
      
      -- if we've had enough states and actions to fill up our net input then add
      -- this new experience to our history
      if(Brain.forward_passes > Brain.temporal_window + 1) then
			print('Brain.backward  inside temporal window')
      	-- make experience and fill it up
        local n = Brain.window_size;
		print('n, brain.window_size=' .. n )
		
	local state0 = Brain.net_window[n-1]:clone();
	print('state0')
	state0_size = state0:size(1)
	print('state0_size')
	local action0 = torch.Tensor({Brain.action_window[n-1]});
	print('action0')
	action0_size = action0:size(1)
	print('action0_size=' .. action0_size)
	print(Brain.reward_window)
	print(Brain.reward_window[n-1])
	print('n index = ' .. n-1 )
	print('reward_window size=' .. table.getn(Brain.reward_window)) 
	--print('reward_window2 size=' .. table.getn(Brain.reward_window[n-1])) 
	print(tonumber(Brain.reward_window[n-1]))
	local reward0 = torch.Tensor({reward});
	--local reward0 = torch.Tensor({Brain.reward_window[n-1]});
	print('reward0')
	reward0_size = reward0:size(1)
	local state1 = Brain.net_window[n]:clone();
	print('state1')
	state1_size = state1:size(1)
        print('Brain.backward  torch.cat')
        local e = torch.cat({state0, action0, reward0, state1})
	e_size = e:size(1) -- experience table size
        
        -- if the number of experiences isn't larger than the max then add more
        if(Brain.eCount < Brain.experience_size) then
          Brain.experience:sub(Brain.eCount*e_size + 1, (Brain.eCount + 1)*e_size):copy(e)
	  Brain.eCount = Brain.eCount + 1 -- track number of experiences
        else 
	-- Otherwise replace random experience due to finite allocated memory for the experience table
	  local ri = torch.random(0, Brain.eCount-1);
	  Brain.experience:sub(ri*e_size + 1, (ri + 1)*e_size):copy(e)
        end
      end
      
	  print('Brain.backward pre-eCount')
	  
      -- if we have enough experience in memory then start training
     if(Brain.eCount > Brain.start_learn_threshold) then
		inputs = torch.Tensor(Brain.batch_size, Brain.net_inputs)
		targets = torch.Tensor(Brain.batch_size, Brain.net_outputs) 
	
        for k = 1, Brain.batch_size do
        	-- choose random experience
        	local re = math.random(0, Brain.eCount-1);
          	local e = torch.Tensor(Brain.experience:sub(re*e_size + 1, (re + 1)*e_size))
          
          	-- copy state from experience
          	local state0 = e:sub(1, state0_size):clone()
   
   		-- compute best action for the new state
   		local state1 = e:sub(state0_size + action0_size + reward0_size + 1, state0_size + action0_size + reward0_size + state1_size):clone()
          	
          	local best_action = Brain.policy(state1);
   
   			--[[ get current action output values
   				we want to make the target outputs the same as the actual outputs
   				expect for the action that was chose - we want to replace this with
	   			the reward that was obtained + the utility of the resulting state
   			--]]
   			local all_outputs = Brain.net:forward(state0);
		  	inputs[k] = state0:clone();      	
		  	targets[k] = all_outputs:clone();
		  	local action0 = e:sub(state0_size + 1, state0_size + action0_size)
			local reward0 = e:sub(state0_size + action0_size + 1, state0_size + action0_size + reward0_size)
		  	targets[k][action0[1]] = reward0[1] + Brain.gamma * best_action.value;   
		end

		-- create training function to give to optim.sgd
		local feval = function(x)
	     collectgarbage()

	     -- get new network parameters
	     if x ~= Brain.parameters then
	        Brain.parameters:copy(x)
	     end

	     -- reset gradients
	     Brain.gradParameters:zero()

	     -- evaluate function for complete mini batch
	     local outputs = Brain.net:forward(inputs)
	     local f = Brain.criterion:forward(outputs, targets)

	     -- estimate df/dW
	     local df_do = Brain.criterion:backward(outputs, targets)
	     Brain.net:backward(inputs, df_do)

	     -- penalties (L1 and L2):
	     if Brain.coefL1 ~= 0 or Brain.coefL2 ~= 0 then
	        -- locals:
	       local norm,sign = torch.norm,torch.sign

	        -- Loss:
	        f = f + Brain.coefL1 * norm(Brain.parameters,1)
	        f = f + Brain.coefL2 * norm(Brain.parameters,2)^2/2

	        -- Gradients:
	        Brain.gradParameters:add( sign(Brain.parameters):mul(Brain.coefL1) + Brain.parameters:clone():mul(Brain.coefL2) )
	     end

	     -- return f and df/dX
	     return f, Brain.gradParameters
	  	end

		-- fire up optim.sgd
		sgdState = {
            learningRate = Brain.learning_rate,
            momentum = Brain.momentum,
            learningRateDecay = Brain.learning_rate_decay
         }
         
         optim.sgd(feval, Brain.parameters, sgdState)
         
     end
end



-- export
return Brain




