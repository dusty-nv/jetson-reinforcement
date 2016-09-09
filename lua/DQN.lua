--
-- http://github.com/dusty-nv/jetson-reinforcement
--

require 'nn'
require 'optim'
--require 'gnuplot'
--require 'catchENV'

torch.setdefaulttensortype('torch.FloatTensor')


--[[ Runs one gradient update using SGD returning the loss.]] --
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end


hiddenSize = 100 -- Number of neurons in the hidden layers.
maxMemory  = 500 -- How large should the memory be (where it stores its past experiences).
batchSize  = 50   -- The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.


--
-- initialize a network with a specified number of input states and output actions
--
function init_network( num_inputs, num_actions )

	print('[deepRL]  init_network(' .. num_inputs .. ', ' .. num_actions .. ') (time=' .. os.clock() .. ')')

	nbStates   = num_inputs
	nbActions  = num_actions

	-- Create the base model.
	model = nn.Sequential()
    model:add(nn.Linear(nbStates, hiddenSize))
    model:add(nn.ReLU())
    model:add(nn.Linear(hiddenSize, hiddenSize))
    model:add(nn.ReLU())
    model:add(nn.Linear(hiddenSize, nbActions))
	
	criterion = nn.MSECriterion()
	
	-- Params for Stochastic Gradient Descent (our optimizer).
    sgdParams = {
        learningRate = 0.1,
        learningRateDecay = 1e-9,
        weightDecay = 0,
        momentum = 0.9,
        dampening = 0,
        nesterov = true
    }

	print('[deepRL]  network initialization complete (time=' .. os.clock() .. ')')
	print('hidden size = ' .. hiddenSize )
	print(model)

end

--init_network(360, 4)


--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
function Memory(maxMemory, discount)
    local memory = {}

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(model, batchSize, nbActions, nbStates)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
        local targets = torch.Tensor(chosenBatchSize, nbActions):zero()
        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = math.random(1, memoryLength)
            local memoryInput = memory[randomIndex]

            local target = model:forward(memoryInput.inputState)

            --Gives us Q_sa, the max q for the next state.
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + ?max a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end

    return memory
end


--
-- given the input state, compute the best action to follow.
--
function qLearning( input_tensor, reward, end_of_episode )

	print('[deepRL]  forward()  (time=' .. os.clock() .. ')')
	--print(input_tensor)
	
	-- process the neural network
	local q = model:forward(input_tensor)

	-- Find the max index (the chosen action).
	local max, index = torch.max(q, 1)
	
	print('[deepRL]  done forward() action=' .. index[1] .. ' q-value=' .. max .. ' (time=' .. os.clock() .. ')')
	
	if not last_state == nil then
		print('[deepRL]  adding memory.remember()')
		
		memory.remember({
					inputState = last_state,
					action     = last_action,
					reward     = last_reward,
					nextState  = input_tensor,
					gameOver   = last_EOE
				})
	end
	
	last_state  = input_tensor
	last_action = index[1]
	last_reward = reward
	last_EOE    = end_of_episode
	
	-- Get training batch
	local inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)

	-- Train the network which returns the error.
	print('[deepRL]  trainNetwork()  (time=' .. os.clock() .. ')')
	local err = trainNetwork(model, inputs, targets, criterion, sgdParams)
	print('[deepRL]  done trainNetwork()  err=' .. err .. ' (time=' .. os.clock() .. ')')
	
	return last_action
end




--[[
function Main()
    local epsilon = 1 -- The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
    local epsilonMinimumValue = 0.001 -- The minimum value we want epsilon to reach in training. (0 to 1)
    local nbActions = 3 -- The number of actions. Since we only have left/stay/right that means 3 actions.
    local epoch = 3000 -- The number of games we want the system to run for.
    local hiddenSize = 100 -- Number of neurons in the hidden layers.
    local maxMemory = 500 -- How large should the memory be (where it stores its past experiences).
    local batchSize = 50 -- The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
    local gridSize = 10 -- The size of the grid that the agent is going to play the game on.
    local nbStates = gridSize * gridSize -- We eventually flatten to a 1d tensor to feed the network.
    local discount = 0.9 -- The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)
 

    -- Create the base model.
    local model = nn.Sequential()
    model:add(nn.Linear(nbStates, hiddenSize))
    model:add(nn.ReLU())
    model:add(nn.Linear(hiddenSize, hiddenSize))
    model:add(nn.ReLU())
    model:add(nn.Linear(hiddenSize, nbActions))

    -- Params for Stochastic Gradient Descent (our optimizer).
    local sgdParams = {
        learningRate = 0.1,
        learningRateDecay = 1e-9,
        weightDecay = 0,
        momentum = 0.9,
        dampening = 0,
        nesterov = true
    }

    -- Mean Squared Error for our loss function.
    local criterion = nn.MSECriterion()

    local env = CatchEnvironment(gridSize)
    local memory = Memory(maxMemory, discount)

    local winCount        = 0	  -- Keep track of the total number of wins/losses
    local winHistorySize  = 50  -- The number of recent games to keep the win history for, for calulating the recent performance
    local winHistoryIndex = 1	  -- Remember which slot in the history table we used last
    local winHistory      = torch.Tensor(winHistorySize):zero()

    io.write(string.char(27) .. "[2J")

    -- configure gnuplot
    gnuplot.figure(1)
    gnuplot.xlabel("epoch")
    --gnuplot.ylabel("win ratio")
    --gnuplot.title("DQN Loss Minimization")

    local plotLabels  = torch.Tensor(1):zero()
    local plotHistory = torch.Tensor(1):zero()
    local lossHistory = torch.Tensor(1):zero()
    local plotIndex   = 1

    gnuplot.plot(plotLabels, plotHistory)
 
    -- run the epoch iterations
    for i = 1, epoch do
        -- Initialise the environment.
        local err = 0
        env.reset()
        local isGameOver = false

        -- The initial state of the environment.
        local currentState = env.observe()
	   

        while (isGameOver ~= true) do
            local action
            -- Decides if we should choose a random action, or an action from the policy network.
            if (randf(0, 1) <= epsilon) then
                action = math.random(1, nbActions)
            else
                -- Forward the current state through the network.
                local q = model:forward(currentState)
                -- Find the max index (the chosen action).
                local max, index = torch.max(q, 1)
                action = index[1]
            end
            -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
            if (epsilon > epsilonMinimumValue) then
                epsilon = epsilon * 0.999
            end

		  -- perform the actor action
            local nextState, reward, gameOver = env.act(action)

		  -- visualize the environment
		  printEnv(nextState, gridSize)

		  -- store the recent wins
		  if gameOver == true then
		       if (reward == 1) then 
				winCount = winCount + 1 
				winHistory[winHistoryIndex] = 1
			  else
				winHistory[winHistoryIndex] = 0
			  end

			  winHistoryIndex = winHistoryIndex + 1

			  if winHistoryIndex > winHistorySize then
				winHistoryIndex = 1
			  end
		  end

		  -- store state for the next training
            memory.remember({
                inputState = currentState,
                action = action,
                reward = reward,
                nextState = nextState,
                gameOver = gameOver
            })

            -- Update the current state and if the game is over.
            currentState = nextState
            isGameOver = gameOver

            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)

            -- Train the network which returns the error.
            err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)
        end

		local plotValue     = winCount/i
		local winPercentage = plotValue * 100
		local winHistorySum = winHistory:sum()

          io.write(string.format("Epoch %03d : err = %f : wins %d (%.1f%%) ", i, err, winCount, winPercentage))

		if i > winHistorySize then
			io.write(string.format(": last%d - %d wins (%.1f%%)\n", winHistorySize, winHistorySum, (winHistorySum/winHistorySize) * 100))
			plotValue = winHistorySum/winHistorySize
		else
			io.write("\n")
		end

		plotLabels[plotIndex]  = i
		plotHistory[plotIndex] = plotValue
		lossHistory[plotIndex] = err
		plotIndex = plotIndex + 1
			
		gnuplot.plot({'win ratio', plotLabels, plotHistory}, {'error rate', plotLabels, lossHistory})

		plotLabels:resize(plotIndex)
		plotHistory:resize(plotIndex)
		lossHistory:resize(plotIndex)
    end
    torch.save("catchQModel.model", model)
    print("Model saved")
end
--print("Call the Main() function at the end of the TorchQLearningExample.lua file to train a new model")
Main() ]]--
