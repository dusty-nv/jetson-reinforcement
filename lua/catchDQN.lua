--[[
            Torch translation of the keras example found here (written by Eder Santana).
            https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

            Example of Re-inforcement learning using the Q function described in this paper from deepmind.
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

            The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
            left/stay/right to catch the fruit before it reaches the ground.
]] --

require 'nn'
require 'optim'

math.randomseed(os.time())

--[[ Helper function: Chooses a random value between the two boundaries.]] --
local function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

--[[ The environment: Handles interactions and contains the state of the environment]] --
function CatchEnvironment(gridSize)
    local env = {}
    local state
    -- Returns the state of the environment.
    function env.observe()
        local canvas = env.drawState()
        canvas = canvas:view(-1)
        return canvas
    end

    function env.drawState()
        local canvas = torch.Tensor(gridSize, gridSize):zero()
        canvas[state[1]][state[2]] = 1 -- Draw the fruit.
        -- Draw the basket. The basket takes the adjacent two places to the position of basket.
        canvas[gridSize][state[3] - 1] = 1
        canvas[gridSize][state[3]] = 1
        canvas[gridSize][state[3] + 1] = 1
        return canvas
    end

    -- Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
    function env.reset()
        local initialFruitColumn = math.random(1, gridSize)
        local initialBucketPosition = math.random(2, gridSize - 1)
        state = torch.Tensor({ 1, initialFruitColumn, initialBucketPosition })
        return env.getState()
    end

    function env.getState()
        local stateInfo = state
        local fruit_row = stateInfo[1]
        local fruit_col = stateInfo[2]
        local basket = stateInfo[3]
        return fruit_row, fruit_col, basket
    end

    -- Returns the award that the agent has gained for being in the current environment state.
    function env.getReward()
        local fruitRow, fruitColumn, basket = env.getState()
        if (fruitRow == gridSize - 1) then -- If the fruit has reached the bottom.
        if (math.abs(fruitColumn - basket) <= 1) then -- Check if the basket caught the fruit.
        return 1
        else
            return -1
        end
        else
            return 0
        end
    end

    function env.isGameOver()
        if (state[1] == gridSize - 1) then return true else return false end
    end

    function env.updateState(action)
        if (action == 1) then
            action = -1
        elseif (action == 2) then
            action = 0
        else
            action = 1
        end
        local fruitRow, fruitColumn, basket = env.getState()
        local newBasket = math.min(math.max(2, basket + action), gridSize - 1) -- The min/max prevents the basket from moving out of the grid.
        fruitRow = fruitRow + 1 -- The fruit is falling by 1 every action.
        state = torch.Tensor({ fruitRow, fruitColumn, newBasket })
    end

    -- Action can be 1 (move left) or 2 (move right)
    function env.act(action)
        env.updateState(action)
        local reward = env.getReward()
        local gameOver = env.isGameOver()
        return env.observe(), reward, gameOver, env.getState() -- For purpose of the visual, I also return the state.
    end

    return env
end

--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
local function Memory(maxMemory, discount)
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
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
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

function Main()
    print("Training new model")
    local epsilon = 1 -- The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
    local epsilonMinimumValue = 0.001 -- The minimum value we want epsilon to reach in training. (0 to 1)
    local nbActions = 3 -- The number of actions. Since we only have left/stay/right that means 3 actions.
    local epoch = 1000 -- The number of games we want the system to run for.
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

            local nextState, reward, gameOver = env.act(action)

		   io.write(string.char(27) .. "[r")
		   for envX=1,gridSize+2 do
				io.write("-")
		   end
		   io.write("\n")

		   for envY=1,gridSize do
			  io.write("|")
			  for envX=1,gridSize do
				local envCell = nextState[(envY-1) * gridSize + envX]

				if envCell == 1 then
					io.write("*")
				else
					io.write(" ")
				end
			  end
			  io.write("|\n")
		   end
	   
		   for envX=1,gridSize+2 do
				io.write("-")
		   end
		   io.write("\n\n")

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

		local winPercentage = (winCount/i) * 100
		local winHistorySum = winHistory:sum()

          io.write(string.format("Epoch %03d : err = %f : wins %d (%.1f%%) ", i, err, winCount, winPercentage))

		if i > winHistorySize then
			io.write(string.format(": last%d - %d wins (%.1f%%)\n", winHistorySize, winHistorySum, (winHistorySum/winHistorySize) * 100))
		else
			io.write("\n")
		end
    end
    torch.save("TorchQLearningModel.model", model)
    print("Model saved")
end
--print("Call the Main() function at the end of the TorchQLearningExample.lua file to train a new model")
Main()
