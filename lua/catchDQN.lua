--
-- http://github.com/dusty-nv/jetson-reinforcement
--

require 'nn'
require 'optim'
require 'gnuplot'
require 'catchENV'

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
Main()
