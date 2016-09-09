--
-- http://github.com/dusty-nv/jetson-reinforcement
--

math.randomseed(os.time())

--[[ Helper function: Chooses a random value between the two boundaries.]] --
function randf(s, e)
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

function printEnv( nextState, gridSize )
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
end

