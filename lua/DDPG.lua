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

--
-- Pendulum
--

-- game = {}
game.init_state = {0, 0} -- 1 angular velocity, 2 angle
game.target_state = {0, math.pi} -- 1 angular velocity, 2 angle

function game.control(s_t, a_t)
    -- Control input
    local f = {[1] = a_t * opt.action_lim}
    -- 1 angular velocity, 2 angle
    local s_t1 = control_sp(s_t, opt.dt, f)[1]
    -- Correct angle
    s_t1[2] = s_t1[2] % (2 * math.pi)
    return s_t1
end

function game.reward(s_t, a_t)
    -- distance reward
    local dist = (math.pi - math.abs(game.target_state[2] - (s_t[2] % (2*math.pi)))) / math.pi
    
    -- bonus reward if +-10 degrees
    local bonus = 0
    if dist > 1 - 0.174532925 / math.pi then
        bonus = 2
    end

    -- action cost
    local action = 0
    if a_t then
        action = - a_t:clone():abs():sum()*0.5
    end
    
    return dist + action + bonus
end

function game.generate_dataset(e)
    local data = {}

    -- Iterate episodes
    for episode = 1, e do
        
        -- Initialise Variables
        local s, x, a, r = {}, {}, {}, {}
        
        -- Receive initial observation state s_1    
        s[1], x[1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)
        for i = 1, opt.action_repeat do
            s[1][i] = tools.dc(game.init_state)
            x[1][i] = draw_sp(s[1][i][2], opt.img_size)
        end
        
        -- Run episode
        local err_episode = 0
        for t = 1, opt.episode_length do
            
            -- Pick random action
            a[t] = torch.randn(opt.action_size):mul(opt.exploration_noise):clamp(-1, 1)
            
            -- Execute action a_t and observe reward r_t and observe new state s_{t+1}
            s[t+1], x[t+1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)
            
            s[t+1][1] = game.control(s[t][opt.action_repeat], a[t])
            x[t+1][1] = draw_sp(s[t+1][1][2], opt.img_size)
            for i = 2, opt.action_repeat do
                s[t+1][i] = game.control(s[t+1][i-1], a[t])
                x[t+1][i] = draw_sp(s[t+1][i][2], opt.img_size)
            end
            
            -- Get reward
            -- r[t] = game.reward(s[t+1][opt.action_repeat])
            r[t] = 0
            for i=1,opt.action_repeat do
                 r[t] = r[t] + game.reward(s[t+1][i], a[t]) / opt.action_repeat
            end
            
            -- Store transition (s_t, a_t, r_t, s_{t+1}) in R
            data[#data+1] = {
                                s_t=s[t],
                                x_t=x[t],
                                a_t=a[t],
                                r_t=r[t],
                                s_t_next=s[t+1],
                                x_t_next=x[t+1]
                            }
        end    
    end
            
    return data
end


--
-- Actor / Critic Model
--
function create_network()
        
    -- Model Specific parameters
    local f_maps_1 = 32
    local f_size_1 = 4
    local f_stride_1 = 4
    
    local f_maps_2 = 32
    local f_size_2 = 4
    local f_stride_2 = 2
    
    local f_maps_3 = 32
    local f_size_3 = 3
    local f_stride_3 = 1
    
    local enc_size = f_maps_3*5*5
    local latent_size = 100
    
    -- Encoder
    encoder = nn.Sequential()
    encoder:add(nn.View(-1, opt.action_repeat, opt.img_size, opt.img_size))
    
    encoder:add(nn.SpatialBatchNormalization(opt.action_repeat))
    
    encoder:add(nn.SpatialConvolution(opt.action_repeat, f_maps_1, f_size_1, f_size_1, f_stride_1, f_stride_1))
    encoder:add(nn.GradientClipping())
    encoder:add(nn.SpatialBatchNormalization(f_maps_1))
    encoder:add(nn.ReLU(true))
    
    encoder:add(nn.SpatialConvolution(f_maps_1, f_maps_2, f_size_2, f_size_2, f_stride_2, f_stride_2))
    encoder:add(nn.GradientClipping())
    encoder:add(nn.SpatialBatchNormalization(f_maps_2))
    encoder:add(nn.ReLU(true))
    
    encoder:add(nn.SpatialConvolution(f_maps_2, f_maps_3, f_size_3, f_size_3, f_stride_3, f_stride_3))
    encoder:add(nn.GradientClipping())
    encoder:add(nn.SpatialBatchNormalization(f_maps_3))
    encoder:add(nn.ReLU(true))
    
    encoder:add(nn.View(-1, enc_size))
    
    
    -- Actor
    actor = nn.Sequential()
    actor:add(encoder:clone()) -- ('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std')
    
    actor:add(nn.Linear(enc_size, latent_size))
    actor:add(nn.GradientClipping())
    actor:add(nn.BatchNormalization(latent_size))
    actor:add(nn.ReLU(true))
    
    actor:add(nn.Linear(latent_size, latent_size))
    actor:add(nn.GradientClipping())
    actor:add(nn.BatchNormalization(latent_size))
    actor:add(nn.ReLU(true))
    
    actor:add(nn.Linear(latent_size, opt.action_size))
    actor:add(nn.GradientClipping())

    actor:add(nn.View(-1, opt.action_size))
    actor:add(nn.Clamp(-1, 1))
    
    
    -- Critic
    critic_encoder = encoder:clone() -- 'weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std'
    critic_encoder:add(nn.Linear(enc_size, latent_size))
    critic_encoder:add(nn.GradientClipping())
    critic_encoder:add(nn.BatchNormalization(latent_size))
    
    critic_action = nn.Sequential()
    critic_action:add(nn.View(-1, opt.action_size))
    critic_action:add(nn.Linear(opt.action_size, latent_size))
    critic_action:add(nn.GradientClipping())
    
    critic_split = nn.ParallelTable()
    critic_split:add(critic_encoder)
    critic_split:add(critic_action)
    
    critic = nn.Sequential()
    critic:add(critic_split)
    critic:add(nn.CAddTable())
    critic:add(nn.GradientClipping())
    critic:add(nn.ReLU(true))

    critic:add(nn.Linear(latent_size, latent_size))
    critic:add(nn.GradientClipping())
    critic:add(nn.ReLU(true))
    
    critic:add(nn.Linear(latent_size, 1))
    critic:add(nn.View(-1, 1))
    
    
    return actor, critic
end


function setup()
    print("Creating network.")
    
    model = {}
    model.replay_buffer = {}
    
    model.actor, model.critic = create_network()
    
    -- Cuda
    model.actor = model.actor:type(opt.dtype)
    model.critic = model.critic:type(opt.dtype)
    if opt.cuda > 0 then
        cudnn.convert(model.actor, cudnn)
        cudnn.convert(model.critic, cudnn)
    end
            
    -- Initialise last layer parameters
    model.actor.modules[#model.actor.modules-3].weight:uniform(-3e-4, 3e-4)
    model.actor.modules[#model.actor.modules-3].bias:zero()
    model.critic.modules[#model.critic.modules-1].weight:uniform(-3e-4, 3e-4)
    model.critic.modules[#model.critic.modules-1].bias:zero()
    
    -- Target networks
    model.actor_target = model.actor:clone('running_mean', 'running_std')
    model.critic_target = model.critic:clone('running_mean', 'running_std')
    
    -- Get parameters
    params, gradParams = {}, {}
    params.actor, gradParams.actor = model.actor:getParameters()
    params.actor_target, _ = model.actor_target:getParameters()
    params.critic, gradParams.critic = model.critic:getParameters()
    params.critic_target, _ = model.critic_target:getParameters()
    
    -- Set criterion
    criterion = nn.MSECriterion():type(opt.dtype)

end

function save_model()
    -- save/log current net
    local filename = paths.concat(opt.save, 'model/ddpg.t7')
    os.execute('mkdir -p ' .. paths.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    -- print('<trainer> saving network to '..filename)
    torch.save(filename, {model, opt, optim_config, train_err, test_r})
end

function load_model()
    model, opt, optim_config, train_err, test_r = unpack(torch.load('model/ddpg.t7'))
end


--
-- initialize DDPG
--
print("DDPG parameters:")
print(opt)

setup()

-- Training configuration
optim_config = {}
optim_config.actor = {
                        learningRate = opt.learningRate_actor,
                        beta2 = 0.9
                        }
optim_config.critic = {
                        learningRate = opt.learningRate_critic,
                        beta2 = 0.9
                        }


-- Performance tables
stats = {
    train_err = {},
    test_r = {}
}


if opt.load == 1 then
    load_model()
end

epoch = #stats.train_err


--
-- train function
--

-- batch function
batch = {}

batch.s_t = {}
batch.x_t = torch.Tensor(opt.batch_size, opt.action_repeat, opt.img_size, opt.img_size):type(opt.dtype):zero()
batch.a_t = torch.Tensor(opt.batch_size, opt.action_size):type(opt.dtype):zero()
batch.r_t = torch.Tensor(opt.batch_size, 1):type(opt.dtype):zero()

batch.s_t_next = {}
batch.x_t_next = torch.Tensor(opt.batch_size, opt.action_repeat, opt.img_size, opt.img_size):type(opt.dtype):zero()

gradCritic = torch.Tensor(opt.batch_size, 1):type(opt.dtype):fill(-1)


function train()
    
    -- Collect garbage
    collectgarbage()
        
    -- Epoch tracker
    epoch = epoch or 0
    
    -- Set model to training mode
    model.actor:training()
    model.critic:training()
    model.actor_target:training()
    model.critic_target:training()
    
    -- Error
    local err = {}
    
    -- Sample a random minibatch of N transitions (si , ai , ri , si+1) from R
    local batch_idx = torch.randperm(#model.replay_buffer)[{{1,opt.batch_size}}]:long()
    for i=1,opt.batch_size do        
        local pair = model.replay_buffer[batch_idx[i]]

        batch.s_t[i] = pair.s_t
        batch.x_t[i] = pair.x_t:type(opt.dtype)
        batch.a_t[i] = pair.a_t:type(opt.dtype)
        batch.r_t[i] = pair.r_t
        batch.s_t_next[i] = pair.s_t_next
        batch.x_t_next[i] = pair.x_t_next:type(opt.dtype)
        
    end

    -- Set y_i = r_i + γ Q'(s_{i+1}, μ'(s_{i+1}|θ^μ') | Θ^Q')
    local batch_mu_target = model.actor_target:forward(batch.x_t_next)
    local batch_q_target = model.critic_target:forward({batch.x_t_next, batch_mu_target})
    local batch_y = batch_q_target:clone():mul(opt.gamma):add(batch.r_t)
        
    -- Update critic MSE: L = 1/N(y_i − Q(s_i, a_i|θ^Q))^2
    local feval_critic = function(x)
        
        -- Zero gradients
        model.critic:zeroGradParameters()
        
        -- Forward pass
        local pred_q = model.critic:forward({batch.x_t, batch.a_t})
        err.critic = criterion:forward(pred_q, batch_y)
        
        -- Backward pass
        local d_err_q = criterion:backward(pred_q, batch_y)
        model.critic:backward({batch.x_t, batch.a_t}, d_err_q)
        
        -- L2 Weight decay
        if opt.l2_critic > 0 then
            local l2 = opt.l2_critic * params.critic:norm(2)^2 / 2
            gradParams.critic:add(opt.l2_critic, params.critic)
            err.critic = err.critic + l2
        end
        
        -- Clip gradients
        gradParams.critic:clamp(-5, 5)
                        
        return err.critic, gradParams.critic
    end
    
    optim.adam(feval_critic, params.critic, optim_config.critic)

    
    -- Update the actor policy using the sampled gradient
    local feval_actor = function(x)
        
        -- Zero gradients
        model.actor:zeroGradParameters()
        
        -- Forward pass
        local pred_mu = model.actor:forward(batch.x_t)
        local pred_q = model.critic:forward({batch.x_t, pred_mu})
        
        -- Backward pass
        local d_a_critic = model.critic:updateGradInput({batch.x_t, pred_mu}, gradCritic)[2]
        model.actor:backward(batch.x_t, d_a_critic)
        
        -- Normalise
        gradParams.actor:div(opt.batch_size)
                
        -- L2 Weight decay
        if opt.l2_actor > 0 then
            local l2 = opt.l2_actor * params.actor:norm(2)^2 / 2
            gradParams.actor:add(opt.l2_actor, params.actor)
        end
        
        -- Clip gradients
        gradParams.actor:clamp(-5, 5)
        
        return 0, gradParams.actor
    end

    optim.adam(feval_actor, params.actor, optim_config.actor)
    
    
    -- Update target networks
    params.actor_target:mul(1 - opt.q_tau):add(opt.q_tau, params.actor)
    params.critic_target:mul(1 - opt.q_tau):add(opt.q_tau, params.critic)
    
    epoch = epoch + 1

    return err.critic
end

--
-- Test function
--
function test(v)
    
    -- Set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model.actor:evaluate()
    model.critic:evaluate()
    model.actor_target:evaluate()
    model.critic_target:evaluate()
    
    local total_reward = torch.Tensor(opt.episode_length)
    
    -- Receive initial observation state s_1
    local s, x, a, r = {}, {}, {}, {}
    
    s[1], x[1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)

    for i = 1, opt.action_repeat do
        s[1][i] = tools.dc(game.init_state)
        x[1][i] = draw_sp(s[1][i][2], opt.img_size)
    end

    -- Run episode
    for t = 1, opt.episode_length do
        
        a[t] = model.actor_target:forward(x[t]:type(opt.dtype))[1]:clone()
        -- a[t] = model.actor:forward(x[t]:type(opt.dtype))[1]:clone()
        
        -- Execute action a_t and observe reward r_t and observe new state s_{t+1}
        s[t+1], x[t+1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)   
        s[t+1][1] = game.control(s[t][opt.action_repeat], a[t])
        x[t+1][1] = draw_sp(s[t+1][1][2], opt.img_size)
        for i = 2, opt.action_repeat do
            s[t+1][i] = game.control(s[t+1][i-1], a[t])
            x[t+1][i] = draw_sp(s[t+1][i][2], opt.img_size)
        end
                
        -- Get reward
        r[t] = 0
        for i=1,opt.action_repeat do
             r[t] = r[t] + game.reward(s[t+1][i], a[t]) / opt.action_repeat
        end
        
        -- Accumulate rewards
        total_reward[t] = r[t] 
        
        if v then
            itorch.image(x[t+1])
            log.infof('step=%d, q=%.3f, r=%.2f, a=%.2f',
                t,
                critic:forward({x[t]:type(opt.dtype), a[t]})[1][1],
                r[t],
                a[t][1])
        end
    end

    
    return total_reward
end


--
-- Generate and Train on initial data
--

model.replay_buffer = {}

episodes_random = 20

-- Generate random episodes
model.replay_buffer = game.generate_dataset(episodes_random)

-- Train for episodes_random * opt.episode_length epochs
for i = 1, episodes_random * opt.episode_length do
    
    local err = train()
    
    if i % 100 == 0 then
        log.infof('pre-training iter=%d, train_mse=%.3f', i, err)
    end
end


--
-- Learn function
--

-- start time
local beginning_time = torch.tic()

-- Iterate episodes
for episode = 1, opt.episodes do
    
    -- Initialise Variables
    local time = sys.clock()
    local s, x, a, r = {}, {}, {}, {}    
    
    -- Initialise a random process N for action exploration
    local N_t = torch.randn(opt.episode_length, opt.action_size):mul(opt.exploration_noise):type(opt.dtype)
    
    -- Receive initial observation state s_1    
    s[1], x[1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)
    for i = 1, opt.action_repeat do
        s[1][i] = tools.dc(game.init_state)
        x[1][i] = draw_sp(s[1][i][2], opt.img_size)
    end
    
    -- Run episode
    local err_episode = 0
    for t = 1, opt.episode_length do
        
        -- Select action a_t = μ(s_t|θ^μ) + N_t according to the current policy and exploration noise
        model.actor:evaluate()
        a[t] = model.actor:forward(x[t]:type(opt.dtype))[1]:clone():add(N_t[t]):clamp(-1, 1)
        
        -- Execute action a_t and observe reward r_t and observe new state s_{t+1}
        s[t+1], x[t+1] = {}, torch.Tensor(opt.action_repeat,opt.img_size,opt.img_size)
        s[t+1][1] = game.control(s[t][opt.action_repeat], a[t])
        x[t+1][1] = draw_sp(s[t+1][1][2], opt.img_size)
        for i = 2, opt.action_repeat do
            s[t+1][i] = game.control(s[t+1][i-1], a[t])
            x[t+1][i] = draw_sp(s[t+1][i][2], opt.img_size)
        end
        
        -- Get reward
        r[t] = 0
        for i=1,opt.action_repeat do
             r[t] = r[t] + game.reward(s[t+1][i], a[t]) / opt.action_repeat
        end
        
        -- Store transition (s_t, a_t, r_t, s_{t+1}) in R
        local pair = {
                        s_t=s[t],
                        x_t=x[t],
                        a_t=a[t],
                        r_t=r[t],
                        s_t_next=s[t+1],
                        x_t_next=x[t+1]
                      }
        
        -- Or replace existing if size exceeded
        if #model.replay_buffer < opt.replay_buffer then
            model.replay_buffer[#model.replay_buffer+1] = pair
        else
            local idx_rand = math.round(torch.uniform(1,#model.replay_buffer))
            model.replay_buffer[idx_rand] = pair
        end
        
        -- Train with experience replay
        if #model.replay_buffer >= opt.batch_size * opt.epochs_per_step then
            for e=1, opt.epochs_per_step do
                local err_train = train()
                err_episode = err_episode + err_train / (opt.episode_length * opt.epochs_per_step)
            end
        end
        
    end
    
    -- Test
    local rewards_test = test()
    
    -- Update stats
    table.insert(stats.train_err, err_episode)
    table.insert(stats.test_r, rewards_test)
    
    if episode == 1 then
        stats.train_err_avg = err_episode
        stats.test_r_avg = rewards_test:mean()
    else
        stats.train_err_avg = stats.train_err_avg * 0.95 + 0.05 * err_episode
        stats.test_r_avg = stats.test_r_avg * 0.95 + 0.05 * rewards_test:mean()
    end
    
    -- Statistics
    if episode % 10 == 0 then        
        log.infof('e=%d, train_mse=%.3f, train_mse_avg=%.3f, test_r=%.3f, test_r_avg=%.3f, t/e=%.2f sec, t=%d min.',
            #stats.train_err,
            stats.train_err[#stats.train_err],
            stats.train_err_avg,
            stats.test_r[#stats.test_r]:mean(),
            stats.test_r_avg,
            sys.clock() - time,
            torch.toc(beginning_time) / 60)
        
        collectgarbage()
    end
    
    -- Save model
    if episode % 100 == 0 then
        save_model()
    end    
    
end


--
-- plot
--
plot = itorch.Plot()
plot_colors = {'blue', 'green', 'red', 'purple', 'orange', 'magenta', 'cyan'}
plot:title('DDPG Performance')
plot:legend(true)

plot:line(torch.range(1, #stats.test_r), torch.ones(#stats.test_r):add(2), '#000000', 'Optimal'):redraw()

local line_test = torch.Tensor(#stats.test_r)
for i=1,#stats.test_r, 1 do
    line_test[i] = stats.test_r[i]:mean()
end

plot:line(torch.range(1, #stats.test_r), line_test, plot_colors[2], 'Test Reward'):redraw()


