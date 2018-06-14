<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/jetson-reinforcement-header.jpg">

# Deep Reinforcement Learning in Robotics
In this tutorial, we'll be creating artificially intelligent agents that learn from interacting with their environment, gathering experience, and a system of rewards with deep reinforcement learning (deep RL).  Using end-to-end neural networks that translate raw pixels into actions, RL-trained agents are capable of exhibiting intuitive behaviors and performing complex tasks.  

Ultimately, our aim will be to train reinforcement learning agents from virtual robotic simulation in 3D and transfer the agent to a real-world robot.  Reinforcement learners choose the best action for the agent to perform based on environmental state (like camera inputs) and rewards that provide feedback to the agent about it's performance.  Reinforcement learning can learn to behave optimally in it's environment given a policy, or task - like obtaining the reward.

In many scenarios, the state space is significantly complex and multi-dimensional to where neural networks are increasingly used to predict the best action, which is where deep reinforcement learning and GPU acceleration comes into play.  With deep reinforcement learning, the agents are typically processing 2D imagery using convolutional neural networks (CNNs), processing inputs that are an order of magnitude more complex than low-dimensional RL, and have the ability to learn "from vision" with the end-to-end network (referred to as "pixels-to-actions").

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/nv_rl_stack_diagram.jpg">

This repository includes discrete Deep Q-Learning (DQN) and continuous A3G algorithms in PyTorch, examples and an interoperability library API in C++ for integrating with Linux applications in robotics, simulation, and deployment to the field.

### **Table of Contents**

* [Building from Source](#building-from-source)
* [Verifying PyTorch](#verifying-pytorch)
* [DQN + OpenAI Gym](#dqn--openai-gym)
	* [Cartpole](#cartpole)
	* [Lunar Lander](#lunar-lander)
* [Digging into the C++ API](#digging-into-the-c-api)
* [Testing the C++ API](#testing-the-c-api)
	* [Catch](#catch)
	* [Fruit](#fruit)
* [3D Simulation](#3d-simulation)
	* [Robotic Arm](#robotic-arm)
	* [Rover Navigation](#rover-navigation)
* [Continuous Control](#continuous-control)
* [Appendix: Using LUA](#appendix-using-lua)

> note:  stream our **[webinar](https://nvda.ws/2KdbtaO)** on the topic that follows this tutorial.

# Building from Source

Run the following commands from terminal to build from source:

``` bash
$ sudo apt-get install cmake
$ git clone http://github.com/dusty-nv/jetson-reinforcement
$ cd jetson-reinforcement
$ git submodule update --init
$ mkdir build
$ cd build
$ cmake ../
$ make
```

During the `cmake` step, PyTorch will be compiled and installed so it can take awhile (around ~30 minutes to an hour on the Jetson).  The stable version of PyTorch we are currently using is `0.3.0`.  The build script will download pacekages and ask you for your `sudo` password during the install.

# Verifying PyTorch

Before proceeding, to make sure that PyTorch installed correctly, and to get an introduction to PyTorch if you aren't already familiar, we have provided a Jupyter IPython notebook called **[`intro-pytorch.ipynb`](python/intro-pytorch.ipynb)** that includes some simple PyTorch examples that verify the install and test the CUDA/cuDNN support in PyTorch.

To launch the [notebook](python/intro-pytorch.ipynb) locally on your system, run the following commands:

``` bash
$ cd jetson-reinforcement/build/aarch64/bin   # or cd x86_64/bin on PC
$ jupyter notebook intro-pytorch.ipynb
```

Alternatively, if you wish to skip the notebook and run the PyTorch verification commands directly, you can do so by launching an interactive Python shell with the `python` command and running the following:

``` python
>>> import pytorch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
```

If PyTorch is installed correctly on your system, the output should be as follows:

``` python
Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import pytorch
>>> print(torch.__version__)
0.3.0b0+af3964a
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
CUDA available: True
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
Tensor a = 
 0
 0
[torch.cuda.FloatTensor of size 2 (GPU 0)]

>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
Tensor b = 
 0.2190
-0.3212
[torch.cuda.FloatTensor of size 2 (GPU 0)]

>>> c = a + b
>>> print('Tensor c = ' + str(c))
Tensor c = 
 0.2190
-0.3212
[torch.cuda.FloatTensor of size 2 (GPU 0)]
```

Now we have verified that PyTorch is loading, able to detect GPU acceleration, is able to allocate tensors on the GPU, and is able to perform basic tensor operations using CUDA. 


# DQN + OpenAI Gym

In order to first test and verify that the deep reinforcement learning algorithms are indeed learning, we'll run them inside OpenAI Gym environments (in 2D).  As an introduction to the DQN algorithm, a second CUDA-enabled IPython notebook is included in the repo, **[`intro-DQN.ipynb`](python/intro-DQN.ipynb)**.  This notebook applies the DQN on video captured from the Gym's [`CartPole`](https://gym.openai.com/envs/CartPole-v0/) environment, so it's learning "from vision" on the GPU, as opposed to low-dimensional parameters from the game like traditional RL.  

Although CartPole is a toy example, it's vital to start with a simple example to eliminate potential issues early on before graduating to more complex 3D scenarios that will become more difficult to debug, and since the DQN learns from a raw 2D pixel array it's still considered deep reinforcement learning.  It's recommended to follow along with the [notebook](python/intro-DQN.ipynb) below to familiarize yourself with the DQN algorithm for when we transition to using it from C++ in more complex environments later in the repo. 

## Cartpole

To launch the [notebook](python/intro-DQN.ipynb) locally from your machine, run the following commands:

``` bash
$ cd jetson-reinforcement/build/aarch64/bin   # or cd x86_64/bin on PC
$ jupyter notebook intro-DQN.ipynb
```

Inside of the notebook, the DQN is set to only run for 50 episodes.  After you have witnessed the DQN start to converge and the CartPole begin to remain upright for longer periods of time, exit the notebook and run the standalone **[`gym-DQN.py`](python/gym-DQN.py)** script from the terminal for improved performance:

``` bash
$ python gym-DQN.py
```
> (assuming the current directory of your terminal is still `jetson-reinforcement/build/<arch>/bin` from above)

Three windows should appear showing the cartpole game, a graph of peformance, and the DQN agent should begin learning.  The longer the DQN agent is able to balance the pole on the moving cart, the more points it's rewarded.  In Gym, a score of 200 indicates the scenario has been mastered.  After a short while of training, the agent should achieve it and the program will quit.

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/DQN-cartpole.png" width="400">

## Lunar Lander

Using a similar script, you can experiment in different Gym environments with the `--env` parameter:

``` bash
$ python gym-RL.py --env=LunarLander-v2 --render
```

The [`LunarLander-v2`](https://gym.openai.com/envs/LunarLander-v2/) environment is fun to explore because it's a similar task to drone auto-landing, and hence relevant to robotics.  At first, the lander will crash wildly, but starting around episode 50, you may notice it start to attempt to remain between the flags, and after a couple hundred episodes, it should start to land with a controlled descent.  In the terminal, you should see the reward becoming positive and increasing over time towards 200:

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/LunarLander.png" width="400">

```
Episode 010   Reward: -508.10   Last length: 079   Average length: 18.20
Episode 020   Reward: -301.04   Last length: 088   Average length: 25.02
Episode 030   Reward: -208.76   Last length: 102   Average length: 31.96
Episode 040   Reward:  -98.75   Last length: 071   Average length: 48.18
Episode 050   Reward: -155.66   Last length: 107   Average length: 53.96
Episode 060   Reward: -103.31   Last length: 091   Average length: 58.13
Episode 070   Reward:  -64.71   Last length: 095   Average length: 64.51
Episode 080   Reward:  -93.23   Last length: 147   Average length: 76.15
Episode 090   Reward: -150.40   Last length: 120   Average length: 86.76
Episode 100   Reward: -218.14   Last length: 100   Average length: 98.21
Episode 110   Reward:  -93.55   Last length: 101   Average length: 100.55
Episode 120   Reward:  -32.54   Last length: 120   Average length: 105.52
Episode 130   Reward: -112.93   Last length: 183   Average length: 120.30
Episode 140   Reward: -188.25   Last length: 110   Average length: 149.31
Episode 150   Reward:  -78.87   Last length: 176   Average length: 148.66
Episode 160   Reward:  +11.95   Last length: 174   Average length: 153.23
Episode 170   Reward: +131.50   Last length: 252   Average length: 155.50
Episode 180   Reward: +110.42   Last length: 128   Average length: 154.47
Episode 190   Reward:  +86.32   Last length: 161   Average length: 156.21
Episode 200   Reward: +111.07   Last length: 505   Average length: 162.06
```

Next, we'll look at integrating these standalone Python examples into robotics code via our C++ wrapper library.

# Digging into the C++ API

To take these deep reinforcement learners from monolithic Python examples into libray form that can be integrated with robots and simulators, we provide a C++ wrapper library and API to the Python code.  Underneath, the library uses Python's low-level C FFI to pass the tensor memory between the application and PyTorch without extra copies (ZeroCopy).  

The library is architected to be modular and extended to support new types of learning algorithms.  Below is pseudocode illustrating the signature of the [`rlAgent`](c/rlAgent.h) interface which the RL implementations inherit from:

``` c++
/**
 * Base class for deep reinforcement learning agent
 */
class rlAgent
{
public:
	/**
	 * Create a new instance of a module for training an agent.
	 */
     static rlAgent* Create( uint32_t width, uint32_t height, 
                             uint32_t channels, uint32_t numActions );

	/**
	 * Destructor
	 */
	virtual ~rlAgent();

	/**
	 * From the input state, predict the next action
	 */
	virtual bool NextAction( Tensor* state, int* action );

	/**
	 * Issue the next reward and training iteration
	 */
	virtual bool NextReward( float reward, bool end_episode );
};
```

Included in the repo are different implementations of the agent, including **[`dqnAgent`](c/dqnAgent.h)** which we will use in the simulation scenarios to follow.  The user provides their sensor data, or environmental state, to the `NextAction()` function, which calls the Python script and returns the predicted action, which the user then applies to their robot or simulation.  

Next the reward is issued in the `NextReward()` function, which provides feedback to the learner from the environment and kicks off the next training iteration that makes the agent learn over time.

# Testing the C++ API

To make sure that the reinforcement learners are still functioning properly from C++, some simple examples of using the API called **[`catch`](samples/catch/catch.cpp)** and **[`fruit`](samples/fruit/fruit.cpp)** are provided.  Similar in concept to pong, in [`catch`](samples/catch/catch.cpp) a ball drops from the top of the environment which the agent must catch before the ball reaches the bottom of the screen, by moving it's paddle left or right.

## Catch

Unlike the previous examples which were monolithic Python scripts, the [`catch`](samples/catch/catch.cpp) sample is a simple C/C++ program which links to the reinforcement learning library outlined above.  To test the textual [`catch`](samples/catch/catch.cpp) sample, run the following executable from the terminal.  After around 100 episodes or so, the agent should start winning the episodes nearly 100% of the time:  

``` bash
$ ./catch 
[deepRL]  input_width:    64
[deepRL]  input_height:   64
[deepRL]  input_channels: 1
[deepRL]  num_actions:    3
[deepRL]  optimizer:      RMSprop
[deepRL]  learning rate:  0.01
[deepRL]  replay_memory:  10000
[deepRL]  batch_size:     32
[deepRL]  gamma:          0.9
[deepRL]  epsilon_start:  0.9
[deepRL]  epsilon_end:    0.05
[deepRL]  epsilon_decay:  200.0
[deepRL]  allow_random:   1
[deepRL]  debug_mode:     0
[deepRL]  creating DQN model instance
[deepRL]  DQN model instance created
[deepRL]  DQN script done init
[cuda]  cudaAllocMapped 16384 bytes, CPU 0x1020a800000 GPU 0x1020a800000
[deepRL]  pyTorch THCState  0x0318D490
[deepRL]  nn.Conv2d() output size = 800
WON! episode 1
001 for 001  (1.0000)  
WON! episode 5
004 for 005  (0.8000)  
WON! episode 10
007 for 010  (0.7000)  
WON! episode 15
010 for 015  (0.6667)  
WON! episode 20
013 for 020  (0.6500)  13 of last 20  (0.65)  (max=0.65)
WON! episode 25
015 for 025  (0.6000)  11 of last 20  (0.55)  (max=0.65)
LOST episode 30
018 for 030  (0.6000)  11 of last 20  (0.55)  (max=0.65)
LOST episode 35
019 for 035  (0.5429)  09 of last 20  (0.45)  (max=0.65)
WON! episode 40
022 for 040  (0.5500)  09 of last 20  (0.45)  (max=0.65)
LOST episode 45
024 for 045  (0.5333)  09 of last 20  (0.45)  (max=0.65)
WON! episode 50
027 for 050  (0.5400)  09 of last 20  (0.45)  (max=0.65)
WON! episode 55
031 for 055  (0.5636)  12 of last 20  (0.60)  (max=0.65)
LOST episode 60
034 for 060  (0.5667)  12 of last 20  (0.60)  (max=0.65)
WON! episode 65
038 for 065  (0.5846)  14 of last 20  (0.70)  (max=0.70)
WON! episode 70
042 for 070  (0.6000)  15 of last 20  (0.75)  (max=0.75)
LOST episode 75
045 for 075  (0.6000)  14 of last 20  (0.70)  (max=0.75)
WON! episode 80
050 for 080  (0.6250)  16 of last 20  (0.80)  (max=0.80)
WON! episode 85
055 for 085  (0.6471)  17 of last 20  (0.85)  (max=0.85)
WON! episode 90
059 for 090  (0.6556)  17 of last 20  (0.85)  (max=0.85)
WON! episode 95
063 for 095  (0.6632)  18 of last 20  (0.90)  (max=0.90)
WON! episode 100
068 for 100  (0.6800)  18 of last 20  (0.90)  (max=0.90)
WON! episode 105
073 for 105  (0.6952)  18 of last 20  (0.90)  (max=0.90)
WON! episode 110
078 for 110  (0.7091)  19 of last 20  (0.95)  (max=0.95)
WON! episode 111
079 for 111  (0.7117)  19 of last 20  (0.95)  (max=0.95)
WON! episode 112
080 for 112  (0.7143)  20 of last 20  (1.00)  (max=1.00)
```

Internally, [`catch`](samples/catch/catch.cpp) is using the [`dqnAgent`](c/dqnAgent.h) API from our C++ library to implement the learning.

#### Alternate Arguments

There are some optional command line parameters to [`catch`](samples/catch/catch.cpp) that you can play around with, to change the dimensions of the environment and pixel array input size, increasing the complexity to see how it impacts convergence and training times:

``` bash
$ ./catch --width=96 --height=96
$ ./catch --render  # enable text output of the environment
```

With `96x96` environment size, the catch agent achieves >75% accuracy after around 150-200 episodes.  
With `128x128` environment size, the catch agent achieves >75% accuracy after around 325 episodes. 

## Fruit

Next, we provide a 2D graphical sample in C++ called [`fruit`](samples/fruit/fruit.cpp), where the agent appears at random locations and must find the "fruit" object to gain the reward and win episodes before running out of bounds or the timeout period expires.  The [`fruit`](samples/fruit/fruit.cpp) agent has 4 possible actions to choose from:  moving up, down, left, and right on the screen in order to navigate to the object.  

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/fruit.gif" width="175">

Note this C++ example is running mostly on the GPU, with the rudimentary 2D rasterization of the environment in CUDA along with the DQN, and the display visualization in OpenGL.  Like before, it is learning "from vision" using to translate the raw pixel array into actions using deep reinforcement learning.

An analog to more complex navigation and motion planning tasks, the simple [`fruit`](samples/fruit/fruit.cpp) example intended to prove that the [`dqnAgent`](c/dqnAgent.h) is able of visually identifying and navigating to objects of interest from any starting location.  Later on in the repo, we will build on that path-planning capability in the 3D robotic simulations.

#### Running the Sample

To start [`fruit`](samples/fruit/fruit.cpp), launch the following executable from the terminal:

``` bash
$ ./fruit
```

It should achieve around 95% accuracy after around ~100 episodes within the default `48x48` environment:

```
action = DOWN   reward = +0.0628     wins = 052 of 094 (0.55)   16 of last 20  (0.80)  (max=0.80)
action = LEFT   reward = +0.0453     wins = 052 of 094 (0.55)   16 of last 20  (0.80)  (max=0.80)
action = LEFT   reward = +0.0271     wins = 052 of 094 (0.55)   17 of last 20  (0.85)  (max=0.85)
action = LEFT   reward = +0.0084     wins = 052 of 094 (0.55)   17 of last 20  (0.85)  (max=0.85)
action = UP     reward = +0.1208     wins = 052 of 094 (0.55)   17 of last 20  (0.85)  (max=0.85)
action = LEFT   reward = +0.1154     wins = 052 of 094 (0.55)   17 of last 20  (0.85)  (max=0.85)
action = UP     reward = +1.0000 EOE wins = 053 of 095 (0.56)   17 of last 20  (0.85)  (max=0.85)
action = DOWN   reward = +0.1441     wins = 053 of 095 (0.56)   18 of last 20  (0.90)  (max=0.90)
action = DOWN   reward = +0.1424     wins = 053 of 095 (0.56)   18 of last 20  (0.90)  (max=0.90)
action = DOWN   reward = +0.1406     wins = 053 of 095 (0.56)   18 of last 20  (0.90)  (max=0.90)
action = DOWN   reward = +0.1386     wins = 053 of 095 (0.56)   18 of last 20  (0.90)  (max=0.90)
action = DOWN   reward = +0.1365     wins = 054 of 096 (0.57)   19 of last 20  (0.95)  (max=0.95)
action = DOWN   reward = +0.1342     wins = 054 of 096 (0.57)   19 of last 20  (0.95)  (max=0.95)
action = RIGHT  reward = +0.0134     wins = 054 of 096 (0.57)   19 of last 20  (0.95)  (max=0.95)
```

#### Alternate Arguments

In a similar vein to the [`catch`](samples/catch/catch.cpp) sample, there are some optional command line parameters to [`fruit`](samples/fruit/fruit.cpp) that you can exercise:

``` bash
$ ./fruit --width=64 --height=64 --episode_max_frames=100
```

When increasing the dimensions of the environment and pixel array input, the `episode_max_frames` should be increased accordingly, as the agent will require more time to get across the screen in a larger environment before the episode time-out.
 

# 3D Simulation

Up until this point in the repo, the environments have been 2D, namely to confirm that the deep RL algorithms are learning as intended.  To migrate the agent to operating in 3D worlds, we're going to use the [Gazebo](http://gazebosim.org) robotic simulator to simulate different autonomous machines including a robotic arm and rover, which can then be transfered to the real-world robots.

<!---
Discussion of Gazebo plugin architecture
(or maybe this should be high-level, then more specific info under #Robotic-Manipulation/#Navigation)
-->

## Robotic Arm

Our first Gazebo environment trains a robotic arm to touch objects without needing explicit IK ([Inverse Kinematics](https://appliedgo.net/roboticarm/)).  
The arm's motion planning is learned internally by the network.  To get started, run the following script from the terminal:

``` bash
$ ./gazebo-arm.sh
```

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gazebo_arm.jpg">

The plugins which hook the learning into the simulation are located in the [`gazebo/`](gazebo/) directory of the repo.  
See **[`ArmPlugin.cpp`](gazebo/ArmPlugin.cpp)** for the code that links Gazebo with the [`dqnAgent`](c/dqnAgent.h) and controls the arm joints.

Once you notice the arm agent converging on the object, you can begin to move the object around the scene by pressing `T` on the keyboard to enable `Translation` mode in Gazebo, and then by clicking and dragging the object around the viewport.

Note that you will want to move the object so that the arm can still reach it, as the arm's rotational base is initially limited to around 45 degrees of travel in either direction. 

## Rover Navigation

We also have a skid-steer rover in Gazebo that learns to follow objects while avoiding the walls of it's environment, similar to the [`fruit`](samples/fruit/fruit.cpp) scenario.  To launch the rover simulation, run this script:

``` bash
$ ./gazebo-rover.sh
```

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gazebo_rover.jpg">

> Press `Ctrl+T` and subscribe to the `~/camera/link/camera/image` topic to visualize the scene from the camera.

Similar to the arm, once you notice the rover consistently finding the object (in this case the green box), you can move the object around the scene by pressing `T` first.  Note that there's an episode timeout similar to [`fruit`](samples/fruit/fruit.cpp), so you won't want to move the object too far away without first increasing the rover's [`maxEpisodeLength`](https://github.com/dusty-nv/jetson-reinforcement/blob/b038a719ff6e50c067e905ecff3582896e3d659a/gazebo/RoverPlugin.cpp#L79) in the code and re-compiling.

# Continuous Control

<img align="right" src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/DQN-argmax.png" width="275"> The DQN agent that we've been using is discrete, meaning that the network selects one output neuron per timestep, that the user then explicitly maps or defines to correspond to an action (typically increasing/decreasing a position or velocity by a delta amount).  This means that for each degree of freedom in the robot, 2 outputs are required - one to increase the variable by the delta and another to decrease it.

In more complex real-world scenarious it's often advantageous to control all degrees of freedom simultaneously and to have the network output the precise value of these variables.  For example, if you wanted to teach a humanoid to walk (which can have 20-40 or more degrees of freedom), controlling all the joints simultaneously would be important to it's stability.

<p align="center"><img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/RL_discrete_continuous.png"></p>

For continuous control, there exists a class of more advanced deep reinforcement learners called Actor/Critic — an active area of research that's recently yielded the latest state-of-the-art solutions like [DDPG](https://arxiv.org/abs/1509.02971), [ACKTR](https://arxiv.org/abs/1708.05144), and [A3C/A3G](https://arxiv.org/abs/1611.06256).

## Bipedal Walker

To demonstrate a continuous learner on one of the most challenging and difficult OpenAI Gym environments, [`BipedalWalkerHardcore-v2`](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/), included in the repo is a demo of A3G, which launches many Gym instances to learn more quickly in parallel using the GPU.  To launch the A3G solver, run the following commands from terminal:

``` bash
$ cd jetson-reinforcement/python/A3G
$ python main.py --env BipedalWalkerHardcore-v2 --workers 8 --gpu-ids 0 --amsgrad True --model CONV --stack-frames 4
```

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gym_bipedal.jpg">

Depending on settings and system resources, it typically takes A3G between 90-120 minutes to master the environment by clearing the hurdles and pitfalls.  If you have multiple GPUs in a PC or server, you can disable rendering and increase the number of worker threads and specify additional `gpu-ids` to speed up training. 

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gym_bipedal.gif">

# Appendix: Using LUA

By default, the repo builds with PyTorch and Python.  However, there's also support included for Torch7 and LUA script with a compile flag.  The process is scripted to automatically install dependencies like Torch7 and build the project from source.  
You may be required to enter the sudo password at some point.

#### 1. Cloning GitHub repo

First, make sure build tools 
``` bash
$ sudo apt-get install git cmake
$ git clone http://github.com/dusty-nv/jetson-reinforcement
```

#### 2. Configuring build

``` bash
$ cd jetson-reinforcement
$ mkdir build
$ cd build
$ cmake ../ -DUSE_LUA=yes -DUSE_PYTHON=no
```

This will initiate the building of dependencies like Torch and it's bindings for CUDA/cuDNN, which can take some time.


#### 3. Compiling

``` bash
$ cd jetson-inference/build     # omit if pwd is already this directory from step #2
$ make
```

Depending on architecture, the package will be built to either armhf or aarch64, with the following directory structure:

```
|-build
   \aarch64		    (64-bit)
      \bin			where the application binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the application binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```
	  
### Verifying Lua + Torch Install

After either [Building from Source](#building-from-source) or [Downloading the Package](#downloading-the-package], verify the LuaJIT-5.1 / Torch7 scripting environment with these commands:

``` bash
$ cd aarch64/bin

$ ./deepRL-console hello.lua			# verify Lua interpreter (consult if unfamiliar with Lua)

[deepRL]  created new lua_State
[deepRL]  opened LUA libraries
[deepRL]  loading 'hello.lua'

HELLO from LUA!
my variable equals 16
list  1
map.x 10
one
two
3
4
5
6
7
8
9
10
multiply = 200
goodbye!

[deepRL]  closing lua_State
```

This command will test loading Torch7 packages and bindings for CUDA/cuDNN:

``` bash
$ ./deepRL-console test-packages.lua    # load Torch packages and bindings

[deepRL]  created new lua_State
[deepRL]  opened LUA libraries
[deepRL]  loading 'test-packages.lua'

[deepRL]  hello from within Torch/Lua environment (time=0.032163)
[deepRL]  loading Lua packages...
[deepRL]  loading torch...
[deepRL]  loading cutorch...
cutorch.hasHalf == false
[deepRL]  loading nn...
[deepRL]  loading cudnn...
[deepRL]  loading math...
[deepRL]  loading nnx...
[deepRL]  loading optim...
[deepRL]  done loading packages. (time=5.234669)

[deepRL]  closing lua_State
```

These scripts should run normally and verify the Lua / Torch environment is sane.

>  the deepRL-console program can launch a user's script from the command line (CLI).


### Playing Catch with the LUA Q-Learner

Next, to verify that the reinforcement Q-learner learns like it's supposed to, let's play a simple game:  half-pong, or catch.

``` bash
$ ./deepRL-console catchDQN.lua
```

Launching the script above should begin your Jetson playing games of catch and plotting the learning process in realtime:

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/cf9c6939c3e5eb7d46056009a8d03904"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/cf9c6939c3e5eb7d46056009a8d03904"></a>

Each epoch is one game of play, where the ball drops from the top of the screen to the bottom.
After a few hundred epochs, the Q-learner should be starting to catch the ball the majority of the time.

