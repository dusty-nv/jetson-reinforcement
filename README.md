<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/jetson-reinforcement-header.jpg">

# Reinforcement Learning in Robotics
In this tutorial, we'll be creating artificially intelligent agents that learn from interacting with their environment, gathering experience, and a system of rewards with deep reinforcement learning (deep RL).  Using end-to-end neural networks that translate raw pixels into actions, RL-trained agents are capable of exhibiting intuitive behaviors and performing complex tasks.  

Ultimately, our aim will be to train RL agents from virtual robotic simulation in 3D and transfer the agent to a real-world robot.  Reinforcement learners choose the best action for the agent to perform based on environmental state and rewards that provide feedback.  They provide an AI agent that can learn to behave optimally in it's environment given a policy, or task - like obtaining the reward.

In many scenarios, the state space is significantly complex and multi-dimensional to where neural networks are increasingly used to estimate the Q-function, which approximates the future reward based on state sequence.

This repository includes Deep Q-Learning (DQN) and A3G algorithms in PyTorch, and an interoperability API in C++ for integrating with applications in robotics, simulation, and elsewhere.

### **Table of Contents**

* [Building from Source](#building-from-source)
* [OpenAI Gym](#openai-gym)
	* [Cartpole](#cartpole)
	* [Lunar Lander](#lunar-lander)
* [Digging into the C++ API](#digging-into-the-c-api)
	* [Testing the API](#testing-the-api)
* [3D Simulation](#3d-simulation)
	* [Robotic Manipulation](#manipulation)
* [Using LUA](#using-lua)

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

During the `cmake` step, Torch will be installed so it can take awhile.  It will download packages and ask you for your `sudo` password during the install.

# OpenAI Gym

First, in order to test and verify the original reinforcement learning algorithms, we'll run them in Gym environments.  There are a few examples included in the repo which you can run.

## Cartpole

Following this [Deep Q-Learning tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) from PyTorch, run these commands to launch the DQN agent:

``` bash
$ cd aarch64/bin   # or cd x86_64/bin on PC
$ python gym-DQN.py
```

Three windows should appear showing the cartpole game, a graph of peformance, and the DQN agent should begin learning.  The longer the DQN agent is able to balance the pole on the moving cart, the more points it's rewarded.  In Gym, a score of 200 indicates the scenario has been mastered.  After a short while of training, the agent should achieve it and the program will quit.

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/DQN-cartpole.png" width="400">

## Lunar Lander

Using a similar script, you can experiment in different Gym environments with the `--env` parameter:

``` bash
$ python gym-RL.py --env=LunarLander-v2 --render
```

At first, the lander will crash wildly, but starting around episode 50, you may notice it start to attempt to remain between the flags, and after several hundred episodes, it may start to land with a controlled descent.  In the terminal, you should see the average length (i.e. reward) increasing over time towards 200:

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/LunarLander.png" width="400">

```
Episode 10	Last length:   138	Average length: 18.19
Episode 20	Last length:   113	Average length: 28.81
Episode 30	Last length:    90	Average length: 36.14
Episode 40	Last length:    81	Average length: 41.13
Episode 50	Last length:    99	Average length: 45.69
Episode 60	Last length:   107	Average length: 49.57
Episode 70	Last length:   103	Average length: 53.05
Episode 80	Last length:   143	Average length: 57.00
Episode 90	Last length:    97	Average length: 61.61
Episode 100	Last length:   114	Average length: 67.69
Episode 110	Last length:   163	Average length: 73.48
Episode 120	Last length:   116	Average length: 75.60
Episode 130	Last length:    98	Average length: 78.31
Episode 140	Last length:    77	Average length: 81.20
Episode 150	Last length:   107	Average length: 84.97
Episode 160	Last length:    97	Average length: 88.41
Episode 170	Last length:   115	Average length: 90.93
Episode 180	Last length:   102	Average length: 91.78
Episode 190	Last length:    98	Average length: 92.57
Episode 200	Last length:   199	Average length: 94.67
Episode 210	Last length:   131	Average length: 96.89
Episode 220	Last length:   118	Average length: 100.67
Episode 230	Last length:   143	Average length: 105.71
Episode 240	Last length:    74	Average length: 107.37
Episode 250	Last length:   230	Average length: 109.72
...
```

Next, we'll look at integrating these standalone Python examples into our existing robotics code.

# Digging into the C++ API

To take the reinforcement learners from standalone examples into libray form that can be integrated with robots and simulators, we provide a C++ interface to the Python code.  Underneath the wrappers use Python's low-level C API to pass the memory between the user's application and Torch without extra copies.  Below is pseudocode illustrating the signature of the [`rlAgent`](c/rlAgent.h) base class:

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

Included in the repo are different implementations of the agent, including DQN.  The user provides their sensor data, or environmental state, to the `NextAction()` function, returning the predicted action which the user applies to their robot or simulation.  Then the reward is issued in the `NextReward()` function, which kicks off the next training iteration that makes the agent learn over time.

## Testing the API

To make sure that the reinforcement learners are still functioning properly from C++, a simple example of using the API called [`catch`](samples/catch/catch.cpp) is provided.  Similar in concept to pong, a ball drops from the top of the screen which the agent must catch before the ball reaches the bottom of the screen, by moving it's paddle left or right.

To test the catch sample, run the following executable from the terminal:

``` bash
$ ./catch
use_cuda: True
[deepRL]  nn.Conv2d() output size = 448
[cuda]  cudaAllocMapped 38400 bytes, CPU 0x1020a600000 GPU 0x1020a600000
[deepRL]  pyTorch THCState  0x0262D030
LOST episode 1
0 for 1  (0.0000)  
WON! episode 2
1 for 2  (0.5000)  
LOST episode 3
1 for 3  (0.3333)  
WON! episode 4
2 for 4  (0.5000)  
LOST episode 5
2 for 5  (0.4000)  
WON! episode 6
3 for 6  (0.5000)  
LOST episode 7
3 for 7  (0.4286)  
WON! episode 8
4 for 8  (0.5000)  
LOST episode 9
4 for 9  (0.4444)  
LOST episode 10
4 for 10  (0.4000)  
LOST episode 11
4 for 11  (0.3636)  
WON! episode 12
5 for 12  (0.4167)  
LOST episode 13
5 for 13  (0.3846)  
LOST episode 14
5 for 14  (0.3571)  
WON! episode 15
6 for 15  (0.4000)  
LOST episode 16
6 for 16  (0.3750)  
WON! episode 17
7 for 17  (0.4118)  
LOST episode 18
7 for 18  (0.3889)  
WON! episode 19
8 for 19  (0.4211)  
WON! episode 20
```

After around 100 episodes or so, the agent should start winning the episodes most of the time, between 70-80%.  Unlike the previous examples which were standalone Python scripts, the [`catch`](samples/catch/catch.cpp) sample is a simple C/C++ program which links to the reinforcement learning library outlined above.


# 3D Simulation

Up until this point in the tutorial, the RL environments have been 2-dimensional.  To migrate the agent to operating in 3D worlds, we're going to use the [Gazebo](http://gazebosim.org) robotics simulator to simulate different autonomous machines including a robotic arm and rover, which can then be transfered to the real-world robots.

<!---
Discussion of Gazebo plugin architecture
(or maybe this should be high-level, then more specific info under #Robotic-Manipulation/#Navigation)
-->

## Robotic Manipulation

Our first Gazebo environment we'll be using involves training a robotic arm to manipulate objects.  To get started, run the following script:

``` bash
$ ./gazebo-arm.sh
```

If you press `Ctrl+T` and subscribe to the `~/camera/link/camera/image` topic, you can visualize the scene from the robot's perspective.

<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/gazebo.png">

The plugins which hook the learning into the simulation are located in the `gazebo/` directory of the repo.

<!---
#Navigation section
-->

# Using LUA

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

