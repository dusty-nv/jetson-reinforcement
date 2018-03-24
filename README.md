<img src="https://github.com/dusty-nv/jetson-reinforcement/raw/master/docs/images/jetson-reinforcement-header.jpg">

# Reinforcement Learning in Robotics
In this tutorial, we'll be creating artificially intelligent agents that learn from interacting with their environment, gathering experience, and a system of rewards with deep reinforcement learning (deep RL).  Using end-to-end neural networks that translate raw pixels into actions, RL-trained agents are capable of exhibiting intuitive behaviors and performing complex tasks.  

Ultimately, our aim will be to train reinforcement learning agents from virtual robotic simulation in 3D and transfer the agent to a real-world robot.  Reinforcement learners choose the best action for the agent to perform based on environmental state (like camera inputs) and rewards that provide feedback to the agent about it's performance.  Reinforcement learning can learn to behave optimally in it's environment given a policy, or task - like obtaining the reward.

In many scenarios, the state space is significantly complex and multi-dimensional to where neural networks are increasingly used to predict the best action, which is where deep reinforcement learning and GPU acceleration comes into play.  With deep reinforcement learning the agents are typically processing 2D imagery using convolutional neural networks (CNNs), inputs that are an order of magnitude more complex than low-dimensional RL, and the ability to learn "from vision" end-to-end.

This repository includes Deep Q-Learning (DQN) and A3G algorithms in PyTorch, examples and an interoperability library API in C++ for integrating with Linux applications in robotics, simulation, and deployment to the field.

### **Table of Contents**

* [Building from Source](#building-from-source)
* [Verifying PyTorch](#verifying-pytorch)
* [DQN + OpenAI Gym](#dqn-openai-gym)
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

During the `cmake` step, PyTorch will be compiled and installed so it can take awhile (around ~30 minutes to an hour on the Jetson).  The stable version of PyTorch we are currently using is `0.3.0`.  The build script will download pacekages and ask you for your `sudo` password during the install.

# Verifying PyTorch

Before proceeding, to make sure that PyTorch installed correctly, and to get an introduction to PyTorch if you aren't already familiar, we have provided a Jupyter IPython notebook called **[`intro-pytorch.ipynb`](python/intro-pytorch.ipynb)** that includes some simple PyTorch examples that verify the install and test the CUDA/cuDNN support in PyTorch.

To launch the [notebook](python/intro-pytorch.ipynb) locally on your system, run the following commands:

``` bash
$ cd jetson-reinforcement/build/aarch64/bin   # or cd x86_64/bin on PC
$ jupyter notebook intro-pytorch.ipynb
```

Alternatively, if you wish to skip the notebook and run the PyTorch verification commands directly, you can do so by launching an interactive Python shell and running the following:

``` python
$ python
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
$ python
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

In order to first test and verify that the deep reinforcement learning algorithms are indeed learning, we'll run them inside OpenAI Gym environments (in 2D).  As an introduction to the DQN algorithm, a second CUDA-enabled IPython notebook is included in the repo, **[`intro-DQN.ipynb`](python/intro-DQN.ipynb)**.  This notebook applies the DQN on video captured from the Gym's [`CartPole`](https://gym.openai.com/envs/CartPole-v0/) environment, so it's learning "from vision" on the GPU, as opposed to low-dimensional parameters from the game like traditional RL.  Although CartPole is a toy example, it's vital to start with a simple example to eliminate potential issues early on before graduating to more complex 3D scenarios that will become more difficult to debug, and since the DQN learns from a 2D pixel array it's still considered deep reinforcement learning.  It's recommended to follow along with the notebook to familiarize yourself with the DQN algorithm for when we transition to using it from C++ in more complex environments later in the repo. 

## Cartpole

To launch the [notebook](python/intro-DQN.ipynb) locally from your machine, run the following commands:

``` bash
$ cd jetson-reinforcement/build/aarch64/bin   # or cd x86_64/bin on PC
$ jupyter notebook intro-DQN.ipynb
```

The DQN is only set to run for 50 episodes inside of the notebook.  After you have witnessed the DQN start to converge, exit the notebook and run the standalone **[`gym-DQN.py`](python/gym-DQN.py)** script from the terminal for improved performance:


Following this [Deep Q-Learning tutorial](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) from PyTorch, run these commands to launch the DQN agent:

``` bash
$ python gym-DQN.py
```
> (the following is assuming the current directory of your terminal is still `cd jetson-reinforcement/build/<arch>/bin` from above)

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

Next, we'll look at integrating these standalone Python examples into our robotics code via a C++ interposer library.

# Digging into the C++ API

To take these deep reinforcement learners from monolithic Python examples into libray form that can be integrated with simulators and real robots, we provide a C++ wrapper library and API to the Python code.  Underneath, the library uses Python's low-level C FFI API to pass the tensor memory between the user's application and PyTorch without extra copies (ZeroCopy).  The library is architected to be extended to new types of learning algorithms.  Below is pseudocode illustrating the signature of the [`rlAgent`](c/rlAgent.h) base class:

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

Included in the repo are different implementations of the agent, including **[`dqnAgent`](c/dqnAgent.h)** which we will use primarily in the simulation scenarios to follow.  The user provides their sensor data, or environmental state, to the `NextAction()` function, which calls the Python script and returns the predicted action, which the user then applies to their robot or simulation.  Next the reward is issued in the `NextReward()` function, which provides feedback to the learner and kicks off the next training iteration that makes the agent learn over time.

## Testing the API

To make sure that the reinforcement learners are still functioning properly from C++, some simple examples of using the API called [`catch`](samples/catch/catch.cpp) and [`fruit`](samples/fruit/fruit.cpp) are provided.  Similar in concept to pong, in `catch` a ball drops from the top of the environment which the agent must catch before the ball reaches the bottom of the screen, by moving it's paddle left or right.

Unlike the previous examples which were monolithic Python scripts, the [`catch`](samples/catch/catch.cpp) sample is a simple C/C++ program which links to the reinforcement learning library outlined above.  To test the textual `catch` sample, run the following executable from the terminal.  After around 100 episodes or so, the agent should start winning the episodes nearly 100% of the time.  

``` bash
$ ./catch 
[deepRL]  use_cuda:       True
[deepRL]  use_lstm:       1
[deepRL]  lstm_size:      256
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
[deepRL]  DRQN::__init__()
[deepRL]  LSTM (hx, cx) size = 256
[deepRL]  DQN model instance created
[deepRL]  DQN script done init
[cuda]  cudaAllocMapped 16384 bytes, CPU 0x1020a800000 GPU 0x1020a800000
[deepRL]  pyTorch THCState  0x0318D490
[deepRL]  nn.Conv2d() output size = 800
WON! episode 1
001 for 001  (1.0000)  
WON! episode 2
002 for 002  (1.0000)  
LOST episode 3
002 for 003  (0.6667)  
WON! episode 4
003 for 004  (0.7500)  
WON! episode 5
004 for 005  (0.8000)  
LOST episode 6
004 for 006  (0.6667)  
WON! episode 7
005 for 007  (0.7143)  
LOST episode 8
005 for 008  (0.6250)  
WON! episode 9
006 for 009  (0.6667)  
WON! episode 10
007 for 010  (0.7000)  
LOST episode 11
007 for 011  (0.6364)  
WON! episode 12
008 for 012  (0.6667)  
WON! episode 13
009 for 013  (0.6923)  
LOST episode 14
009 for 014  (0.6429)  
WON! episode 15
010 for 015  (0.6667)  
WON! episode 16
011 for 016  (0.6875)  
LOST episode 17
011 for 017  (0.6471)  
LOST episode 18
011 for 018  (0.6111)  
WON! episode 19
012 for 019  (0.6316)  
WON! episode 20
013 for 020  (0.6500)  13 of last 20  (0.65)  (max=0.65)
LOST episode 21
013 for 021  (0.6190)  12 of last 20  (0.60)  (max=0.65)
WON! episode 22
014 for 022  (0.6364)  12 of last 20  (0.60)  (max=0.65)
LOST episode 23
014 for 023  (0.6087)  12 of last 20  (0.60)  (max=0.65)
LOST episode 24
014 for 024  (0.5833)  11 of last 20  (0.55)  (max=0.65)
WON! episode 25
015 for 025  (0.6000)  11 of last 20  (0.55)  (max=0.65)
WON! episode 26
016 for 026  (0.6154)  12 of last 20  (0.60)  (max=0.65)
WON! episode 27
017 for 027  (0.6296)  12 of last 20  (0.60)  (max=0.65)
WON! episode 28
018 for 028  (0.6429)  13 of last 20  (0.65)  (max=0.65)
LOST episode 29
018 for 029  (0.6207)  12 of last 20  (0.60)  (max=0.65)
LOST episode 30
018 for 030  (0.6000)  11 of last 20  (0.55)  (max=0.65)
LOST episode 31
018 for 031  (0.5806)  11 of last 20  (0.55)  (max=0.65)
LOST episode 32
018 for 032  (0.5625)  10 of last 20  (0.50)  (max=0.65)
LOST episode 33
018 for 033  (0.5455)  09 of last 20  (0.45)  (max=0.65)
WON! episode 34
019 for 034  (0.5588)  10 of last 20  (0.50)  (max=0.65)
LOST episode 35
019 for 035  (0.5429)  09 of last 20  (0.45)  (max=0.65)
WON! episode 36
020 for 036  (0.5556)  09 of last 20  (0.45)  (max=0.65)
LOST episode 37
020 for 037  (0.5405)  09 of last 20  (0.45)  (max=0.65)
WON! episode 38
021 for 038  (0.5526)  10 of last 20  (0.50)  (max=0.65)
LOST episode 39
021 for 039  (0.5385)  09 of last 20  (0.45)  (max=0.65)
WON! episode 40
022 for 040  (0.5500)  09 of last 20  (0.45)  (max=0.65)
WON! episode 41
023 for 041  (0.5610)  10 of last 20  (0.50)  (max=0.65)
WON! episode 42
024 for 042  (0.5714)  10 of last 20  (0.50)  (max=0.65)
LOST episode 43
024 for 043  (0.5581)  10 of last 20  (0.50)  (max=0.65)
LOST episode 44
024 for 044  (0.5455)  10 of last 20  (0.50)  (max=0.65)
LOST episode 45
024 for 045  (0.5333)  09 of last 20  (0.45)  (max=0.65)
WON! episode 46
025 for 046  (0.5435)  09 of last 20  (0.45)  (max=0.65)
WON! episode 47
026 for 047  (0.5532)  09 of last 20  (0.45)  (max=0.65)
LOST episode 48
026 for 048  (0.5417)  08 of last 20  (0.40)  (max=0.65)
LOST episode 49
026 for 049  (0.5306)  08 of last 20  (0.40)  (max=0.65)
WON! episode 50
027 for 050  (0.5400)  09 of last 20  (0.45)  (max=0.65)
WON! episode 51
028 for 051  (0.5490)  10 of last 20  (0.50)  (max=0.65)
LOST episode 52
028 for 052  (0.5385)  10 of last 20  (0.50)  (max=0.65)
WON! episode 53
029 for 053  (0.5472)  11 of last 20  (0.55)  (max=0.65)
WON! episode 54
030 for 054  (0.5556)  11 of last 20  (0.55)  (max=0.65)
WON! episode 55
031 for 055  (0.5636)  12 of last 20  (0.60)  (max=0.65)
LOST episode 56
031 for 056  (0.5536)  11 of last 20  (0.55)  (max=0.65)
WON! episode 57
032 for 057  (0.5614)  12 of last 20  (0.60)  (max=0.65)
WON! episode 58
033 for 058  (0.5690)  12 of last 20  (0.60)  (max=0.65)
WON! episode 59
034 for 059  (0.5763)  13 of last 20  (0.65)  (max=0.65)
LOST episode 60
034 for 060  (0.5667)  12 of last 20  (0.60)  (max=0.65)
WON! episode 61
035 for 061  (0.5738)  12 of last 20  (0.60)  (max=0.65)
LOST episode 62
035 for 062  (0.5645)  11 of last 20  (0.55)  (max=0.65)
WON! episode 63
036 for 063  (0.5714)  12 of last 20  (0.60)  (max=0.65)
WON! episode 64
037 for 064  (0.5781)  13 of last 20  (0.65)  (max=0.65)
WON! episode 65
038 for 065  (0.5846)  14 of last 20  (0.70)  (max=0.70)
WON! episode 66
039 for 066  (0.5909)  14 of last 20  (0.70)  (max=0.70)
WON! episode 67
040 for 067  (0.5970)  14 of last 20  (0.70)  (max=0.70)
WON! episode 68
041 for 068  (0.6029)  15 of last 20  (0.75)  (max=0.75)
LOST episode 69
041 for 069  (0.5942)  15 of last 20  (0.75)  (max=0.75)
WON! episode 70
042 for 070  (0.6000)  15 of last 20  (0.75)  (max=0.75)
LOST episode 71
042 for 071  (0.5915)  14 of last 20  (0.70)  (max=0.75)
WON! episode 72
043 for 072  (0.5972)  15 of last 20  (0.75)  (max=0.75)
WON! episode 73
044 for 073  (0.6027)  15 of last 20  (0.75)  (max=0.75)
WON! episode 74
045 for 074  (0.6081)  15 of last 20  (0.75)  (max=0.75)
LOST episode 75
045 for 075  (0.6000)  14 of last 20  (0.70)  (max=0.75)
WON! episode 76
046 for 076  (0.6053)  15 of last 20  (0.75)  (max=0.75)
WON! episode 77
047 for 077  (0.6104)  15 of last 20  (0.75)  (max=0.75)
WON! episode 78
048 for 078  (0.6154)  15 of last 20  (0.75)  (max=0.75)
WON! episode 79
049 for 079  (0.6203)  15 of last 20  (0.75)  (max=0.75)
WON! episode 80
050 for 080  (0.6250)  16 of last 20  (0.80)  (max=0.80)
WON! episode 81
051 for 081  (0.6296)  16 of last 20  (0.80)  (max=0.80)
WON! episode 82
052 for 082  (0.6341)  17 of last 20  (0.85)  (max=0.85)
WON! episode 83
053 for 083  (0.6386)  17 of last 20  (0.85)  (max=0.85)
WON! episode 84
054 for 084  (0.6429)  17 of last 20  (0.85)  (max=0.85)
WON! episode 85
055 for 085  (0.6471)  17 of last 20  (0.85)  (max=0.85)
WON! episode 86
056 for 086  (0.6512)  17 of last 20  (0.85)  (max=0.85)
WON! episode 87
057 for 087  (0.6552)  17 of last 20  (0.85)  (max=0.85)
WON! episode 88
058 for 088  (0.6591)  17 of last 20  (0.85)  (max=0.85)
LOST episode 89
058 for 089  (0.6517)  17 of last 20  (0.85)  (max=0.85)
WON! episode 90
059 for 090  (0.6556)  17 of last 20  (0.85)  (max=0.85)
WON! episode 91
060 for 091  (0.6593)  18 of last 20  (0.90)  (max=0.90)
LOST episode 92
060 for 092  (0.6522)  17 of last 20  (0.85)  (max=0.90)
WON! episode 93
061 for 093  (0.6559)  17 of last 20  (0.85)  (max=0.90)
WON! episode 94
062 for 094  (0.6596)  17 of last 20  (0.85)  (max=0.90)
WON! episode 95
063 for 095  (0.6632)  18 of last 20  (0.90)  (max=0.90)
WON! episode 96
064 for 096  (0.6667)  18 of last 20  (0.90)  (max=0.90)
WON! episode 97
065 for 097  (0.6701)  18 of last 20  (0.90)  (max=0.90)
WON! episode 98
066 for 098  (0.6735)  18 of last 20  (0.90)  (max=0.90)
WON! episode 99
067 for 099  (0.6768)  18 of last 20  (0.90)  (max=0.90)
WON! episode 100
068 for 100  (0.6800)  18 of last 20  (0.90)  (max=0.90)
WON! episode 101
069 for 101  (0.6832)  18 of last 20  (0.90)  (max=0.90)
WON! episode 102
070 for 102  (0.6863)  18 of last 20  (0.90)  (max=0.90)
WON! episode 103
071 for 103  (0.6893)  18 of last 20  (0.90)  (max=0.90)
WON! episode 104
072 for 104  (0.6923)  18 of last 20  (0.90)  (max=0.90)
WON! episode 105
073 for 105  (0.6952)  18 of last 20  (0.90)  (max=0.90)
WON! episode 106
074 for 106  (0.6981)  18 of last 20  (0.90)  (max=0.90)
WON! episode 107
075 for 107  (0.7009)  18 of last 20  (0.90)  (max=0.90)
WON! episode 108
076 for 108  (0.7037)  18 of last 20  (0.90)  (max=0.90)
WON! episode 109
077 for 109  (0.7064)  19 of last 20  (0.95)  (max=0.95)
WON! episode 110
078 for 110  (0.7091)  19 of last 20  (0.95)  (max=0.95)
WON! episode 111
079 for 111  (0.7117)  19 of last 20  (0.95)  (max=0.95)
WON! episode 112
080 for 112  (0.7143)  20 of last 20  (1.00)  (max=1.00)
```

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

