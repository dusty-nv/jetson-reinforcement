<img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/10dc4a454bb70fe282e196d5758bf3bf">

# Reinforcement Learning
Reinforcement learners choose actions to perform based on environmental state and a reward system.  They provide an AI agent that can learn to behave optimally in it's environment given a policy, or task - like obtaining the reward.

In many scenarios, the state space is significantly complex and multi-dimensional to where neural networks are increasingly used to estimate the Q-function, which approximates the future reward based on state sequence.

This repository includes Q-learning algorithms in Torch7 and an API in C++ for integrating with applications in robotics, simulation, and elsewhere.


## Downloading the Package

These archives contain a full snapshot of the repo on the indicate date including binary build and prerequisites like Torch for the indicated JetPack/L4T release.
These are available to download, extract, and run straight away.

> `JetPack 2.2 / JetPack 2.2.1 64-bit` <br />
> `L4T R24.1 aarch64` <br />
>
> `jetson-reinforcement-R241-aarch64-20160803.tar.gz  (110MB)`
> https://drive.google.com/file/d/0BwYxpotGWRNOYlBmM2xMUXZ0eUE/view?usp=sharing


Alternatively the project and Torch can be built from source, a process which is mostly scripted on supported platforms.
If you downloaded one of the archives above, skip ahead to [Verifying Lua + Torch](#verifying-lua-torch).


## Building from Source

If you want to incorporate the latest source changes or build on a different release, the project can be built from source relatively easily (but can take a bit of time).
The process is scripted to automatically install dependencies like Torch7 and build the project from source.  
You may be required to enter the sudo password at some point.

Note: some versions of JetPack/L4T already have [pre-built archives](#downloading-the-package) available for download, see above.

#### 1. Cloning GitHub repo

First, make sure build tools 
``` bash
$ sudo apt-get install git cmake
$ git clone http://github.org/dusty-nv/jetson-reinforcement
```

#### 2. Configuring build

``` bash
$ cd jetson-reinforcement
$ mkdir build
$ cd build
$ cmake ../
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
	  
## Verifying Lua + Torch Install

After either [Building from Source](#building-from-source) or [Downloading the Package](#downloading-the-package], verify the LuaJIT-5.1 / Torch7 scripting environment with these commands:

``` bash
$ cd aarch64/bin

$ ./deepRL-console hello.lua			# verify Lua interpreter (consult if unfamiliar with Lua)
$ ./deepRL-console test-packages.lua    # load Torch packages and bindings
```

These scripts should run OK and verify the Lua / Torch environment is sane.
Note the deepRL-console program can launch a user's script from the command line (CLI).


## Playing Catch with the Q-Learner

Next, to verify that the reinforcement Q-learner learns like it's supposed to, let's play a simple game:  half-pong, or catch.

``` bash
$ ./deepRL-console catchDQN.lua
```

Launching the script above should begin your Jetson playing games of catch and plotting the learning process in realtime:

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/cf9c6939c3e5eb7d46056009a8d03904"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/cf9c6939c3e5eb7d46056009a8d03904"></a>

Each epoch is one game of play, where the ball drops from the top of the screen to the bottom.
After a few hundred epochs, the Q-learner should be starting to catch the ball the majority of the time.

