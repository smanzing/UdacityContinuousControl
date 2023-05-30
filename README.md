# Continuous Control

## Project Details

In this project, we train multiple agents that are double-jointed robot arms to follow moving target regions. 

### Environment

**Number of Agents**: There are 20 agents in the environment.

**Reward**: Each agent receives a reward of +0.1 for each time step its end effector is located inside the target region.

**Observation Space**: For each agent, we observe 33 variables that are the position, rotation, velocity, and angular 
velocities of the robot arm.

**Action Space**: For each agent, we have 4 continuous actions corresponding to the torque of the two joints. 
The torque values are limited between -1 and 1.

**Success Criteria**: The environment is solved, if the average score over all agents is at least +30 averaged 
over 100 consecutive episodes.


## Getting Started

Follow the instructions below to set up your python environment to run the code in this repository.
Note that the installation was only tested for __Mac__.

1. Initialize the git submodules in the root folder of this repository. 

	```bash
	git submodule init
	git submodule update
	```
 
2. Create (and activate) a new conda environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	
3. Install the base Gym library and the **box2d** environment group:

	```bash
	pip install gym
	pip install Box2D gym
	```

4. Navigate to the `external/deep-reinforcement-learning/python/` folder.  Then, install several dependencies.

    ```bash
    cd external/deep-reinforcement-learning/python
    pip install .
    ```
    **Note**: If an error appears during installation regarding the torch version, please remove the torch version 0.4.0 in
    external/deep-reinforcement-learning/python/requirements.txt and try again.

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    **Note**: Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

6. Download the [Unity environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)  in the **root folder** of this repository. Note that this should be the environment containing 20 agents.

    
## Instructions

After you have successfully installed the dependencies, open the jupyter notebook [a2c.ipynb](a2c.ipynb) 
and follow the instructions to learn how to train the agent.

Below, you can find an overview of the files in the repository.

### Repository Overview

The folder **external** contains the repository [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
as submodule. The repository is only used for installation purposes.

The folder **rl_lib** contains the library that implements the agent and the algorithms for training the agent: 

- [a2c_training.py](rl_lib/a2c_training.py): The file contains the main function **a2c_training** for training the agent. 
- [a2c_agent.py](rl_lib/a2c_agent.py): Implementation of the agent that interacts and learns from the environment. 
                               The member function **learn** implements the A2C algorithm.
- [a2c_model.py](rl_lib/a2c_model.py): Neural network models for the actor and critic.

The jupyter notebook [a2c.ipynb](a2c.ipynb) shows how to train the agent using the A2C algorithm.

## References 

The implementation is based on the research paper [[1]](#1) and the code provided by Udacity, see 
[deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning#dependencies).

Following repositories served as inspiration for the implementation:

- [Tom Lin's repository on continuous control](https://github.com/TomLin/RLND-project/tree/master/p2-continuous-control)
- [Shangtong Zhang's repository on deep RL](https://github.com/ShangtongZhang/DeepRL)

<a id="1">[1]</a> 
Mnih, V. *et al.* 
**Asynchronous Methods for Deep Reinforcement Learning.**
*arXiv* **10.48550/ARXIV.1602.01783**, (2016).
