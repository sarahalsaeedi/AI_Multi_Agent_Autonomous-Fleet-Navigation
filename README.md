# AI Multi Agents Autonomous Fleet Navigation
Reinforcement Learning Autonomous navigation project that uses reinforcement learning algorithms to train multi AI agents to navigate from one point to targeted point using built-in navigation system. 

# Implementation Environment

This project is an autonomous navigation project that uses reinforcement learning to train an agent to navigate from point A to point B, which was 
implemented and tested on [CARLA](https://carla.org/) Open-Source Simulator. 

The goal of this project is to create one or more agents that drive cars through a virtual environment and efficiently complete several mission goals.

##### This project's main contributions are as follows:
* We formulate the problem of Autonomous Fleet Navigation in the reinforcement learning (RL) framework and build the RL-environment on the CARLA simulator.
* Dealing with different sensors provided by CARLA environment and preparing the data for our agents.
* We used different logic for the agent navigation inputs and design an architecture to learn planning and controlling.
* We investigate the use of Q-Learning and Deep Q-Learning off-policy RL algorithms for learning Autonomous Fleet Navigation, as well as an analysis of the various issues in training instability and sharing techniques.
* Achieving mission goals which include ( Spawning and keeping the Car on the road, Avoid collisions and Obstacles, Steer properly to take turns, Drive vehicle from point A to point B).


### Recommended system

* Intel i7 gen 9th - 11th / Intel i9 gen 9th - 11th / AMD ryzen 7 / AMD ryzen 9
* +16 GB RAM memory 
* NVIDIA RTX 2070 / NVIDIA RTX 2080 / NVIDIA RTX 3070, NVIDIA RTX 3080
* Ubuntu 18.04/Windows7/10 

## Installation and Running The Project

* CARLA Environment Downloading [here](https://github.com/carla-simulator/carla/blob/master/Docs/download.md)
* CARLA Documentation [here](https://carla.readthedocs.io/en/0.9.11/)

### Installation

Navigate to the project folder 

```bash
cd autonomous-fleet-navigation\code
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the project requirements.

```bash
pip install -r requirements.txt
#OR 
python -m pip install -r requirements.txt
```

### Running CARLA

```bash
#Windows
Carla\CarlaUE4.exe -quality-level=Low/high
#Lunix
Carla\CarlaUE4.sh -quality-level=Low/high
```
