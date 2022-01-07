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

## Reinforcement Learning:
Reinforcement Learning (RL) has shown promising results in the learning of complex decision-making tasks ranging from strategic games to challenging robotics tasks. Furthermore, autonomous driving is a promising application for RL due to its dense reward structure and short-term horizons.

In this project, we have used two reinforcement learning algorithms which are Q-Learning, and Deep Q-Learning and each agent is implemented with different parameter inputs. 

## Q-Learning Agent
This agent will currently controls the steering control, which moves the steer scale levels from right to left and 0 steer is the forward-moving, currently works with more than **10 steering levels**, and it has an error function that adjusts the accuracy of the steering to give the correct angle values.

In Q-Learning agent we have achieved the following tasks (**Move Straight without lane  invasion, Avoid obstacles, Turn Left & Right, Navigate from A-B, Traffic Rules, Avoid Crashing with Dynamic Objects, Navigation System, Adjusting Steers***)

#### Run Q-Learning
Navigate to the project folder then from /code run the following command: 

```bash
#Windows
py -3.7 run_QLearning_Agent.py
#Linux
python run_QLearning_Agent.py
```


### Running CARLA

```bash
#Windows
Carla\CarlaUE4.exe -quality-level=Low/high
#Lunix
Carla\CarlaUE4.sh -quality-level=Low/high
```


#### Demo Video for Q-Learning
[![Watch the video](/media/images/qn/Q-learning.png)](/media/videos/QLearning%20A-B%20Final%20Presnts.mp4)

#### Training Results for Q-Learning
After training the Q-Learning agent for about 350 episodes we have got the following results: (Note: QN and DQN has different reward values): 

| Total Cost | Total Steps | Total Agnet Reached Presnted by (0 or 1) |
|:---:|:---:|:---:|
| ![total_cost](/media/images/qn/total_cost.png?raw=true "total_cost")  | ![total_steps](/media/images/qn/total_steps.png?raw=true "Total Stpes") | ![total_reached_distination](/media/images/qn/total_reached_distination.png?raw=true "total_reached_distination") |

| Total Lane Crossed | Total Wrong Turns |
|:---:|:---:|
|   ![total_lane_crossed](/media/images/qn/total_lane_crossed.png?raw=true "total_lane_crossed")  |   ![total_turns](/media/images/qn/total_turns.png?raw=true "total_turns")   |


