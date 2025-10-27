#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../utils/CarlaEgg/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2
import math
import pandas as pd
import os.path

IM_WIDTH = 640
IM_HEIGHT = 480

SHOW_PREVIEW = True

SECONDS_PER_EPISODE = 10
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200

EPISODES = 100


DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


class Q_Learning:

    def __init__(self,
                 actions,
                 epsilon=0.1,
                 gamma=0.9,
                 alpha=0.2,
                 new_table=True,
                 save_table_ever_step=False,
                 descending_epsilon=False,
                 epsilon_min=0.1,
                 descend_epsilon_until=0,
                 path="learning/Q_Tables/q_table.pkl"):

        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.save_table_every_step = save_table_ever_step
        self.path = path
        self.epsilon_min = epsilon_min

        if new_table:
            self.q_table = pd.DataFrame(columns=self.actions)
        else:
            if not os.path.isfile(self.path):
                print("File Not found!")
            self.q_table = pd.read_pickle(path)

        if descending_epsilon and descend_epsilon_until > 0:
            self.delta_epsilon = epsilon / (descend_epsilon_until * 0.9)
        else:
            self.delta_epsilon = 0

        self.old_state_action_pair = None

    def load_model(self, path):
        self.q_table = pd.read_pickle(path)

    def get_action(self, state, reward_for_last_action):

        state_as_string = str(state)

        next_action = self.choose_action(state_as_string)

        if self.old_state_action_pair is not None:
            self.learn(self.old_state_action_pair, reward_for_last_action, state_as_string)

        if self.save_table_every_step:
            self.save_q_table()

        self.old_state_action_pair = (state_as_string, next_action)

        return next_action

    def get_q(self, state, action, default_val):
        self.check_if_state_exists(state, default_val)
        return self.q_table.loc[state, action]

    def check_if_state_exists(self, state, default_val):
        if state not in self.q_table.index:
            self.q_table.loc[state, :] = default_val
        val = self.q_table.loc[state, 0]
        if val is not None and math.isnan(val):
            print("Nan_4")

    def choose_action(self, state):

        state = str(state)

        rand = random.random()

        if rand < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_q(state, a, 1) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]

        return action

    def learn(self, state_action_pair, reward, new_state):
        next_actions = [self.get_q(new_state, a, 1) for a in self.actions]
        best_next_action = max(next_actions)
        q_value_next_state = self.gamma * best_next_action
        self.learn_q(state_action_pair, reward, q_value_next_state)

    def learn_q(self, state_action_pair, reward, q_value_next_state):
        q_val = self.get_q(state_action_pair[0], state_action_pair[1], 1)

        self.q_table.loc[state_action_pair] = q_val + self.alpha * (reward + q_value_next_state - q_val)

        if math.isnan(q_val + self.alpha * (reward + q_value_next_state - q_val)):
            print("NAN")

    def descend_epsilon(self):
        if self.delta_epsilon > 0 and \
                self.epsilon > self.epsilon_min and \
                self.epsilon - self.delta_epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.delta_epsilon

    def save_q_table(self, path=""):
        # print("write Q-table")
        if path == "":
            path = self.path
        if not os.path.isfile(path):
            print("File Not found!")
        self.q_table.to_pickle(path)

    def print_q(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('expand_frame_repr', False)
        print(self.q_table)
        pd.set_option('expand_frame_repr', True)
        pd.reset_option('display.max_columns')


if __name__ == '__main__':

   learner = Q_Learning(range(3),
                                  epsilon=epsilon,
                                  descending_epsilon=True,
                                  descend_epsilon_until=EPSILON_DECAY,
                                  alpha=0.2)

   env = CarEnv()

   while True:

        
        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = learner.get_action((current_state/255)[0], 1)
            action = qs

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            # frame_time = time.time() - step_start
            # fps_counter.append(frame_time)
            # print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()