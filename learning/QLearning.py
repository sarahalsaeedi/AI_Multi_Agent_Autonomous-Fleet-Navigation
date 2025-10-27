
import glob
import os
import sys
 
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import cv2
import math
import pandas as pd
import os.path

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
                 path="./q_training/train_QAgent_train_Q_Learning/model.pkl"):
        print("Starting With Q_Learning")
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.save_table_every_step = save_table_ever_step
        self.path = path
        self.epsilon_min = epsilon_min

        if self.path is not None and  os.path.isfile(self.path):
            self.q_table = pd.read_pickle(path)
        else:
            self.q_table = pd.DataFrame(columns=self.actions)
            
                
            
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
