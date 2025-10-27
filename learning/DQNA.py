#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# import carla

import random
import time
import numpy as np
import cv2
import math
from keras import backend as K

from collections import deque
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Activation, MaxPooling2D, \
    Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1  # How many samples to predict at once (the more, the faster)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 2  # How many samples to fit at once (the more, the faster) - should be MINIBATCH_SIZE divided by power of 2
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
START_EPSILON = 1
EPSILON_DECAY = 0.99995  # 0.99975
MIN_EPSILON = 0.1
IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = True
OPTIMIZER_LEARNING_RATE = 0.001
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 6_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
ACTIONS = ['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']
MODEL_NAME = "DQN"
MODEL_PATH = 'models/QDN'
MEMORY_FRACTION = 0.6
MIN_REWARD = 100
EPISODES = 1000
AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        with tf.compat.v1.Graph().as_default():
            self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        # self.graph = graph
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):

        IM_WIDTH = 640
        IM_HEIGHT = 480
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4, activation='linear'))
        # with self.graph.as_default():
        model.compile(loss=self.huber_loss, optimizer=Adam(learning_rate=OPTIMIZER_LEARNING_RATE))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        # print('DEBUG------------Fitting')
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            # print('Not enough memory')
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0]
                                   for transition in minibatch]) / 255
        # with self.graph.as_default():
        current_qs_list = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array(
            [transition[3] for transition in minibatch]) / 255
        # with self.graph.as_default():

        future_qs_list = self.target_model.predict(
            new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step
        print('--------------Fitting------------------')
        # with self.graph.as_default():
        self.model.fit(np.array(X) / 255, np.array(y),
                       batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        # print('Model History ' , hist.history)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_in_loop(self):
        X = np.random.uniform(
            size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1,4)).astype(np.float32)
        # with self.graph.as_default():
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        # while True:
        #     if self.terminate:
        #         return
        self.train()
            # time.sleep(0.01)
