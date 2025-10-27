import glob
import os
import sys
import random
import time
import cv2
import math
import os.path

class Runner:
    def __init__(self, learner, env, rl_algorithm=''):
        
        self.learner = learner
        self.env = env
        self.rl_algorithm = rl_algorithm
    def run(self):
        while True:
                
            
            print('Restarting episode')

            # Reset environment and get initial state
            current_state = self.env.reset()
            #self.env.collision_hist = []

            done = False
            # Loop over steps
            while True:

                # For FPS counter
                step_start = time.time()

                # Show current frame
                cv2.imshow(f'Agent - preview', current_state)
                cv2.waitKey(1)

                # Predict an action based on current observation space
                if self.rl_algorithm == "Q_learning":
                    action = self.learner.get_action((current_state/255)[0], 1)

                # Step environment (additional flag informs environment to not break an episode by time limit)
                new_state, reward, done, _ = self.env.step(action)

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
            for actor in self.env.actor_list:
                actor.destroy()