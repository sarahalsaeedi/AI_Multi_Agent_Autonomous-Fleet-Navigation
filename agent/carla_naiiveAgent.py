#!/usr/bin/env python
# This is only basic spawned car on random road
# Without containing any controling or doing any actions
# It's just for testing.
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

import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480
EPS = 100
R_p_s = 10


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    print(i3)
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0


def main():
    actor_list = []

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()\

        bp = blueprint_library.filter('model3')[0]
        print(bp)
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        # to-do 1 these actions should be saved as default actions for the naive agent in the memory
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        blueprint = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        # to-do 3 Create a reward function for the states of the agent
        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

        # add sensor to list of actors
        actor_list.append(sensor)

        # do something with this sensor
        # to-do 2 for loop for episodes + [the episode will end at collision, reach a certain destination, number of episodes are reached to an end ]
        sensor.listen(lambda data: process_img(data))
        time.sleep(30)
    finally:

        print('destroying actors')
        sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    main()
