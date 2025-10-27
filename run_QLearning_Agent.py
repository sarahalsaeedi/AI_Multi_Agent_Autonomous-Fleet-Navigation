""" Running CAU Q Learning Agent"""

from __future__ import print_function
from learning.QLearning import Q_Learning
from env.carla.agents.navigation.basic_agent import BasicAgent
from localStoragePy import localStoragePy

import argparse
import collections
import datetime
import glob
import logging
import math
import json
import os
import random
import re
import sys
import tensorflow as tf
import weakref
import time
import matplotlib.pyplot as plt

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.obstacle_sensor = None
        self.camera_manager = None
        self.is_obstacle = False
        self.control_option = ControlOptions()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def set_control_options(self, control, value):
        if control == "obs":
            self.is_obstacle = value

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        blueprint = self.world.get_blueprint_library().filter("model3")[0]
        # Spawn the player.
        #print("Spawning the player")

        if self.player is not None:

            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            print("is non nonr Spawning the player", spawn_point)
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()[15]
            spawn_points2 = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points2) if spawn_points else carla.Transform()
            #print("Spawning the player", spawn_point)
            self.player = self.world.try_spawn_actor(blueprint, spawn_points)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(
            self.player, self.hud, self.control_option, self)
        self.gnss_sensor = GnssSensor(
            self.player, self.hud, self.control_option)
        self.obstacle_sensor = ObstaclesSensor(
            self.player, self.hud, self.control_option)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)

        actor_type = get_actor_display_name(self.player)
        actor_list = self.world.get_actors()
        # print(actor_list)
        self.hud.notification(actor_type)

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock, "")

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.obstacle_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 16 if os.name == 'nt' else 18)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.reward = 0
        self.penalty = 0
        self.steps = 0
        self.solid = 0
        self.broken = 0
        self.sbroken = 0
        self.brokens = 0
        self.lane = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock, option):
        """HUD method for every tick"""
        if option == "Steps":
            self.steps += 1
        elif option == "Reward":
            self.reward += 1
        elif option == "Solid":
            self.penalty += 1
            self.solid += 1
        elif option == "BrokenSolid":
            self.penalty += 1
            self.brokens += 1
        elif option == "SolidBroken":
            self.penalty += 1
            self.sbroken += 1
        elif option == "Lane":
            self.penalty += 1
            self.lane += 1
        elif option == "Broken":
            self.penalty += 1
            self.broken += 1

        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        #obshist = len(world.obstacle_sensor.get_obstacles_history())
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (
                transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1, 1),
                ('Brake:', control.brake, 0.0, 1.0),
                #('Reverse:', control.reverse),
                #('Hand brake:', control.hand_brake),
                #('Manual:', control.manual_gear_shift),
                #'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)
            ]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            # 'Agent Took %d steps ' % self.steps,

            # 'Total Rewards: %d' % self.reward,

            # 'Total Penalty: %d' % self.penalty,
            # '',


            # 'Total Crossed Solid Lines: %d' % self.solid,
            # 'Total Crossed Broken Lines: %d' % self.broken,
            # 'Total Crossed SolidBroken',
            # 'Lines:  %d' % self.sbroken,
            # 'Total Crossed BrokenSolid',
            # 'Lines:  %d' % self.brokens,
            # '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x)
                    for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=4.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((310, self.dim[1]))
            info_surface.set_alpha(150)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 0, 0), seconds=6.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- ControlOptions -----------------------------------------------------------
# ==============================================================================

# This class for controlling different data from the sensors
class ControlOptions:
    def __init__(self):
        self.is_obstacle = False
        self.obs_time = time.time()
        self.lane_time = time.time()
        self.is_sidewalk = False
        self.lane_line = False
        self.red_traffic = False
        self.is_stop = False

    def set_control_option(self, control, value, obs_time):
        if control == "obs":
            self.is_obstacle = value
            self.obs_time = obs_time
        elif control == "sidewalk":
            self.is_sidewalk = value
        elif control == "lane":
            self.lane_line = value
            self.lane_time = obs_time
        elif control == "Red":
            self.red_traffic = value
        elif control == "stop":
            self.is_stop = value

    def get_control_option(self, control):
        if control == "obs":
            return self.is_obstacle, self.obs_time
        elif control == "sidewalk":
            return self.is_sidewalk, self.obs_time
        elif control == "lane":
            return self.lane_line, self.lane_time
        elif control == "Red":
            return self.red_traffic, self.obs_time
        elif control == "stop":
            return self.is_stop, self.obs_time

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- ObstaclesSensor --------------------------------------------------------
# ==============================================================================

class ObstaclesSensor(object):
    """Class for Obstacle sensors"""

    def __init__(self, parent_actor, hud, control_option):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.world = self._parent.get_world()

        bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute("hit_radius", str(0))
        bp.set_attribute("distance", str(30))
        bp.set_attribute("only_dynamics", 'true')
        bp.set_attribute("debug_linetrace", 'true')
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self._parent,
                                             attachment_type=carla.AttachmentType.Rigid)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstaclesSensor._on_detect(
            weak_self, event, control_option))

    def get_obstacles_history(self):
        """Gets the history of obstacles"""
        return self.history

    @staticmethod
    def _on_detect(weak_self, event, control_option):
        """On obstacles method"""
        self = weak_self()
        ObstaclesSensor.measure_obs_distance(self, event, control_option)
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        rounded_distance = round(float(event.distance), 2)
        #self.hud.notification('obstacle: %r with distance of %s m' % (actor_type, rounded_distance))
        distance = event.distance
        self.history.append((actor_type, distance))
        if len(self.history) > 4000:
            self.history.pop(0)

    # On each Obs detected we measure the distance and take the right decision
    def measure_obs_distance(self, event, control_option):
        actor_name = get_actor_display_name(event.other_actor)
        actor_type = event.other_actor.type_id
        is_road = False
        if "roadline" or "road" or "ride" in actor_type:
            # always ignore the road as obstacle
            is_road = True
            control_option.set_control_option("obs", False, time.time())

        if not "roadline" or "road" or "ride" in event.other_actor.type_id:
            # Everything other than the road
            if event.distance < 15:
                # send the agent flag to stop driving
                #print("force stop")
                self.hud.notification(
                    'obstacle: %r is near -> Emergency Stop' % actor_name)

                control_option.set_control_option("obs", True, time.time())

            else:
                # keep going when the obs gone away
                control_option.set_control_option("obs", False, time.time())
                self.hud.notification(
                    'obstacle: %r moved -> Save to go' % actor_name)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud, control_option, world):
        """Constructor method"""
        self.world = world
        self.sensor = None
        self.lane = False
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(
            weak_self, event, parent_actor, control_option, self.world))

    def set_lane_history(self, lane):
        """Gets the history of lanes"""
        self.lane = lane

    def get_lane_history(self):
        """Gets the history of lanes"""
        return self.lane

    @staticmethod
    def _on_invasion(weak_self, event, parent_actor, control_option, world):
        """On invasion method"""
        print("sensor lane")
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        lane_type = str(text[0]).replace("'", '')
        self.hud.notification('Made a wrong lane ')
        self.hud.tick(world, pygame.time.Clock(), "Lane")
        control_option.set_control_option("lane", lane_type, time.time())
        self.lane = True
        #self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor, hud, control_option):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.world = self._parent.get_world()
        blueprint = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = self.world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                             attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(
            weak_self, hud, event, self.world, control_option, parent_actor))

    @staticmethod
    def _on_gnss_event(weak_self, hud, event, world, control_option, parent_actor):
        """GNSS method"""

        self = weak_self()
        map = world.get_map()
        if parent_actor.is_alive:
            waypoint = map.get_waypoint(self._parent.get_location(), project_to_road=True,
                                        lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            landmarks = waypoint.get_landmarks(20.0, True)
            opposite_lane = None
            if len(landmarks) > 0:
                opposite_lane = landmarks[0].orientation

            if parent_actor.is_at_traffic_light():
                traffic_light = parent_actor.get_traffic_light()
                if str(traffic_light.get_state()) == "Red" and str(opposite_lane) == "Positive":
                    hud.notification('Traffic Light is Red -> Emergency Stop')

                    control_option.set_control_option(
                        "Red", False, time.time())
                else:
                    control_option.set_control_option(
                        "Red", False, time.time())
                    #hud.notification('Traffic Light is Green -> Continue')
            else:
                control_option.set_control_option("Red", False, time.time())
                #hud.notification('Traffic Light is Green -> Continue')

            lane_type = waypoint.lane_type
            if str(lane_type) == "Sidewalk":
                hud.notification('Crossed Sidewalk -> Penalty')
                control_option.set_control_option(
                    "sidewalk", True, time.time())
            else:
                control_option.set_control_option(
                    "sidewalk", False, time.time())
        else:
            print("parent_actor.is_not_alive", parent_actor.is_alive)

        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self.front_camera = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(
                lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.front_camera = array
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # if self.recording:
            #image.save_to_disk('_out/%08d' % image.frame)


def run_agent(args, reward, learner, load_path, training_path, agent_action, episodes, localStorage, storedParmas):
    """ Main Agent Runner"""
    if storedParmas:
        all_costs = storedParmas["all_costs"]
        all_lane = storedParmas["all_lane"]
        all_wrong_turn = storedParmas["all_wrong_turn"]
        all_success = storedParmas["all_success"]
        all_steps = storedParmas["all_steps"]
    else:
        all_costs = []
        all_lane = []
        all_wrong_turn = []
        all_success = []
        all_steps = []

    print("---------------", storedParmas)
    steps = 0
    reward = None
    total_sucess = 0
    total_reward = 0
    total_wrong_lane = 0
    total_wrong_turn = 0
    for running_episode in range(episodes):
        pygame.init()
        pygame.font.init()
        world = None
        tot_target_reached = 0
        num_min_waypoints = 21
        current_state = None

        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(2.0)

            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)
            world = World(client.get_world(), hud, args)

            # client.get_world().set_weather(carla.WeatherParameters.HardRainSunset)
            controller = KeyboardControl(world)

            final_spawn_point = world.map.get_spawn_points()[2]
            agent = BasicAgent(world.player)

            waypoints = agent.set_destination((final_spawn_point.location.x,
                                               final_spawn_point.location.y,
                                               final_spawn_point.location.z))
            print("you have %r waypoints to destination" % len(waypoints))
            agentControl = agent.run_step()

            clock = pygame.time.Clock()
            clock_run = time.time()

            # ------------------------------------

            score = 0  # store for the scores of an episode
            # episode = 1  # episode counter

            tf.compat.v1.disable_eager_execution()
            writer = tf.compat.v1.summary.FileWriter(
                training_path, tf.compat.v1.get_default_graph())

            if not True and load_path is not None and os.path.isdir(load_path):
                learner.load_model(load_path, '/model.pkl')
                print("load_model")

            time.sleep(2)
            while world.camera_manager.front_camera is None:
                time.sleep(0.01)

        # ------------------------------------
            # print(reward)
            # Resulted list for the plotting Episodes via Steps
            steps = 0
            total_reward = 0
            total_wrong_lane = 0
            total_wrong_turn = 0
            # Summed costs for all episodes in resulted list

            i = 0
            cost = 0
            done = False
            waypoint_number = 0
            waypoint_steering = 0
            avg_steer = 0.0
            steer_loss = False
            while not done:
                steps += 1
                is_stop, obs_time3 = world.control_option.get_control_option(
                    "stop")
                clock.tick_busy_loop(60)
                if controller.parse_events():
                    return

                # As soon as the server is ready continue!
                if not world.world.wait_for_tick(10.0):
                    continue

                world.world.wait_for_tick(10.0)
                world.tick(clock)

                if not is_stop:
                    hud.tick(world, clock, "Steps")
                total_wrong_lane = hud.lane
                # print(hud.lane)
                world.render(display)
                pygame.display.flip()
                agentControl = agent.run_step()
                waypoint_steering += agentControl.steer
                waypoint_number += 1
                if waypoint_number == 5:  # get avarage steer degree of the next 5 waypoints
                    avg_steer = waypoint_steering/5
                    waypoint_steering = 0
                    waypoint_number = 0

                current_location = world.player.get_location()
                vel = world.player.get_velocity()
                speed = (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

                current_state = np.array(
                    [current_location, speed, agentControl.steer])

                world.player.apply_control(carla.VehicleControl(
                    throttle=0.2, brake=0.0, steer=agentControl.steer))
                action = learner.get_action(current_state, reward)
                error_loss = agent_action[action] - avg_steer
                if error_loss > 0.05 or error_loss < -0.05:
                    steer_loss = True

                    total_wrong_turn += 1
                else:
                    steer_loss = False
                option = check_conditions(world)

                agent_control, new_state, new_reward, done1, apply, _ = run_step(
                    hud, clock_run, world, agent_action[action], steer_loss, option, clock)
                total_reward += new_reward
                if done1:
                    all_success.append(0)
                if apply:
                    if agent.done():
                        print("you did it! ", steps)
                        hud.notification('Reach Destination --> Great Job!')
                        agent_control = agentControl
                        total_sucess += 1
                        all_success.append(1)
                        done1 = True

                    world.player.apply_control(agent_control)
                    writer.add_summary(tf.compat.v1.Summary(
                        value=[tf.compat.v1.Summary.Value(tag='Score per Episode', simple_value=score)]),
                        running_episode
                    )
                    if True and running_episode % 10 == 0:
                        #print("running_episode", running_episode)
                        learner.save_q_table(training_path + '/model.pkl')
                        try:
                            learner.update_target_model()
                        except AttributeError:
                            ...
                    running_episode += 1
                    score = 0

                    current_state = new_state
                    reward = new_reward

                    done = done1
                else:
                    done = False
            print("this is episode", running_episode)

            print("this is steps", steps)
            all_costs.append(total_reward)
            all_lane.append(total_wrong_lane)
            all_wrong_turn.append(total_wrong_turn)
            all_steps.append(steps)
            print("this is total reward", total_reward)
            print("this is total lane", total_wrong_lane)
            print("this is total turns", total_wrong_turn)
            print("total sucess", total_sucess)
            print("total steps", steps)

            storeEpisodeParams(localStorage, episodes, running_episode, all_costs, all_lane, all_wrong_turn,
                               all_success, all_steps)
        finally:
            if world is not None:
                world.destroy()

            pygame.quit()
            # return reward

    print("total sucess", (total_sucess/episodes) * 100)
    learner.print_q()
    plotting_result(all_costs, all_lane, all_wrong_turn,
                    all_success, all_steps, episodes)


def storeEpisodeParams(localStorage, episodes, running_episode, all_costs, all_lane, all_wrong_turn,
                       all_success, all_steps):
    print("SAVE PARAMS", running_episode)
    storedParams = {
        "episodes": episodes,
        "running_episode": running_episode,
        "all_costs": all_costs,
        "all_lane": all_lane,
        "all_wrong_turn": all_wrong_turn,
        "all_success": all_success,
        "all_steps": all_steps
    }
    localStorage.setItem("storedParams", json.dumps(storedParams))


def plotting_result(all_costs, all_lane, all_wrong_turn, total_sucess, all_steps, episodes):
    #
    #f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    #
    plt.figure()
    plt.plot(np.arange(len(all_costs)), all_costs, 'b')
    plt.title('Episode via Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')

    #
    plt.figure()
    plt.plot(np.arange(len(all_lane)), all_steps, label="Steps")
    #plt.plot(np.arange(len(all_lane)), all_lane, label="Wrong Lane")
    plt.title('Total Steps')
    plt.xlabel('Episode')
    plt.ylabel('Total Lane')
    # plt.legend()

    #
    plt.figure()
    #plt.plot(np.arange(len(all_lane)), all_steps, label="Steps")
    plt.plot(np.arange(len(all_lane)), all_lane, label="Wrong Lane")
    plt.title('Total Lane Crossed')
    plt.xlabel('Episode')
    plt.ylabel('Total Lane')
    #

    plt.figure()
    plt.plot(np.arange(len(all_wrong_turn)), all_steps, label="Steps")
    plt.plot(np.arange(len(all_wrong_turn)),
             all_wrong_turn, label="Wrong Turns")
    plt.title('Steps via Turns')
    plt.xlabel('Episode')
    plt.legend()
    #
    plt.figure()

    #plt.plot(np.arange(5), total_sucess, 'o')
    xaxis = np.arange(len(total_sucess))
    yaxis = np.array(total_sucess)
    plt.step(xaxis, yaxis)
    plt.title('Destination Point B Reached')
    plt.xlabel('Episode')

    print("all_steps", all_steps)
    # Showing the plots
    plt.show()


# this function to check the status of each controlling attributes "sensors att"
def check_conditions(world):
    obs_Control, obs_time = world.control_option.get_control_option("obs")
    red_traffic, obs_time = world.control_option.get_control_option("Red")
    lane, lane_time = world.control_option.get_control_option("lane")
    #print("old time", lane, lane_time)
    option = None
    if obs_Control:
        option = "Stop"
        world.control_option.set_control_option("stop", True, time.time())
        if time.time() - obs_time > 10:
            world.control_option.set_control_option("obs", False, time.time())
            world.control_option.set_control_option("stop", False, time.time())
            obs_Control, obs_time = world.control_option.get_control_option(
                "obs")
            option = "Go"

    if red_traffic:
        option = "Stop"
        world.control_option.set_control_option("stop", True, time.time())
    elif not obs_Control and not red_traffic:
        option = "Go"
        world.control_option.set_control_option("stop", False, time.time())

    # s = time.time() - lane_time
    # # print(s)
    # if lane and s > 1:
    #     print("old lane,,, change")
    #     #world.control_option.set_control_option("lane", False, time.time())
    # else:
    #     print(s)
    return option


def run_step(hud, clock_run, world, action, steer_loss, option, clock):
    control, apply = set_action(world, action, option)
    colhist = world.collision_sensor.get_collision_history()
    lanhist = world.lane_invasion_sensor.get_lane_history()

    is_sidewalk, obs_time = world.control_option.get_control_option("sidewalk")
    lane_line, lane_time2 = world.control_option.get_control_option("lane")
    is_stop, obs_time3 = world.control_option.get_control_option("stop")
    # # print(lane_line)
    # if lanhist:
    #     if not is_stop:
    #         hud.tick(world, clock, "Lane")
    #     done = False
    #     #reward = -2
    if len(colhist) != 0:
        done = True
        reward = -9
    elif is_sidewalk:
        done = True
        reward = -7
    elif lane_line == 'Solid':

        hud.notification('Crossed Solid Line -> Penalty')
        if not is_stop:
            hud.tick(world, clock, "Solid")
        done = False
        reward = -6
    elif lane_line == 'SolidBroken':
        hud.notification('Crossed SolidBroken Line -> Penalty')
        if not is_stop:
            hud.tick(world, clock, "SolidBroken")
        done = False
        reward = -5
    elif lane_line == 'BrokenSolid':
        hud.notification('Crossed BrokenSolid Line -> Penalty')
        if not is_stop:
            hud.tick(world, clock, "BrokenSolid")
        done = False
        reward = -5
    elif lane_line == 'Broken':
        hud.notification('Crossed Broken Line -> Penalty')
        if not is_stop:
            hud.tick(world, clock, "Broken")
        done = False
        reward = -3

    elif steer_loss:
        done = False
        reward = -8
    elif not steer_loss:
        done = False
        reward = 2
    else:
        done = False
        reward = 1
        if not is_stop:
            hud.tick(world, clock, "Reward")
            #hud.notification('Good Move -> Reward')

    if clock_run + 150 < time.time():
        print("you reached time")
        done = True
    world.control_option.set_control_option("lane", None, time.time())
    current_location = world.player.get_location()
    vel = world.player.get_velocity()
    speed = (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

    new_state = np.array([current_location, speed, control.steer])
    return control, new_state, reward, done, apply, None


def set_action(world, action, option):
    control = None
    apply = False
    if option == "Stop":
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        apply = True

    elif option == "Go":
        apply = True
        control = carla.VehicleControl()
        control.steer = action
        control.throttle = 0.4
        control.brake = 0.0
        control.hand_brake = False
        apply = True

    return control, apply

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description=' CAU Q Learning Agent ')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    learner = Q_Learning(range(11),
                         epsilon=1,
                         descending_epsilon=True,
                         descend_epsilon_until=0.95,
                         alpha=0.2)
    agent_action = [1.0, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -0.1]
    load_path = './q_training/train_QAgent_train_Q_Learning'
    training_path = './q_training/train_QAgent' + ('_train_' if True else 'run_') \
        + type(learner).__name__
    print(__doc__)
    reward = 1
    localStorage = localStoragePy('QN-CARLA', 'json')
    # localStorage.removeItem("storedParams")
    storedParams = localStorage.getItem("storedParams")
    storedEpisode = 0
    readsJson = None
    if storedParams:
        readsJson = json.loads(storedParams)
        storedEpisode = int(9 + readsJson["running_episode"])
    else:
        storedEpisode = 50

    try:
        run_agent(args, reward, learner, load_path,
                  training_path, agent_action, episodes=1, localStorage=localStorage, storedParmas=readsJson)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
