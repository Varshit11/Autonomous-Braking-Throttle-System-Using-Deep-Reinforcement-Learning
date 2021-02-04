import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from .actor import Actor
from .critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer



try:
    sys.path.append(glob.glob('C:/Users/User/Pictures/CARLA_0.9.5/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
# import numpy as np
import cv2
import math
# import tensorflow as tf
# import gym
import matplotlib.pyplot as plt
from datetime import datetime


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 8.5
REPLAY_MEMORY_SIZE = 5_0000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
safe_dist = 5.0
# MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 2000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10
distance_data = []
brake_control = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]


# environment code...

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.model_32 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        vel = random.uniform(8.33, 27.77)
        self.diff = 60.0

        self.spawn_point = carla.Transform(carla.Location(-88.4,-152.1,0.0),carla.Rotation(yaw=90))
        self.spawn_point2 = carla.Transform(carla.Location(-88.4,-152.1+self.diff,0.0),carla.Rotation(yaw=90))
        self.vehicle = self.world.try_spawn_actor(self.model_3, self.spawn_point)
        self.vehicle2 = self.world.try_spawn_actor(self.model_32, self.spawn_point2)
        if self.vehicle is None:
            self.reset()
        if self.vehicle2 is None:
            self.reset()
        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.vehicle2)
        
        # # self.vehicle.apply_control(carla.VehicleControl(throttle=3.0, brake=0.0))
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        time.sleep(1)
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        temp = []
        self.vehicle2.set_velocity(carla.Vector3D(x=0,y=0,z=0))
        # think of using autopilot using command vehicle.set_autopilot(True) for some second or till I get temp length = 5.
        
        self.vehicle.set_velocity(carla.Vector3D(y=vel))
        while True:
            min_dist = [1,2,3,999999]
            t1 = self.vehicle.get_transform()
            t2 = self.vehicle2.get_transform()
            if len(temp) == 40:
                break
            else:
                temp.append(abs(t1.location.x - t2.location.x))
                temp.append(abs(t1.location.y - t2.location.y))
                temp.append(self.vehicle.get_velocity().x)
                temp.append(self.vehicle.get_velocity().y)
        return temp

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

    def get_states_dim(self):
        return(15)

    def get_num_actions(self):
        return(1)

    def step(self, action, step):
        if action <= 0:
            print("Now brake is {}".format(abs(action)))
            self.vehicle.apply_control(carla.VehicleControl(brake=float(action), throttle=0.0,
                steer=0.0))
        else:
            print("Now throttle is {}".format(abs(action)))
            self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=float(action),
                steer=0.0))

        time.sleep(0.1)
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        t = self.vehicle.get_transform()
        v2 = self.vehicle2.get_velocity()
        brake = c.brake
        throt = c.throttle
        vehicles = self.world.get_actors().filter('vehicle.*')
        # print("Now brake is {}".format(brake))
        if len(vehicles) > 1:
            distance = lambda l: [l.x-t.location.x, l.y-t.location.y, l.z-t.location.z, math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)]
            vehicles = [distance(x.get_location()) for x in vehicles if x.id != self.vehicle.id]
            vehicles_n = vehicles
            min_dist = [1,2,3,999999]
            for i in vehicles_n:
                if i[3] < min_dist[3]:
                    min_dist = i

        velo = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        print("Now distance to vehicle is {}".format(min_dist[3]))
        print("Now velocity of the vehicle is {}".format(velo))
        if len(self.collision_hist) != 0:
            reward = -1*(0.01*min_dist[3]**2+ 0.1)*abs(action) - (0.01*velo**2+ 50)
            done = True

        elif min_dist[3] > 15.0 and step > 15 and velo == 0:
            reward = -0.01*min_dist[3]**2 - 15
            done = True
        else:
            reward = +0.5
            done = False

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return [min_dist[0], min_dist[1], v.x, v.y], reward, done, None





class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, k, buffer_size = 20000, gamma = 0.99, lr = 0.00005, tau = 0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        # self.env_dim = (k,) + env_dim
        self.env_dim = (40,)
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)


    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, summary_writer):
        env = CarEnv()
        results = []
        i = 0
        # First, gather experience
        tqdm_e = tqdm(range(2000), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            old_state = np.array(old_state).reshape(40,)
            actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                # if args.render: env.render()
                # Actor picks an action (following the deterministic policy)
                a = self.policy_action(old_state)
                # Clip continuous values to be valid w.r.t. environment
                a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
                a = float(a[0])
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a, time)
                print("Now r is {}".format(r))
                # Add outputs to memory buffer
                temp_next = old_state.copy()
                temp_next[:4] = temp_next[4:8]
                temp_next[4:8] = temp_next[8:12]
                temp_next[8:12] = temp_next[12:16]
                temp_next[12:16] = temp_next[16:20]
                temp_next[16:20] = temp_next[20:24]
                temp_next[20:24] = temp_next[24:28]
                temp_next[24:28] = temp_next[28:32]
                temp_next[28:32] = temp_next[32:36]
                temp_next[32:36] = temp_next[36:40]
                temp_next[36:40] = new_state
                temp_next = np.array(temp_next).reshape(40,)
                self.memorize(old_state, a, r, done, temp_next)
                old_state = temp_next.copy()
                cumul_reward += r
                time += 1

            # since episode is over destroying actors in the scenario
            for actor in env.actor_list:
                actor.destroy()
            # Sample experience from buffer
            for i in range(50):
                states, actions, rewards, dones, new_states, _ = self.sample_batch(64)
                # Predict target q-values using target networks
                q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                # Compute critic target
                critic_target = self.bellman(rewards, q_values, dones)
                # Train both networks on sampled batch, update target networks
                self.update_models(states, actions, critic_target)
                print("learning happened")

            mean, stdev = gather_stats(self, env)
            results.append([e, mean, stdev])

            # Export results for Tensorboard
            print(cumul_reward)
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()
            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            i+=1
            if i % 10 == 0:
                df = pd.DataFrame(np.array(results))
                df.to_csv("DDPG" + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

        return results


    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
