import sys, os, math
# from rllab.spaces import Box, Discrete
import numpy as np
import time
## v-rep
from environment.vrep_plugin import vrep
import pickle as pickle
import cv2
import matplotlib.pyplot as plt

from gym import spaces

print ('import env vrep')

action_list = []
for x in range(-1, 2):
    for y in range(-1, 2):
        for w in range(-1, 2):
            # for h in range(-1, 2):
            #     for l in range(-1, 2):
            action = []
            action.append(x)
            action.append(y)
            action.append(w)
            action.append(0)
            action.append(0)

            if np.count_nonzero(action) == 0:
                continue 

            action_list.append(action)
            # print action_list

observation_range = 1.5

map_size = 1
map_shift = map_size/2
grid_size = 0.1
map_pixel = int(map_size/grid_size)

observation_pixel = int(observation_range/grid_size)

obstacle_num = 5
observation_space = map_pixel*map_pixel + 6  # 60 x 60 + 8  2*3 + 2 + 2
# observation_space = obstacle_num*3 + 2 + 2

action_space = 5 #len(action_list)


REWARD_GOAL = 100
REWARD_STEP =  -0.01
REWARD_CRASH = -1 #REWARD_STEP*10

class Simu_env():
    def __init__(self, port_num):
        self.reward_goal = REWARD_GOAL
        self.reward_crash = REWARD_CRASH
        self.action_list = action_list
        self.port_num = port_num
        self.dist_pre = 0
        self.min_obsdist_pre = 0.2
        self.obs_dist_pre = 0
        self.state_pre = []
        self.ep_init = False
        self.collide_num = 0
        self.ep_step = 0
        self.total_ep_reward = REWARD_GOAL
        self.goal_reached = False
        self.goal_counter = 0

        self.step_size = 300
        self.init_step = 0
        # self.object_num = 0
        self.terrain_map = np.zeros((map_pixel, map_pixel), np.float32)
        self.obs_grid = np.zeros((observation_pixel*2, observation_pixel*2), np.float32)

        self.connect_vrep()
        # self.get_terrain_map()
        # self.reset()

    @property
    def observation_space(self):
       return spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space,))
        # return spaces.Discrete(observation_space)
        # return observation_space

    @property
    def action_space(self):
        # return spaces.Box(-1, 1, shape = (5,))
        return spaces.Discrete(len(action_list))

    def convert_state(self, robot_state):
        state = np.asarray(robot_state)
        observation = self.terrain_map.flatten()
        target_info = state[2:4]
        robot_info = state[-4:-1]

        # print(target_info, robot_info)

        state = np.append(observation, target_info)
        state = np.append(state, robot_info)
        state = np.append(state, 0) 
        # state = np.append(state, 0)
        # state = np.append(state, [0,0,0,0,0])
        state = state.flatten()
        # print(state[-6:])
        return state

        # state = robot_state
        # state = np.asarray(state[2:-1])
        # print(len(state))
        # # sort ostacle 
        # # at, dt, ao, do, ho, rh, rl
        # obs_infos = []
        # for i in range(2, len(state)-2, 3):
        #     obs_info = (state[i], state[i+1], state[i+2])
        #     obs_infos.append(obs_info)
        #     if (len(obs_infos) == obstacle_num):
        #         break

        # sorted_info = sorted(obs_infos, key=lambda obs: obs[1])
        # sorted_info = np.asarray(sorted_info).flatten()
        # sorted_state = np.append(state[:2], sorted_info)
        # sorted_state = np.append(sorted_state, state[-2:])
        # # print('in conver state')
        # # print(len(sorted_state))

        # return sorted_state

    def reset(self, step_size, env_mode, reset_mode, save_ep):
        # print('reset')
        self.dist_pre = 1000
        self.min_obsdist_pre = 0.2
        self.obs_dist_pre = 0
        self.collide_num = 0
        self.ep_step = 0
        self.step_size = step_size
        self.state_pre = []
        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('centauro', 'reset', [observation_range*2, env_mode, reset_mode, save_ep])        
        # state, reward, min_dist, obs_count, is_finish, info = self.step([0, 0, 0, 0, 0])
        state, reward, is_finish, info = self.step([0, 0, 0, 0, 0])

        # print('after reset', self.dist_pre)
        self.ep_init = False        
        self.collide_num = 0
        self.total_ep_reward = REWARD_GOAL
        return state

    def step(self, action): 
        if isinstance(action, np.int32) or isinstance(action, int) or isinstance(action, np.int64):
            if action == -1:
                action = [0,0,0,0,0]
            else:            
                action = action_list[action]
                
        self.ep_step += 1
        # a = [0,0,0,0,0]
        # a[0] = action[0]
        # a[1] = action[1] 
        # a[2] = action[2] 
        
        # if action_space != 5:
        #     action = a 

        _, _, _, _, found_pose = self.call_sim_function('centauro', 'step', action)

        robot_state = []
        for i in range(10):
            _, _, robot_state, _, _ = self.call_sim_function('centauro', 'get_robot_state') # x, y, theta, h, l,   ////   tx, ty t_theta, th, tl
            if len(robot_state) != 0:
                break
        # print((robot_state))

        # obs_grid = self.get_observation_gridmap(robot_state[0], robot_state[1])
        self.get_terrain_map()
        # plt.clf()
        # plt.imshow(self.terrain_map, cmap='gray')
        # plt.pause(0.01)

        #compute reward and is_finish
        reward, min_dist, obs_count, is_finish, info = self.compute_reward(robot_state, action, found_pose)

        state_ = self.convert_state(robot_state)

        # return state_, reward, min_dist, obs_count, is_finish, info
        return state_, reward, is_finish, info

    def compute_reward(self, robot_state, action, found_pose):
        # 0,  1,  2,      3,  4,  5              -5,    -4, -3, -2, -1 
        # tx, ty, ttheta, th, tl, obs..........  theta,  h,  h  leg, min_dist   
        # _, _, min_dist, _, _ = self.call_sim_function('centauro', 'get_minimum_obs_dist') 
        info = 'unfinish'
        save_ep = False
        is_finish = False
        action = state = np.asarray(action)
        reward = 0 #REWARD_CRASH/(self.max_length*2)
        event_reward = 0

        robot_l = robot_state[-2]
        robot_h = robot_state[-3]
        diff_l = abs(robot_l)
        diff_h = abs(robot_h)

        min_dist = robot_state[-1]

        off_pose = 1 - max(diff_l, diff_h)/0.2
        obs_reward = min_dist/0.3

        dist = robot_state[0]
        target_reward = -(dist - self.dist_pre)/0.1
        # target_reward = target_reward/(self.max_length*4)
        if target_reward < 0:
            target_reward = 0

        # target_reward = 1 - target_reward

        # action_reward = -0.0005 * np.square(action[-2:]).sum()

        # target_reward = -(dist - self.dist_pre) * 5

        self.dist_pre = dist
        self.min_obsdist_pre = min_dist
        self.state_pre = robot_state

        #compute clearance
        obs_count = 0
        target_x = robot_state[4]
        target_y = robot_state[5]
        target_z = robot_state[6]
        for i in range(7, len(robot_state)-3, 4):
            obs_x = robot_state[i]
            obs_y = robot_state[i+1]
            obs_z = robot_state[i+2] * 2
            obs_h = robot_state[i+3]
            if obs_y > 0 and obs_y < target_y and abs(obs_x) < 0.5:
                obs_count += 3*obs_h*(1-abs(obs_x)/0.5)
        # print(obs_count)

        if found_pose == bytearray(b"a"):       # when collision or no pose can be found
            # is_finish = True
            event_reward = REWARD_CRASH
            # print('crash a')
            # reward = reward*10       
            info = 'crash'

        if found_pose == bytearray(b"c"):       # when collision or no pose can be found
            # is_finish = True
            event_reward = REWARD_CRASH
            # print('crash')
            # reward = reward * 10
            info = 'crash'

        # if np.count_nonzero(action) == 0:
            # event_reward = REWARD_CRASH

        if dist < 0.1 and info != 'crash': # and diff_l < 0.02:
            is_finish = True
            event_reward = REWARD_GOAL #self.total_ep_reward
            info = 'goal'
            # print('goal')

        if dist > 1.2: # out of boundary
            is_finish = True
            event_reward = -REWARD_GOAL
            info = 'out'
            # print('outof bound', robot_state[1])

        # if self.ep_step >= self.step_size:
        #     is_finish = True
        #     # print(self.ep_step, self.step_size)
        #     # event_reward = np.exp(-dist)*REWARD_GOAL/2
        #     info = 'nostep'
        #     # print('no step', self.ep_step)

        # if is_finish:
        #     self.ep_step = 0

        # reward = REWARD_STEP + REWARD_STEP*np.square(action).sum()
        # reward = reward * (1 + diff_l + diff_h)
        # reward = reward * target_reward

        # t = 50
        # # is_finish = False        
        # reward = -dist/200
        # if dist < 0.1 and (not self.goal_reached):
        #     reward += 1.
        #     self.goal_counter += 1
        #     if self.goal_counter > t:
        #         event_reward = REWARD_GOAL
        #         self.goal_reached = True
        #         info = 'goal'
        #         is_finish = True
        # elif dist > 0.1:
        #     self.goal_counter = 0
        #     self.goal_reached = False
        if event_reward != 0:
            reward = event_reward
        else:
            if min_dist > 0.3:
                reward = target_reward + 0.5*off_pose #event_reward #+ target_reward - dist/200
            else:
                reward = target_reward + 0.5*obs_reward

        print(reward, min_dist)
        return reward, min_dist, obs_count, is_finish, info

    ####################################  interface funcytion  ###################################
    def save_ep(self):
        self.call_sim_function('centauro', 'save_ep')
    def save_start_end_ep(self):
        self.call_sim_function('centauro', 'save_start_end_ep')

    def clear_history(self):
        self.call_sim_function('centauro', 'clear_history')

    def clear_history_leave_one(self):
        self.call_sim_function('centauro', 'clear_history_leave_one')

    def compute_dist(self, x, y):
        return math.sqrt(x*x + y*y)

    def connect_vrep(self):
        clientID = vrep.simxStart('127.0.0.1', self.port_num, True, True, 5000, 5)
        if clientID != -1:
            print ('Connected to remote API server with port: ', self.port_num)
        else:
            print ('Failed connecting to remote API server with port: ', self.port_num)

        self.clientID = clientID

        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)

    def disconnect_vrep(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(self.clientID)
        print ('Program ended')

    def get_observation_gridmap(self, robot_x, robot_y):
        x = robot_x + map_shift
        y = robot_y + map_shift
        c_row = self.terrain_map.shape[0] - int(y/grid_size)
        c_col = int(x/grid_size)

        sub_start_r = 0
        sub_end_r = observation_pixel*2
        sub_start_c = 0
        sub_end_c = observation_pixel*2

        start_r = c_row - observation_pixel
        end_r = c_row + observation_pixel

        start_c = c_col - observation_pixel
        end_c = c_col + observation_pixel

        if start_r < 0:
            sub_start_r = -start_r
            start_r = 0
        if end_r >= self.terrain_map.shape[0]:
            sub_end_r = self.terrain_map.shape[0] - start_r - 1
            end_r = self.terrain_map.shape[0] -1

        if start_c < 0:
            sub_start_c = -start_c
            start_c = 0
        if end_c >= self.terrain_map.shape[1]:
            sub_end_c = self.terrain_map.shape[1] - start_c - 1
            end_c = self.terrain_map.shape[1] -1

        # print(x, y, c_row, c_col)
        # print(start_r, end_r, start_c, end_c)
        # print(sub_start_r, sub_end_r, sub_start_c, sub_end_c)
        self.obs_grid.fill(0)
        # self.obs_grid[sub_start_r:sub_end_r, sub_start_c:sub_end_c] = self.terrain_map[start_r:end_r, start_c:end_c]

        return self.obs_grid 

    def get_terrain_map(self):
        self.terrain_map = np.zeros((map_pixel, map_pixel), np.float32)
        _, _, obstacle_info, _, _ = self.call_sim_function('centauro', 'get_obstacle_info')
        for i in range(0, len(obstacle_info), 5):
            x = obstacle_info[i+0] + map_shift
            y = obstacle_info[i+1] + map_shift

            if x >= 5 or x <= 0:
                continue
            if y >= 5 or y <= 0:
                continue
            r = obstacle_info[i+2]
            h = obstacle_info[i+4]

            row = self.terrain_map.shape[0] - int(y/grid_size)
            col = int(x/grid_size)
            radius = int(r/grid_size )
            height = (h-0.15)/(0.5-0.15)  #255.0/0.5 * h 
            cv2.circle(self.terrain_map, (col,row), radius, height, -1)
        # print('max', self.terrain_map.max())
        # ## for boundaries
        # boundary_height = 1
        # cv2.line(self.terrain_map, (0, 0), (0, self.terrain_map.shape[1]), 1, 3)
        # cv2.line(self.terrain_map, (0, 0), (self.terrain_map.shape[0], 0), 1, 3)
        # cv2.line(self.terrain_map, (0, self.terrain_map.shape[1]), (self.terrain_map.shape[0], self.terrain_map.shape[1]), boundary_height, 3)
        # cv2.line(self.terrain_map, (self.terrain_map.shape[0], 0), (self.terrain_map.shape[0], self.terrain_map.shape[1]), boundary_height, 3)

        # # ## for two static obstacles
        # # -3.4, -1, 2.6, -1      -2.6, 1, 3.4, 1
        # p1_r = self.terrain_map.shape[0] - int((-1 + map_shift)/grid_size)
        # p1_c = int((-1.9 + map_shift)/grid_size)
        # p2_r = self.terrain_map.shape[0] - int((-1 + map_shift)/grid_size)
        # p2_c = int((1.1 + map_shift)/grid_size)        

        # p3_r = self.terrain_map.shape[0] - int((1 + map_shift)/grid_size)
        # p3_c = int((-1.1 + map_shift)/grid_size)
        # p4_r = self.terrain_map.shape[0] - int((1 + map_shift)/grid_size)
        # p4_c = int((1.9 + map_shift)/grid_size)     
        # cv2.line(self.terrain_map, (p1_c, p1_r), (p2_c, p2_r), boundary_height, 1)
        # cv2.line(self.terrain_map, (p3_c, p3_r), (p4_c, p4_r), boundary_height, 1)

        # np.save("./data/auto/map", self.terrain_map)
        # # mpimg.imsave('./data/auto/map.png', self.terrain_map)
        # print('map updated!!!!!')
        # # self.terrain_map = cv2.imread('./data/map.png')
    ########################################################################################################################################
    ###################################   interface function to communicate to the simulator ###############################################
    def call_sim_function(self, object_name, function_name, input_floats=[]):
        inputInts = []
        inputFloats = input_floats
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID, object_name,vrep.sim_scripttype_childscript,
                    function_name, inputInts, inputFloats, inputStrings,inputBuffer, vrep.simx_opmode_blocking)

        # print 'function call: ', self.clientID
        return res, retInts, retFloats, retStrings, retBuffer


# env = Simu_env(20000)
# env.reset()
# env.get_terrain_map()
# img = env.get_observation_gridmap(0, 0)
# plt.imshow(env.obs_grid, cmap='gray')
# plt.imshow(env.terrain_map, cmap='gray')
# plt.show()

# action = [0,0,0,0,0.1]
# for i in range(100):
#     for j in range(5):
#         a = (np.random.rand()-0.5) * 2
#         action[j] = a

#     s_, r, done, _ = env.step(action)
#     print (r, done)

# print (env.action_space())
# print (env.observation_space())