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
            for h in range(-1, 2):
                for l in range(-1, 2):
                    # w = 0
                    # h = 0
                    # l = 0        
                    action = []
                    action.append(x)
                    action.append(y)
                    action.append(w)
                    action.append(h)
                    action.append(l)

                    if np.count_nonzero(action) == 0:
                        continue 

                    action_list.append(action)
                    # print action_list

observation_range = 1.5

map_size = 1.2
map_shift = map_size/2
grid_size = 0.04
map_pixel = int(map_size/grid_size)

observation_pixel = int(observation_range/grid_size)

obstacle_num = 5
observation_space = map_pixel*map_pixel + 4  # 60 x 60 + 8  2*3 + 2 + 2
# observation_space = obstacle_num*3 + 2 + 2

action_space = len(action_list)
# action_type = spaces.Box(-1, 1, shape = (action_space,))
action_type = spaces.Discrete(action_space)

REWARD_GOAL = 1
REWARD_STEP =  -0.01
REWARD_CRASH = -0.05

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
        self.ep_step = 0
        self.total_ep_reward = REWARD_GOAL
        self.goal_reached = False
        self.goal_path = False
        self.goal_counter = 0
        self.return_end = -1

        self.ep_count = 0

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
        return spaces.Box(-1, 1, shape = (5,))
        # return spaces.Discrete(len(action_list))

    def convert_state(self, robot_state):
        state = np.asarray(robot_state)
        observation = self.terrain_map.flatten()
        target_info = state[3:5]
        robot_info = state[-3:-1]

        # print(target_info, robot_info)

        state = np.append(observation, target_info)
        state = np.append(state, robot_info)
        # state = np.append(state, 0) 
        # state = np.append(state, 0)
        # state = np.append(state, [0,0,0,0,0])
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

    def reset(self, env_mode, reset_mode, save_ep):
        # print('reset')
        self.dist_pre = 1000
        self.min_obsdist_pre = 0.2
        self.obs_dist_pre = 0
        self.ep_step = 0
        self.state_pre = []
        # target_dist = 0.15+0.4*self.ep_count/3000
        target_dist = 1.5
        self.goal_path = False

        if target_dist > 1:
            target_dist = 1
        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('centauro', 'reset', [target_dist, env_mode, reset_mode, save_ep])        
        # state, reward, min_dist, obs_count, is_finish, info = self.step([0, 0, 0, 0, 0])
        state, reward_short, reward_long, is_finish, info = self.step([0, 0, 0, 0, 0])

        # print('after reset', self.dist_pre)
        self.total_ep_reward = REWARD_GOAL

        self.init_step = 1
        self.ep_count += 1
        return state

    def step(self, action): 
        self.init_step -= 0.005
        if isinstance(action, np.int32) or isinstance(action, int) or isinstance(action, np.int64):
            if action == -1:
                action = [0,0,0,0,0]
            else:            
                action = action_list[action]
                
        self.ep_step += 1
        if isinstance(action_type, spaces.Box):
            a = [0,0,0,0,0]
            a[0] = action[0]
            a[1] = action[1] 
            a[2] = action[2] 
            
            if action_space != 5:
                action = a 

        _, _, _, _, found_pose = self.call_sim_function('centauro', 'step', action)

        robot_state = []
        for i in range(10):
            _, _, robot_state, _, _ = self.call_sim_function('centauro', 'get_robot_state') # x, y, theta, h, l,   ////   tx, ty t_theta, th, tl
            if len(robot_state) != 0:
                break
        # print((robot_state))

        # obs_grid = self.get_observation_gridmap(robot_state[0], robot_state[1])
        self.get_terrain_map(robot_state[2], robot_state[3], robot_state[-3], robot_state[-2])
        # plt.clf()
        # plt.imshow(self.terrain_map)
        # plt.pause(0.01)

        #compute reward and is_finish
        reward_short, reward_long, obs_count, is_finish, info = self.compute_reward(robot_state, action, found_pose)

        state_ = self.convert_state(robot_state)

        # return state_, reward, min_dist, obs_count, is_finish, info
        return state_, reward_short, reward_long, is_finish, info

    def compute_reward(self, robot_state, action, found_pose):
        # 0,  1,  2,      3,  4,  5              -5,    -4, -3, -2, -1 
        # tx, ty, ttheta, th, tl, obs..........  theta,  h,  h  leg, min_dist   
        # _, _, min_dist, _, _ = self.call_sim_function('centauro', 'get_minimum_obs_dist') 
        info = 'unfinish'
        save_ep = False
        is_finish = False
        self.return_end = -1
        action = state = np.asarray(action)
        reward_short = 0 #REWARD_GOAL/2 #REWARD_CRASH/(self.max_length*2)
        reward_long = 0

        # weight_sum = 1
        # if action[0] == 0 and action[1] == 0:
        #     reward_short = 0
        # else:
        #     weight = [1, 2, 2, 3, 3]
        #     weight_sum = weight * np.abs(action)
        #     weight_sum = np.sum(weight_sum)

            # print(action, weight_sum)

        robot_l = robot_state[-2]
        robot_h = robot_state[-3]
        diff_l = abs(robot_l-0.13)
        diff_h = abs(robot_h)

        min_dist = robot_state[-1]

        off_pose = max(diff_l, diff_h)/0.2 * -0.5

        obs_reward = min_dist/0.2
        # reward_short = reward_short * obs_reward

        dist = robot_state[0]
        target_reward = -(dist - self.dist_pre) *5
        if target_reward <= 0:
            target_reward = -1  
        # else:
        #     target_reward = target_reward * (1 - abs(robot_state[1]))

        self.dist_pre = dist
        self.min_obsdist_pre = min_dist
        self.state_pre = robot_state

        #compute clearance
        obs_count = 0
        target_x = robot_state[5]
        target_y = robot_state[6]
        target_z = robot_state[7]
        for i in range(8, len(robot_state)-3, 4):
            obs_x = robot_state[i]
            obs_y = robot_state[i+1]
            obs_z = robot_state[i+2] * 2
            obs_h = robot_state[i+3]
            if obs_y > 0 and obs_y < target_y and abs(obs_x) < 0.2:
                obs_count += 3*obs_h*(1-abs(obs_x)/0.5)
        # print(obs_count)

        if found_pose == bytearray(b"a"):       # when collision or no pose can be found
            # is_finish = True
            # reward_short = -1
            reward_long = REWARD_CRASH
            # print('crash a')
            # reward = reward*10       
            info = 'crash'

        if found_pose == bytearray(b"c"):       # when collision or no pose can be found
            # is_finish = True
            # reward_short = -1
            reward_long = REWARD_CRASH
            target_reward = 0
            # print('crash')
            # reward = reward * 10
            info = 'crash'

        # if np.count_nonzero(action) == 0:
            # event_reward = REWARD_CRASH

        if dist < 0.2 and info != 'crash': # and diff_l < 0.02:
        # if robot_state[2] > 0.2 and info != 'crash':
            is_finish = True
            reward_long = REWARD_GOAL
            info = 'goal'

        if abs(robot_state[1]) > 1 or abs(robot_state[2]) > 0.6: # out of boundary
            is_finish = True
            reward_short = -2
            reward_long = REWARD_CRASH
            target_reward = 0
            info = 'out'
            # print('outof bound', robot_state[1])

        # if obs_count == 0:
        #     if self.goal_path == False and info != 'crash' and info != 'goal':
        #         reward_long = 0.5*REWARD_GOAL
        #         self.goal_path = True
        #         info = 'on_goal_path'
        #         # print('on goal')
        # else:
        #     if self.goal_path == True and info != 'crash' and info != 'goal':
        #         reward_long = -0.5*REWARD_GOAL
        #         self.goal_path = False
        #         info = 'off_goal_path'
        #         # print('off goal')
        #         # info = 'off_goal_path'
        #     # print('goal')

        # if info == 'goal':
        #     self.goal_counter += 1
        #     if self.goal_counter == 10:
        #         is_finish = True 
        # else:
        #     self.goal_counter = 0

        # if target_reward < 0:
        #     target_reward = -1/(dist+1)*REWARD_GOAL
        # else:
        #     target_reward = 1/(dist+1)*REWARD_GOAL
                
        # reward_short += target_reward

        # reward = event_reward + REWARD_STEP + target_reward
        # print(reward, min_dist)
        # if obs_count == 0:
        #     reward_long += target_reward
        # if info != 'crash':
        #     reward_short = reward_short/np.sum(weight_sum)

        # if obs_count == 0 and info != 'crash' and info != 'crash_a' and target_reward < 0:
        #     reward_short = REWARD_CRASH
        #     info = 'crash_a'

        reward_long += target_reward + REWARD_STEP
        # reward_short += off_pose
        return reward_short, reward_long, obs_count, is_finish, info

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

    def get_terrain_map(self, t_x, t_y, r_h, r_l):
        self.terrain_map = np.zeros((map_pixel, map_pixel, 3), np.float32)
        terrain_map_r = np.zeros((map_pixel, map_pixel), np.float32)
        terrain_map_o = np.zeros((map_pixel, map_pixel), np.float32)
        terrain_map_t = np.zeros((map_pixel, map_pixel), np.float32)

        _, _, obstacle_info, _, _ = self.call_sim_function('centauro', 'get_obstacle_info')
        for i in range(0, len(obstacle_info), 7):
            x = obstacle_info[i+0] + map_shift
            y = obstacle_info[i+1] + map_shift

            if x >= 6 or x <= 0:
                continue
            if y >= 6 or y <= 0:
                continue
            if obstacle_info[i+6] == 5:
                r = obstacle_info[i+2]
                h = obstacle_info[i+4]
                col = map_pixel - int(y/grid_size)
                row = int(x/grid_size)
                radius = int(r/grid_size )
                height = (h)/(0.5)  #255.0/0.5 * h 
                cv2.circle(terrain_map_o, (col,row), radius, height, -1)
            elif obstacle_info[i+6] == 3:
                w = int(obstacle_info[i+2]/grid_size)
                b = int(obstacle_info[i+3]/grid_size)
                col = int(map_pixel - y/grid_size)
                row = int(map_pixel - x/grid_size)

                rect = ((row, col), (w, b), obstacle_info[i+5]*180/3.14)
                box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                cv2.drawContours(terrain_map_o,[box],0,1,-1)                
                # print(obstacle_info[i+5])
                # if obstacle_info[i+5] == 0:
                #     start = (int(col-w/2), int(row-b/2))
                #     end = (int(col+w/2), int(row+b/2))
                # else:
                #     start = (int(col-b/2), int(row-w/2))
                #     end = (int(col+b/2), int(row+w/2))                    
                # cv2.rectangle(terrain_map_o, start, end, 1, -1)

        x = t_x + map_shift
        y = t_y + map_shift 
        col = map_pixel - int(y/grid_size)
        row = map_pixel - int(x/grid_size)
        radius = int(0.3/grid_size)
        cv2.circle(terrain_map_t, (col,row), radius, 1, -1)

        w = int(r_l/grid_size)
        b = int(0.3/grid_size)
        h = r_h/0.4
        cv2.circle(terrain_map_r, (int(map_shift/grid_size), int(map_shift/grid_size)), w, h, -1)
        self.terrain_map[:,:,0] = terrain_map_o
        self.terrain_map[:,:,1] = terrain_map_t
        self.terrain_map[:,:,2] = terrain_map_r

        self.terrain_map = terrain_map_o


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