from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

#from IPython.display import HTML, display
#import base64
#import glob
#import io
import cv2
import glfw
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import imageio
import types
import time

import os
file_path = os.path.dirname(os.path.abspath(__file__))

from reward_functions import *
from transform_utils import euler2quat, quat2mat

class pushpixel_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, task=0, reward_type='binary', goal_type='circle', hide_goal=False, separate=False):
        self.env = ur5_env 
        self.num_blocks = num_blocks
        self.num_bins = 8

        self.task = task # 0: Reach / 1: Push / 2: Push-feature
        self.mov_dist = mov_dist
        self.block_spawn_range_x = [-0.25, 0.25] #[-0.20, 0.20] #[-0.25, 0.25]
        self.block_spawn_range_y = [-0.12, 0.3] #[-0.10, 0.30] #[-0.15, 0.35]
        self.block_range_x = [-0.31, 0.31] #[-0.25, 0.25]
        self.block_range_y = [-0.18, 0.36] #[-0.15, 0.35]
        self.eef_range_x = [-0.35, 0.35]
        self.eef_range_y = [-0.22, 0.40]
        self.z_push = 1.045
        self.z_prepush = self.z_push + 0.12 #2.5 * self.mov_dist
        self.z_collision_check = self.z_push + 0.025
        self.time_penalty = 0.02 #0.1
        self.max_steps = max_steps
        self.step_count = 0
        self.threshold = 0.05

        self.reward_type = reward_type
        self.goal_type = goal_type
        self.hide_goal = hide_goal
        self.separate = separate

        self.init_pos = [0.0, -0.23, 1.4]
        self.background_img = imageio.imread(os.path.join(file_path, 'background.png')) / 255.
        self.goals = []

        self.cam_id = 1
        self.cam_theta = 30 * np.pi / 180 # 0.
        # cam_mat = self.env.sim.data.get_camera_xmat("rlview")
        # cam_pos = self.env.sim.data.get_camera_xpos("rlview")

        self.colors = np.array([
            [0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]
            # [0.6784, 1.0, 0.1843], [0.93, 0.545, 0.93], [0.9686, 0.902, 0]
            ])

        self.pre_selected_objects = None
        self.init_env()
        # self.env.sim.forward()

    def set_num_blocks(self, num_blocks):
        self.num_blocks = num_blocks

    def get_reward(self, info):
        if self.task == 0:
            return reward_reach(self)
        elif self.separate:
            if self.reward_type=="binary":
                return reward_binary_separate(self, info)
            elif self.reward_type=="inverse":
                return reward_inverse_separate(self, info)
            elif self.reward_type=="linear":
                return reward_linear_separate(self, info)
            elif self.reward_type=="sparse":
                return reward_sparse_separate(self, info)
            elif self.reward_type=="new":
                return reward_new_separate(self, info)
        else:
            if self.reward_type=="binary":
                return reward_push_binary(self, info)
            elif self.reward_type=="inverse":
                return reward_push_inverse(self, info)
            elif self.reward_type=="linear":
                return reward_push_linear(self, info)
            elif self.reward_type=="sparse":
                return reward_push_sparse(self, info)
            elif self.reward_type=="new":
                return reward_push_new(self, info)
            elif self.reward_type=="linear_penalty":
                return reward_push_linear_penalty(self, info)
            elif self.reward_type=="linear_maskpenalty":
                return reward_push_linear_maskpenalty(self, info)

    def init_env(self, scenario=-1):
        self.env._init_robot()
        self.env.selected_objects = self.env.selected_objects[:self.num_blocks]
        range_x = self.block_spawn_range_x
        range_y = self.block_spawn_range_y
        threshold = 0.10 #0.12

        # mesh grid 
        x = np.linspace(-0.4, 0.4, 7)
        y = np.linspace(0.4, -0.2, 5)
        xx, yy = np.meshgrid(x, y, sparse=False)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # init all blocks
        if self.pre_selected_objects is None:
            for obj_idx in range(self.env.num_objects):
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]
        else:
            for obj_idx in self.pre_selected_objects:
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]

        if self.goal_type=='circle':
            check_feasible = False
            while not check_feasible:
                self.goal_image = deepcopy(self.background_img)
                self.goals = []
                init_poses = []
                for i, obj_idx in enumerate(self.env.selected_objects):
                    check_init_pos = False
                    check_goal_pos = False
                    if i < self.num_blocks:
                        while not check_goal_pos:
                            gx = np.random.uniform(*range_x)
                            gy = np.random.uniform(*range_y)
                            if i==0:
                                break
                            check_goals = (i == 0) or (np.linalg.norm(np.array(self.goals) - np.array([gx, gy]), axis=1) > threshold).all()
                            if check_goals:
                                check_goal_pos = True
                        self.goals.append([gx, gy])
                        cv2.circle(self.goal_image, self.pos2pixel(gx, gy), 1, self.colors[i], -1)
                        while not check_init_pos:
                            tx = np.random.uniform(*range_x)
                            ty = np.random.uniform(*range_y)
                            if i==0:
                                break
                            check_inits = (i == 0) or (np.linalg.norm(np.array(init_poses) - np.array([tx, ty]), axis=1) > threshold).all()
                            check_overlap = (np.linalg.norm(np.array(self.goals) - np.array([tx, ty]), axis=1) > threshold).all()
                            if check_inits and check_overlap:
                                check_init_pos = True
                        init_poses.append([tx, ty])
                        tz = 0.95
                        self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [tx, ty, tz]
                        x, y, z, w = euler2quat([0, 0, np.random.uniform(2*np.pi)])
                        self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]

                    else:
                        self.env.sim.data.qpos[7*obj_idx + 12: 7*obj_idx + 15] = [0, 0, 0]
                self.env.sim.step()
                check_feasible = self.check_blocks_in_range()

            if self.env.data_format=='NCHW':
                self.goal_image = np.transpose(self.goal_image, [2, 0, 1])

        elif self.goal_type=='pixel':
            check_feasible = False
            while not check_feasible:
                self.goals = []
                init_poses = []
                goal_ims = []
                for i, obj_idx in enumerate(self.env.selected_objects):
                    check_init_pos = False
                    check_goal_pos = False
                    if i < self.num_blocks:
                        while not check_goal_pos:
                            gx = np.random.uniform(*range_x)
                            gy = np.random.uniform(*range_y)
                            if i == 0:
                                break
                            check_goals = (i == 0) or (np.linalg.norm(np.array(self.goals) - np.array([gx, gy]), axis=1) > threshold).all()
                            if check_goals:
                                check_goal_pos = True
                        self.goals.append([gx, gy])
                        zero_array = np.zeros([self.env.camera_height, self.env.camera_width])
                        cv2.circle(zero_array, self.pos2pixel(gx, gy), 1, 1, -1)
                        goal_ims.append(zero_array)

                        while not check_init_pos:
                            tx = np.random.uniform(*range_x)
                            ty = np.random.uniform(*range_y)
                            check_inits = (i == 0) or (np.linalg.norm(np.array(init_poses) - np.array([tx, ty]), axis=1) > threshold).all()
                            check_overlap = (np.linalg.norm(np.array(self.goals) - np.array([tx, ty]), axis=1) > threshold).all()
                            if check_inits and check_overlap:
                                check_init_pos = True
                        init_poses.append([tx, ty])
                        tz = 0.95
                        self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [tx, ty, tz]
                        x, y, z, w = euler2quat([0, 0, np.random.uniform(2*np.pi)])
                        self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
                    else:
                        self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [0, 0, 0]
                self.env.sim.step()
                check_feasible = self.check_blocks_in_range()

            goal_image = np.concatenate(goal_ims)
            self.goal_image = goal_image.reshape([self.num_blocks, self.env.camera_height, self.env.camera_width])
            if self.env.data_format=='NHWC':
                self.goal_image = np.transpose(self.goal_image, [1, 2, 0])

        elif self.goal_type=='block':
            check_feasible = False
            while not check_feasible:
                goals, inits = self.generate_scene(scenario)
                for i, obj_idx in enumerate(self.env.selected_objects):
                    if i>=self.num_blocks:
                        self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]
                        continue
                    gx, gy = goals[i]
                    gz = 0.95
                    euler = np.zeros(3) 
                    euler[2] = 2*np.pi * np.random.random()
                    if self.env.real_object:
                        if obj_idx in self.env.obj_orientation:
                            euler[:2] = np.pi * np.array(self.env.obj_orientation[obj_idx])
                        else:
                            euler[:2] = [0, 0]
                    x, y, z, w = euler2quat(euler)
                    self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [gx, gy, gz]
                    self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
                for i in range(50):
                    self.env.sim.step()
                    if self.env.render: self.env.sim.render(mode='window')
                check_goal_feasible = self.check_blocks_in_range()
                self.goal_image = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)

                for i, obj_idx in enumerate(self.env.selected_objects):
                    if i>=self.num_blocks:
                        continue
                    tx, ty = inits[i]
                    tz = 0.95
                    euler = np.zeros(3) 
                    euler[2] = 2*np.pi * np.random.random()
                    if self.env.real_object:
                        if obj_idx in self.env.obj_orientation:
                            euler[:2] = np.pi * np.array(self.env.obj_orientation[obj_idx])
                        else:
                            euler[:2] = [0, 0]
                    x, y, z, w = euler2quat(euler)
                    self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [tx, ty, tz]
                    self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
                for i in range(50):
                    self.env.sim.step()
                    if self.env.render: self.env.sim.render(mode='window')
                check_init_feasible = self.check_blocks_in_range()
                im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)

                check_feasible = check_goal_feasible and check_init_feasible

        self.goals = goals
        self.pre_selected_objects = self.env.selected_objects
        self.step_count = 0
        return im_state

    def generate_scene(self, scene):
        threshold = 0.1
        range_x = self.block_spawn_range_x
        range_y = self.block_spawn_range_y
        # randon scene
        goals = []
        inits = []
        if scene==-1:
            check_feasible = False
            while not check_feasible:
                nb = self.num_blocks
                goal_x = np.random.uniform(*range_x, size=self.num_blocks)
                goal_y = np.random.uniform(*range_y, size=self.num_blocks)
                goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
                init_x = np.random.uniform(*range_x, size=self.num_blocks)
                init_y = np.random.uniform(*range_y, size=self.num_blocks)
                inits = np.concatenate([init_x, init_y]).reshape(2, -1).T
                dist_g = np.linalg.norm(goals.reshape(nb, 1, 2) - goals.reshape(1, nb, 2), axis=2) + 1
                dist_i = np.linalg.norm(inits.reshape(nb, 1, 2) - inits.reshape(1, nb, 2), axis=2) + 1
                if not((dist_g < threshold).any() or (dist_i < threshold).any()):
                    check_feasible = True

        # random grid #
        elif scene==0:
            num_grid = 4
            #x = np.linspace(range_x[0], range_x[1], num_grid)
            #y = np.linspace(range_y[0], range_y[1], num_grid)
            offset_x = (range_x[1] - range_x[0]) / (3*num_grid) # 2*num_grid
            offset_y = (range_y[1] - range_y[0]) / (3*num_grid) # 2*num_grid
            x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        # no blocking #
        elif scene==1:
            range_x = self.block_spawn_range_x
            range_y = self.block_spawn_range_y
            num_grid = 4
            x = np.linspace(range_x[0], range_x[1], num_grid)
            y = np.linspace(range_y[0], range_y[1], num_grid)
            #offset_x = (range_x[1] - range_x[0]) / (2*num_grid)
            #offset_y = (range_y[1] - range_y[0]) / (2*num_grid)
            #x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            #y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                if num_collisions==0:
                    check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        # crossover #
        elif scene==2:
            range_x = self.block_spawn_range_x
            range_y = self.block_spawn_range_y
            num_grid = 4
            x = np.linspace(range_x[0], range_x[1], num_grid)
            y = np.linspace(range_y[0], range_y[1], num_grid)
            #offset_x = (range_x[1] - range_x[0]) / (2*num_grid)
            #offset_y = (range_y[1] - range_y[0]) / (2*num_grid)
            #x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            #y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                if num_collisions==1:
                    check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        return np.array(goals), np.array(inits)

    def generate_goal(self, info):
        goal_flags = info['goal_flags']
        if self.goal_type == 'circle':
            goal_image = deepcopy(self.background_img)
            for i, goal in enumerate(self.goals):
                if goal_flags[i]:
                    continue
                gx, gy = goal
                cv2.circle(goal_image, self.pos2pixel(gx, gy), 1, self.colors[i], -1)
            if self.env.data_format == 'NCHW':
                goal_image = np.transpose(goal_image, [2, 0, 1])

        elif self.goal_type=='pixel':
            goal_ims = []
            for i, goal in enumerate(self.goals):
                gx, gy = goal
                zero_array = np.zeros([self.env.camera_height, self.env.camera_width])
                if not goal_flags[i]:
                    cv2.circle(zero_array, self.pos2pixel(gx, gy), 1, 1, -1)
                goal_ims.append(zero_array)
            goal_image = np.concatenate(goal_ims)
            goal_image = goal_image.reshape([self.num_blocks, self.env.camera_height, self.env.camera_width])
            if self.env.data_format=='NHWC':
                goal_image = np.transpose(goal_image, [1, 2, 0])

        elif self.goal_type=='block':
            goal_image = self.goal_image

        return goal_image

    def reset(self, sidx=-1, scenario=-1):
        # glfw.destroy_window(self.env.viewer.window)
        # self.env.viewer = None
        if self.env.real_object:
            self.env.select_objects(self.num_blocks, sidx)
        im_state = self.init_env(scenario)

        poses, rotations = self.get_poses()
        goals = np.array(self.goals)

        info = {}
        info['num_blocks'] = self.num_blocks
        info['target'] = -1
        info['goals'] = np.array(self.goals)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        if self.num_blocks>0:
            info['dist'] = np.linalg.norm(info['goals']-info['poses'], axis=1)
            info['goal_flags'] = np.linalg.norm(info['goals']-info['poses'], axis=1) < self.threshold
        info['out_of_range'] = not self.check_blocks_in_range()
        pixel_poses = []
        for p in poses:
            _y, _x = self.pos2pixel(*p)
            pixel_poses.append([_x, _y])
        info['pixel_poses'] = np.array(pixel_poses)
        pixel_goals = []
        for g in self.goals:
            _y, _x = self.pos2pixel(*g)
            pixel_goals.append([_x, _y])
        self.pixel_goals = np.array(pixel_goals)
        info['pixel_goals'] = self.pixel_goals

        if self.task==0:
            if self.env.camera_depth:
                return im_state, info
            else:
                return [im_state], info

        elif self.task==1:
            return [im_state, self.goal_image], info

        elif self.task==2:
            poses, rotations = self.get_poses()
            poses = np.concatenate(poses)
            goals = np.concatenate(self.goals)
            # rotations = np.concatenate(rotations)
            state = np.concatenate([poses, goals])
            return state, info

        elif self.task==3:
            poses, rotations = self.get_poses()
            poses = np.concatenate(poses)
            goals = np.concatenate(self.goals)
            rotations = np.concatenate(rotations)
            state = np.concatenate([poses, goals, rotations])
            return state, info

    def step(self, action, target=-1):
        pre_poses, _ = self.get_poses()

        px, py, theta_idx = action
        if theta_idx >= self.num_bins:
            print("Error! theta_idx cannot be bigger than number of angle bins.")
            exit()
        theta = theta_idx * (2*np.pi / self.num_bins)
        rgb, collision, contact, depth = self.push_from_pixel(px, py, theta)

        poses, rotations = self.get_poses()

        info = {}
        info['num_blocks'] = self.num_blocks
        info['target'] = target
        info['goals'] = np.array(self.goals)
        info['contact'] = contact
        info['collision'] = collision
        info['pre_poses'] = np.array(pre_poses)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        info['dist'] = np.linalg.norm(info['goals']-info['poses'], axis=1)
        info['goal_flags'] = np.linalg.norm(info['goals']-info['poses'], axis=1) < self.threshold
        info['out_of_range'] = not self.check_blocks_in_range()
        pixel_poses = []
        for p in poses:
            _y, _x = self.pos2pixel(*p)
            pixel_poses.append([_x, _y])
        info['pixel_poses'] = np.array(pixel_poses)
        info['pixel_goals'] = self.pixel_goals

        reward, done, block_success = self.get_reward(info)
        info['success'] = np.all(block_success)
        info['block_success'] = block_success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        #if self.separate and type(reward) is float:
        #    reward = [reward] * self.num_blocks

        if self.task == 0:
            if self.env.camera_depth:
                return [rgb, depth], reward, done, info
            else:
                return [rgb], reward, done, info
        elif self.task == 1:
            state = [rgb, depth]
            if self.hide_goal:
                goal_image = self.generate_goal(info)
            else:
                goal_image = self.goal_image
            return [state, goal_image], reward, done, info
        elif self.task == 2:
            poses = info['poses'].flatten()
            goals = info['goals'].flatten()
            # rotations = info['rotations'].flatten()
            state = np.concatenate([poses, goals])
            return state, reward, done, info
        elif self.task == 3:
            poses = info['poses'].flatten()
            goals = info['goals'].flatten()
            rotations = info['rotations'].flatten()
            state = np.concatenate([poses, goals, rotations])
            return state, reward, done, info

    def get_poses(self):
        poses = []
        rotations = []
        for obj_idx in self.env.selected_objects:
            pos = deepcopy(self.env.sim.data.get_body_xpos(self.env.object_names[obj_idx])[:2])
            poses.append(pos)
            quat = deepcopy(self.env.sim.data.get_body_xquat(self.env.object_names[obj_idx]))
            rotation_mat = quat2mat(np.concatenate([quat[1:],quat[:1]]))
            rotations.append(rotation_mat[0][:2])
        return poses, rotations

    def clip_pos(self, pose):
        x, y = pose
        range_x = self.eef_range_x
        range_y = self.eef_range_y
        x = np.max((x, range_x[0]))
        x = np.min((x, range_x[1]))
        y = np.max((y, range_y[0]))
        y = np.min((y, range_y[1]))
        return x, y

    def check_blocks_in_range(self):
        poses, _ = self.get_poses()
        if len(poses)==0:
            return True
        x_max, y_max = np.concatenate(poses).reshape(-1, 2).max(0)
        x_min, y_min = np.concatenate(poses).reshape(-1, 2).min(0)
        if x_max > self.block_range_x[1] or x_min < self.block_range_x[0]:
            return False
        if y_max > self.block_range_y[1] or y_min < self.block_range_y[0]:
            return False
        return True

    def push_from_pixel_delta(self, px, py, dx, dy):
        pos_before = np.array(self.pixel2pos(px, py))
        pos_before[:2] = self.clip_pos(pos_before[:2])
        pos_after = pos_before + np.array([dx, dy, 0.])
        pos_after[:2] = self.clip_pos(pos_after[:2])

        theta = np.arctan2(dx, dy)
        x, y, z, w = euler2quat([np.pi, 0, -theta]) #+np.pi/2])
        quat = [w, x, y, z]
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_collision_check], quat, grasp=1.0)
        force = self.env.sim.data.sensordata
        if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
            #print("Collision!")
            self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
            if self.env.camera_depth:
                rgb, depth = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            else:
                rgb = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
                depth = None
            return rgb, True, np.zeros(self.num_blocks), depth
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], quat, grasp=1.0)
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_push], quat, grasp=1.0)
        contacts = self.check_block_contact()
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_prepush], quat, grasp=1.0)
        if self.env.camera_depth:
            rgb, depth = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        else:
            rgb = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            depth = None
        return rgb, False, contacts, depth

    def push_from_pixel(self, px, py, theta):
        pos_before = np.array(self.pixel2pos(px, py))
        pos_before[:2] = self.clip_pos(pos_before[:2])
        pos_after = pos_before + self.mov_dist * np.array([np.sin(theta), np.cos(theta), 0.])
        pos_after[:2] = self.clip_pos(pos_after[:2])

        x, y, z, w = euler2quat([np.pi, 0, -theta]) #+np.pi/2])
        quat = [w, x, y, z]
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_collision_check], quat, grasp=1.0)
        force = self.env.sim.data.sensordata
        if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
            #print("Collision!")
            self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
            if self.env.camera_depth:
                rgb, depth = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            else:
                rgb = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
                depth = None
            return rgb, True, np.zeros(self.num_blocks), depth
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], quat, grasp=1.0)
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_push], quat, grasp=1.0)
        contacts = self.check_block_contact()
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_prepush], quat, grasp=1.0)
        if self.env.camera_depth:
            rgb, depth = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        else:
            rgb = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            depth = None
        return rgb, False, contacts, depth

    def check_block_contact(self):
        collisions = np.zeros(self.num_blocks)
        for i in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[i]
            geom1 = self.env.sim.model.geom_id2name(contact.geom1)
            geom2 = self.env.sim.model.geom_id2name(contact.geom2)
            if geom1 is None or geom2 is None: continue
            if 'target_' in geom1 and 'target_' in geom2:
                if max(int(geom1[-1]), int(geom2[-1])) > self.num_blocks:
                    continue
                collisions[int(geom1[-1])-1] = 1
                collisions[int(geom2[-1])-1] = 1
        return collisions

    def pixel2pos(self, v, u): # u, v
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = (cz - z0) / (np.sin(theta) + np.cos(theta) * f / (v - v0 + 1e-10))
        x_cam = (u - u0) / (v - v0 + 1e-10) * y_cam
        x = - x_cam
        y = np.tan(theta) * (z0 - cz) + cy + 1 / np.cos(theta) * y_cam
        z = z0
        # print("cam pos:", [x_cam, y_cam])
        # print("world pos:", [x, y])
        # print()
        return x, y, z

    def pos2pixel(self, x, y):
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = np.cos(theta) * (y - cy - np.tan(theta) * (z0 - cz))
        dv = f * np.cos(theta) / ((cz - z0) / y_cam - np.sin(theta))
        v = dv + v0
        u = - dv * x / y_cam + u0
        return int(np.round(u)), int(np.round(v))

    def move2pixel(self, u, v):
        target_pos = np.array(self.pixel2pos(v, u))
        target_pos[2] = 1.05
        frame = self.env.move_to_pos(target_pos, get_img=True)
        plt.imshow(frame.transpose([1,2,0]))
        plt.show()

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


if __name__=='__main__':
    visualize = True
    from realobjects_env import *
    #from ur5_env import *
    env = UR5Env(render=True, camera_height=96, camera_width=96, control_freq=5, data_format='NCHW')
    '''
    ## saving background image ##
    im = env.move_to_pos([0.0, -0.23, 1.4], grasp=1.0)
    from PIL import Image
    backim = Image.fromarray((255*im.transpose([1,2,0])).astype(np.uint8))
    backim.save('background.png')
    '''
    env = pushpixel_env(env, num_blocks=3, mov_dist=0.05, max_steps=100, task=1, reward_type='binary', goal_type='circle')

    # eef_range_x = [-0.3, 0.3]
    # eef_range_y = [-0.2, 0.4]
    print(env.pos2pixel(env.eef_range_x[0], env.eef_range_y[0]))
    print(env.pos2pixel(env.eef_range_x[0], env.eef_range_y[1]))
    print(env.pos2pixel(env.eef_range_x[1], env.eef_range_y[1]))
    print(env.pos2pixel(env.eef_range_x[1], env.eef_range_y[0]))
    print(env.pos2pixel(env.block_range_x[0], env.block_range_y[0]))
    print(env.pos2pixel(env.block_range_x[0], env.block_range_y[1]))
    print(env.pos2pixel(env.block_range_x[1], env.block_range_y[1]))
    print(env.pos2pixel(env.block_range_x[1], env.block_range_y[0]))

    state = env.reset()
    if visualize:
        fig = plt.figure()
        if env.task == 1:
            ax0 = fig.add_subplot(121)
            ax1 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(111)

        s0 = deepcopy(state[0]).transpose([1, 2, 0])
        if env.task == 1:
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :env.num_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            action = input("Put action x, y, theta: ")
            action = [int(a) for a in action.split()]
            # action = [np.random.randint(10, 64), np.random.randint(10, 64), np.random.randint(8)]
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action[2] > 9:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(action)
        if visualize:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.task == 1:
                if env.goal_type == 'pixel':
                    s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                    s1[:, :, :env.num_blocks] = state[1].transpose([1, 2, 0])
                else:
                    s1 = deepcopy(state[1]).transpose([1, 2, 0])
                im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            im = ax1.imshow(s0)
            fig.canvas.draw()

        print('Reward: {}. Done: {}'.format(reward, done))
        if done:
            print('Done. New episode starts.')
            states = env.reset()
            if visualize:
                s0 = deepcopy(state[0]).transpose([1, 2, 0])
                if env.task == 1:
                    if env.goal_type == 'pixel':
                        s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                        s1[:, :, :env.num_blocks] = state[1].transpose([1, 2, 0])
                    else:
                        s1 = deepcopy(state[1]).transpose([1, 2, 0])
                    im0 = ax0.imshow(s1)
                s0[action[0], action[1]] = [1, 0, 0]
                im = ax1.imshow(s0)
                fig.canvas.draw()
