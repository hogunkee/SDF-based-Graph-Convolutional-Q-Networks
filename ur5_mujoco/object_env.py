from pushpixel_env import *
from reward_functions import *
import cv2
import imageio
from transform_utils import euler2quat, quat2mat

class objectwise_env(pushpixel_env):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, threshold=0.10, \
            reward_type='binary', conti=False, detection=False, delta_action=False):
        self.threshold = threshold
        self.conti = conti                  # continuous theta
        self.detection = detection
        self.delta_action = delta_action    # pushing with dx, dy
        self.depth_bg = np.load(os.path.join(file_path, 'depth_bg_480.npy'))
        super().__init__(ur5_env, num_blocks, mov_dist, max_steps, 1, reward_type, 'block', False, False)

    def reset(self, sidx=-1, scenario=-1):
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

        if self.detection:
            return [im_state, self.goal_image], info
        else:
            state_goal = [poses, goals]
            return [state_goal, im_state], info

    def step(self, action, sdf=None):
        poses, _ = self.get_poses()

        if self.delta_action:
            rx, ry, dx, dy = action
            dx, dy = np.clip([dx, dy], -0.1, 0.1)
            push_center = np.array([rx, ry])
            py, px = self.pos2pixel(rx, ry)

            if sdf is not None:
                vec = np.sqrt(2) * np.array([-dy, dx]) / np.linalg.norm([-dy, dx])
                count_negative = 0
                px_before, py_before = px, py
                px_before2, py_before2 = px + vec[0], py + vec[1]
                while count_negative < 12:
                    px_before += vec[0]
                    py_before += vec[1]
                    px_before2 += vec[0]
                    py_before2 += vec[1]
                    if px_before < 0 or py_before < 0:
                        px_before -= vec[0]
                        py_before -= vec[1]
                        break
                    elif px_before >= self.env.camera_width or py_before >= self.env.camera_height:
                        px_before -= vec[0]
                        py_before -= vec[1]
                        break

                    pxb = np.round(px_before).astype(int)
                    pyb = np.round(py_before).astype(int)
                    pxb2 = np.round(px_before2).astype(int)
                    pyb2 = np.round(py_before2).astype(int)
                    if sdf[pxb, pyb] <= 0 and sdf[pxb2, pyb2] <= 0:
                        count_negative += 1

                    rx_before, ry_before = np.array(self.pixel2pos(px_before, py_before))[:2]
                    if rx_before < self.eef_range_x[0] or rx_before > self.eef_range_x[1]:
                        break
                    elif ry_before < self.eef_range_y[0] or ry_before > self.eef_range_y[1]:
                        break
                im_state, collision, contact, depth = self.push_from_pixel_delta(px_before, py_before, dx, dy)
            else:
                pos_before = push_center - self.mov_dist * np.array([dx, dy])/np.linalg.norm([dx, dy])
                py, px = self.pos2pixel(*pos_before)
                im_state, collision, contact, depth = self.push_from_pixel_delta(px, py, dx, dy)

        else:
            if self.detection:
                rx, ry, theta = action
                push_center = np.array([rx, ry])
                py, px = self.pos2pixel(rx, ry)
                #px, py, theta = action
                #push_center = np.array(self.pixel2pos(px, py))[:2]
            else:
                push_obj, theta = action
                if theta >= self.num_bins:
                    print("Error! theta_idx cannot be bigger than number of angle bins.")
                    exit()
                push_center = poses[push_obj]

            if not self.conti:
                theta = theta * (2*np.pi / self.num_bins)

            if self.detection and sdf is not None:
                vec = np.round(np.sqrt(2) * np.array([-np.cos(theta), np.sin(theta)])).astype(int)
                count_negative = 0
                px_before, py_before = px, py
                px_before2, py_before2 = px + vec[0], py + vec[1]
                while count_negative < 12:
                    px_before += vec[0]
                    py_before += vec[1]
                    px_before2 += vec[0]
                    py_before2 += vec[1]
                    if px_before <0 or py_before < 0:
                        px_before -= vec[0]
                        py_before -= vec[1]
                        break
                    elif px_before >= self.env.camera_width or py_before >= self.env.camera_height:
                        px_before -= vec[0]
                        py_before -= vec[1]
                        break
                    if sdf[px_before, py_before] <= 0 and sdf[px_before2, py_before2] <= 0:
                        count_negative += 1

                    rx_before, ry_before = np.array(self.pixel2pos(px_before, py_before))[:2]
                    if rx_before < self.eef_range_x[0] or rx_before > self.eef_range_x[1]:
                        break
                    elif ry_before < self.eef_range_y[0] or ry_before > self.eef_range_y[1]:
                        break
                if self.mov_dist is None:
                    im_state, collision, contact, depth = self.push_pixel2pixel(
                            [px_before, py_before], [px, py], theta)
                else:
                    im_state, collision, contact, depth = self.push_from_pixel(px_before, py_before, theta)
            else:
                pos_before = push_center - self.mov_dist * np.array([np.sin(theta), np.cos(theta)])
                py, px = self.pos2pixel(*pos_before)
                im_state, collision, contact, depth = self.push_from_pixel(px, py, theta)
        pre_poses = deepcopy(poses)
        poses, rotations = self.get_poses()

        info = {}
        info['num_blocks'] = self.num_blocks
        info['target'] = -1
        info['action'] = action
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
        #info['success'] = np.all(block_success)
        info['block_success'] = block_success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        if self.detection:
            if self.env.camera_depth:
                return [[im_state, depth], self.goal_image], reward, done, info
            else:
                return [im_state, self.goal_image], reward, done, info
        else:
            poses = info['poses']
            goals = info['goals']
            state_goal = [poses, goals]
            return [state_goal, im_state], reward, done, info

    def push_pixel2pixel(self, pixel_before, pixel_target, theta):
        bx, by = pixel_before
        tx, ty = pixel_target
        pos_before = np.array(self.pixel2pos(bx, by))
        pos_before[:2] = self.clip_pos(pos_before[:2])
        pos_after = np.array(self.pixel2pos(tx, ty))
        pos_after[:2] = self.clip_pos(pos_after[:2])

        x, y, z, w = euler2quat([np.pi, 0, -theta+np.pi/2])
        quat = [w, x, y, z]
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_collision_check], quat, grasp=1.0)
        force = self.env.sim.data.sensordata
        if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
            #print("Collision!")
            self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
            if self.env.camera_depth:
                im_state, depth_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            else:
                im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
                depth_state = None
            return im_state, True, np.zeros(self.num_blocks), depth_state
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], quat, grasp=1.0)
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_push], quat, grasp=1.0)
        contacts = self.check_block_contact()
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_prepush], quat, grasp=1.0)
        if self.env.camera_depth:
            im_state, depth_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        else:
            im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            depth_state = None
        return im_state, False, contacts, depth_state
    
    def get_center_from_sdf(self, sdf, depth):
        px, py = np.where(sdf==sdf.max())
        px = px[0]
        py = py[0]
        cx, cy, _ = self.pixel2pos(px, py)
        #dy = (self.depth_bg - depth)[sdf>0].max() * np.sin(self.cam_theta) / 2
        #cy += dy
        return cx, cy
        
