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


def save_video(frames, filename='video/mujoco.mp4', fps=60):
    writer = imageio.get_writer(filename, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
"""
def show_video(filname='video/mujoco.mp4'):
    mp4list = glob.glob(filname)
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay 
            loop controls style="height: 400px;">
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
"""
        
def show_image(img):
    #cv2.imshow("test", img)
    plt.figure(figsize = (16,9))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


class UR5Env():
    def __init__(
            self, 
            render=True,
            image_state=True,
            camera_height=64,
            camera_width=64,
            control_freq=8,
            data_format='NHWC',
            camera_depth=False,
            camera_name='rlview',
            xml_ver=0,
            color=False,
            gpu=-1,
            testset=False
            ):
        if xml_ver==0:
            self.model_xml = 'make_urdf/ur5_robotiq_push.xml'
            #self.model_xml = 'make_urdf/ur5_robotiq_cube.xml'
        elif xml_ver==1:
            if color:
                self.model_xml = 'make_urdf/ur5_robotiq_cube_v2_color.xml'
            else:
                self.model_xml = 'make_urdf/ur5_robotiq_cube_v2.xml'
        elif xml_ver==2:
            if color:
                self.model_xml = 'make_urdf/ur5_robotiq_cube_v3_color.xml'
            else:
                self.model_xml = 'make_urdf/ur5_robotiq_cube_v3.xml'
        elif xml_ver=='test':
            self.model_xml = 'make_urdf/ur5_robotiq_cube_test.xml'

        self.real_object = False
        self.render = render
        self.image_state = image_state
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq
        self.data_format = data_format
        self.camera_depth = camera_depth
        self.camera_name = camera_name
        self.gpu = gpu

        self.xml_ver = xml_ver
        self.color = color

        self.object_names = ['target_body_%d'%d for d in range(15)]
        #self.object_names = ['target_body_%d'%(d+1) for d in range(6)]
        self.num_objects = len(self.object_names)
        self.selected_objects = list(range(self.num_objects))

        self.model = load_model_from_path(os.path.join(file_path, self.model_xml))
        # self.model = load_model_from_path(os.path.join(file_path, 'make_urdf/ur5_robotiq.xml'))
        self.n_substeps = 1  # 20
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        if self.render:
            self.viewer = MjViewer(self.sim)
            self.viewer._hide_overlay = True
            # Camera pose
            lookat_refer = [0., 0., 0.9]  # self.sim.data.get_body_xpos('target_body_1')
            self.viewer.cam.lookat[0] = lookat_refer[0]
            self.viewer.cam.lookat[1] = lookat_refer[1]
            self.viewer.cam.lookat[2] = lookat_refer[2]
            self.viewer.cam.azimuth = -90 #0 # -65 #-75 #-90 #-75
            self.viewer.cam.elevation = -60  # -30 #-60 #-15
            self.viewer.cam.distance = 2.0  # 1.5
        else:
            if self.gpu==-1:
                self.viewer = MjRenderContextOffscreen(self.sim)
            else:
                self.viewer = MjRenderContextOffscreen(self.sim, self.gpu)

        self._init_robot()
        self.sim.forward()


    def _init_robot(self):
        self.arm_joint_list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.gripper_joint_list = ['left_finger_joint', 'right_finger_joint']

        str = ''
        for joint_name in self.gripper_joint_list:
            str += "{} : {:.3f}, ".format(joint_name, self.sim.data.get_joint_qpos(joint_name))
        # print(str)

        self.init_arm_pos = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
        for joint_idx, joint_name in enumerate(self.arm_joint_list):
            self.sim.data.set_joint_qpos(joint_name, self.init_arm_pos[joint_idx])
            self.sim.data.set_joint_qvel(joint_name, 0.0)
        self.reset_mocap_welds()

        '''
        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, -0.1]) + self.sim.data.get_body_xpos('wrist_3_link')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        '''
        im_state = self.move_to_pos(get_img=True)
        return im_state
        '''
        for _ in range(10):
            self.sim.step()
            if self.render: self.sim.render(mode='window')
            else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')
        '''

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation. """
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        # self.sim.forward()
    
    def calculate_depth(self, depth):
        zNear = 0.01
        zFar = 50
        return zNear / (1 - depth * (1 - zNear / zFar))

    def move_to_pos(self, pos=[0.0, 0.0, 1.20], quat=[0, 1, 0, 0], grasp=0.0, get_img=False):
        control_timestep = 1. / self.control_freq
        cur_time = time.time()
        end_time = cur_time + control_timestep

        while cur_time < end_time:
            self.sim.data.mocap_pos[0] = np.array(pos)
            self.sim.data.mocap_quat[0] = np.array(quat)

            self.sim.step()
            cur_time += self.sim.model.opt.timestep
            if self.render: self.sim.render(mode='window')
            #else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')

        pre_grasp = float(bool(sum(self.sim.data.ctrl)))
        self.sim.data.ctrl[0] = grasp
        self.sim.data.ctrl[1] = grasp
        if grasp != pre_grasp:
            cur_time = time.time()
            end_time = cur_time + 2.0*control_timestep
            #for i in range(20):
            while cur_time < end_time:
                self.sim.step()
                cur_time += self.sim.model.opt.timestep
                if self.render: self.sim.render(mode='window')
                #else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')

        diff_pos = np.linalg.norm(np.array(pos) - self.sim.data.get_body_xpos('robot0:mocap'))
        diff_quat = np.linalg.norm(np.array(quat) - self.sim.data.get_body_xquat('robot0:mocap'))
        #if diff_pos + diff_quat > 1e-3:
        #    print('Failed to move to target position.')
        #    print('Target pose:', pos)

        if get_img:
            if self.render:
                self.viewer._set_mujoco_buffers()
                self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
                camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
                if self.camera_depth:
                    im_rgb, im_depth = camera_obs
                else:
                    im_rgb = camera_obs
                self.viewer._set_mujoco_buffers()

            else:
                self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')
                camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
                if self.camera_depth:
                    im_rgb, im_depth = camera_obs
                else:
                    im_rgb = camera_obs

            im_rgb = np.flip(im_rgb, axis=1) / 255.0
            if self.data_format=='NCHW':
                im_rgb = np.transpose(im_rgb, [2, 0, 1])

            if self.camera_depth:
                im_depth = self.calculate_depth(np.flip(im_depth, axis=1))
                return im_rgb, im_depth
            else:
                return im_rgb

    def move_to_pos_slow(self, pos, quat=[0, 1, 0, 0], grasp=0.0):
        control_timestep = 1. / self.control_freq
        cur_time = time.time()
        end_time = cur_time + control_timestep

        ctime = 0.0
        init_pos = deepcopy(self.sim.data.mocap_pos[0])
        pos = np.array(pos)
        while ctime < control_timestep:
            self.sim.data.mocap_pos[0] = (ctime * pos + (control_timestep - ctime) * init_pos) / control_timestep
            self.sim.data.mocap_quat[0] = np.array(quat)

            self.sim.step()
            ctime += self.sim.model.opt.timestep
            if self.render: self.sim.render(mode='window')
            else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')

        pre_grasp = float(bool(sum(self.sim.data.ctrl)))
        self.sim.data.ctrl[0] = grasp
        self.sim.data.ctrl[1] = grasp
        if grasp != pre_grasp:
            cur_time = time.time()
            end_time = cur_time + 2.0*control_timestep
            #for i in range(20):
            while cur_time < end_time:
                self.sim.step()
                cur_time += self.sim.model.opt.timestep
                if self.render: self.sim.render(mode='window')
                else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')

        diff_pos = np.linalg.norm(np.array(pos) - self.sim.data.get_body_xpos('robot0:mocap'))
        diff_quat = np.linalg.norm(np.array(quat) - self.sim.data.get_body_xquat('robot0:mocap'))
        #if diff_pos + diff_quat > 1e-3:
        #    print('Failed to move slowly to target position.')
        #    print('Target pose:', pos)

        if self.render:
            self.viewer._set_mujoco_buffers()
            self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
            camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
            if self.camera_depth:
                im_rgb, im_depth = camera_obs
            else:
                im_rgb = camera_obs
            self.viewer._set_mujoco_buffers()

        else:
            self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')
            camera_obs = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
            if self.camera_depth:
                im_rgb, im_depth = camera_obs
            else:
                im_rgb = camera_obs

        im_rgb = np.flip(im_rgb, axis=1) / 255.0
        if self.data_format=='NCHW':
            im_rgb = np.transpose(im_rgb, [2, 0, 1])

        if self.camera_depth:
            im_depth = self.calculate_depth(np.flip(im_depth, axis=1))
            return im_rgb, im_depth
        else:
            return im_rgb

    def move_pos_diff(self, posdiff, quat=[0, 1, 0, 0], grasp=0.0):
        cur_pos = deepcopy(self.sim.data.mocap_pos[0])# - self.sim.data.get_body_xpos('box_link')
        target_pos = cur_pos + np.array(posdiff)
        return self.move_to_pos(target_pos, quat, grasp)


if __name__=='__main__':
    env = UR5Env(xml_ver=0, camera_height=512, camera_width=512)
    env.move_to_pos()
    '''
    im = env.move_to_pos([0.0, -0.23, 1.4], grasp=1.0)
    import matplotlib
    matplotlib.image.imsave('background.png', im)
    '''
    # place objects #
    x = np.linspace(-0.3, 0.3, 5)
    y = np.linspace(0.4, -0.2, 5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    for obj_idx in range(15): #16
        env.sim.data.qpos[7 * obj_idx + 12: 7 * obj_idx + 15] = [xx[obj_idx], yy[obj_idx], 0.9]
        print(obj_idx, xx[obj_idx], yy[obj_idx])
    env.sim.forward()

    grasp = 0.0
    for i in range(100):
        dist = 0.05
        action = dist * np.random.random(3)
        frame = env.move_pos_diff(action, grasp=grasp)
        frame = env.move_pos_diff(-action, grasp=grasp)
        '''
        x = input('Ctrl+c to exit. next?')
        if x==' ':
            x = x[0]
            grasp = 1.0 - grasp
            env.move_pos_diff([0.0, 0.0, 0.0], grasp=grasp)
            continue

        dist = 0.05
        if x=='w':
            frame = env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
        elif x=='s':
            frame = env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
        elif x=='6':
            frame = env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
        elif x=='4':
            frame = env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
        elif x=='8':
            frame = env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
        elif x=='2':
            frame = env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)
        '''

        ## test ##
        # print(env.sim.data.ncon)
        # for i in range(env.sim.data.ncon):
        #     contact = env.sim.data.contact[i]
        #     geom1 = env.sim.model.geom_id2name(contact.geom1)
        #     geom2 = env.sim.model.geom_id2name(contact.geom2)
        #     print(i, geom1, geom2)
        # print(frame.shape)
        #plt.imshow(frame)
        #plt.show()
