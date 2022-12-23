import copy
import cv2
import glfw
import imageio
import numpy as np
import os
import time
import types
import time
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from collections import OrderedDict
from base import MujocoXML

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen

file_path = os.path.dirname(os.path.abspath(__file__))


def new_joint(**kwargs):
    element = ET.Element("joint", attrib=kwargs)
    return element

def array_to_string(array):
    return " ".join(["{}".format(x) for x in array])

def string_to_array(string):
    return np.array([float(x) for x in string.split(" ")])


class UR5Robot(MujocoXML):
    def __init__(self):
        super().__init__(os.path.join(file_path, 'make_urdf/meshes_mujoco/ur5_robotiq.xml'))
        self.set_sensor()

    def set_sensor(self):
        sensor = ET.SubElement(self.root, 'sensor')
        f1 = ET.SubElement(sensor, 'force')
        f2 = ET.SubElement(sensor, 'force')
        f1.set('name', 'left_finger_force')
        f1.set('site', 'left_inner_finger_sensor')
        f2.set('name', 'right_finger_force')
        f2.set('site', 'right_inner_finger_sensor')


class MujocoXMLObject(MujocoXML):
    def __init__(self, fname):
        MujocoXML.__init__(self, fname)

    def get_bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='bottom_site']")
        return string_to_array(bottom_site.get("pos"))

    def get_top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='top_site']")
        return string_to_array(top_site.get("pos"))

    def get_horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='horizontal_radius_site']"
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_collision(self, name=None):

        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(name, i))
        return collision

    def get_visual(self, name=None, site=False):
        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        if name is not None:
            visual.attrib["name"] = name
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            visual.append(ET.Element("site", attrib=template))
        return visual


class PushTask(UR5Robot):
    def __init__(self, mujoco_objects):
        """
        mujoco_objects: a list of MJCF models of physical objects
        """
        super().__init__()

        # temp: z-rotation
        self.z_rotation = True

        self.merge_objects(mujoco_objects)
        self.set_objects_geom(mass=0.02)
        self.save = False

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name)
            obj.append(new_joint(name=obj_name, type="free", damping="0.0005"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def set_objects_geom(self, mass=0.02):
        for o in self.objects:
            o.find('geom').set('mass', f'{mass}')
            o.find('geom').set('friction', "0.1 0.1 0.5")
            o.find('geom').set('solimp', "0.9 0.95 0.001")
            o.find('geom').set('solref', "0.001 1.0")

    def sample_quat(self):
        """Samples quaternions of random rotations along the z-axis."""
        if self.z_rotation:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        return [1, 0, 0, 0]

    def random_quat(self, rand=None):
        """Return uniform random unit quaternion.
        rand: array like or None
            Three independent random variables that are uniformly distributed
            between 0 and 1.
        >>> q = random_quat()
        >>> np.allclose(1.0, vector_norm(q))
        True
        >>> q = random_quat(np.random.random(3))
        >>> q.shape
        (4,)
        """
        if rand is None:
            rand = np.random.rand(3)
        else:
            assert len(rand) == 3
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = np.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        return np.array(
            (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
            dtype=np.float32,
        )    

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        placed_objects = []
        index = 0
        # place objects by rejection sampling
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            #print('horizontal_radius', horizontal_radius)
            #print('bottom_offset', bottom_offset)
            success = False
            for _ in range(5000):  # 5000 retries
                object_z = np.random.uniform(high=0.2, low=0.2)
                #bin_x_half = self.bin_size[0] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
                #bin_y_half = self.bin_size[1] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
                object_x = np.random.uniform(high=0.2, low=-0.2)
                object_y = np.random.uniform(high=0.3, low=-0.1)

                # make sure objects do not overlap
                pos = np.array([object_x, object_y, object_z])
                location_valid = True
                for pos2, r in placed_objects:
                    dist = np.linalg.norm(pos[:2] - pos2[:2], np.inf)
                    if dist <= 0.02: #r + horizontal_radius:
                        location_valid = False
                        break

                # place the object
                if location_valid:
                    # add object to the position
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set("pos", array_to_string(pos))
                    # random z-rotation
                    #quat = self.sample_quat()
                    quat = self.random_quat()
                    self.objects[index].set("quat", array_to_string(quat))
                    success = True
                    break

            # raise error if all objects cannot be placed after maximum retries
            if not success:
                raise Exception #RandomizationError("Cannot place all objects in the bins")
            index += 1

    def place_single_objects(self, index):
        placed_objects = []
        obj_list = []
        for _, obj_mjcf in self.mujoco_objects.items():
            obj_list.append(obj_mjcf)
        obj_mjcf = obj_list[index]
        horizontal_radius = obj_mjcf.get_horizontal_radius()
        bottom_offset = obj_mjcf.get_bottom_offset()
        object_z = np.random.uniform(high=self.bin_size[2], low=0.02)
        bin_x_half = self.bin_size[0] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
        bin_y_half = self.bin_size[1] / 2.0 - horizontal_radius - (self.bin_size[2] - object_z) - 0.02
        object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
        object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)

        # make sure objects do not overlap
        object_xyz = np.array([object_x, object_y, object_z])
        pos = self.bin_offset - bottom_offset + object_xyz

        self.objects[index].set("pos", array_to_string(pos))
        quat = self.random_quat()
        self.objects[index].set("quat", array_to_string(quat))
        index += 1

    def place_col_objects(self):
        placed_objects = []
        index = 0

        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            object_z = np.random.uniform(high=self.bin_size[2] + 0.5, low=self.bin_size[2])
            bin_x_half = self.bin_size[0] / 2 - horizontal_radius - 0.05
            bin_y_half = self.bin_size[1] / 2 - horizontal_radius - 0.05
            object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
            object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)
            object_xyz = np.array([object_x, object_y, object_z])
            pos = self.bin_offset - bottom_offset + object_xyz
            placed_objects.append((pos, horizontal_radius))
            self.objects[index].set("pos", array_to_string(pos))
            quat = self.sample_quat()
            self.objects[index].set("quat", array_to_string(quat))
            index += 1


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
            color=False,
            gpu=-1,
            dataset='train1',
            small=False
            ):

        self.real_object = True
        self.dataset = dataset
        self.small = small
        self.render = render
        self.image_state = image_state
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq
        self.data_format = data_format
        self.camera_depth = camera_depth
        self.camera_name = camera_name
        self.gpu = gpu

        self.color = color

        mujoco_objects = self.load_objects(num=0)
        self.object_names = list(mujoco_objects.keys())
        self.num_objects = len(self.object_names)
        self.selected_objects = list(range(self.num_objects))

        self.model = PushTask(mujoco_objects)
        self.model.place_objects()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        # self.model = load_model_from_path(os.path.join(file_path, 'make_urdf/ur5_robotiq.xml'))
        self.n_substeps = 1  # 20

        self.set_sim()
        self._init_robot()
        self.sim.forward()
        self.obj_orientation = self.predefine_orientation()

    def destroy_viewer(self):
        glfw.destroy_window(self.viewer.window)
        self.viewer = None

    def reset_viewer(self):
        if self.viewer is not None:
            if self.render:
                glfw.destroy_window(self.viewer.window)
            else:
                glfw.destroy_window(self.viewer.opengl_context.window)
            self.viewer = None
        self.set_sim()
        self._init_robot()
        self.sim.forward()

    def set_sim(self):
        self.sim = MjSim(self.mjpy_model, nsubsteps=self.n_substeps)
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

    def predefine_orientation(self):
        defined_orient = {}
        # trainset #
        defined_orient['train1-4'] = [1/2, 0]
        defined_orient['train1-6'] = [0, 1/2]
        defined_orient['train1-7'] = [0, 1/2]
        defined_orient['train1-9'] = [0, 1/2]
        defined_orient['train1-11'] = [1/2, 0]
        defined_orient['train1-14'] = [0, 1]

        defined_orient['small-train1-4'] = [1/2, 0]
        defined_orient['small-train1-6'] = [0, 1/2]
        defined_orient['small-train1-7'] = [0, 1/2]
        defined_orient['small-train1-9'] = [0, 1/2]
        defined_orient['small-train1-11'] = [1/2, 0]
        defined_orient['small-train1-14'] = [0, 1]

        #defined_orient['milk'] = [1/2, 0]
        #defined_orient['GreenCup'] = [1/2, 7/4]
        defined_orient['train1-12'] = [0, 1/2]
        defined_orient['train1-13'] = [0, 1/2]
        defined_orient['train1-15'] = [1/2, 0]
        defined_orient['small-train1-12'] = [0, 1/2]
        defined_orient['small-train1-13'] = [0, 1/2]
        defined_orient['small-train1-15'] = [1/2, 0]
        # testset #
        defined_orient['test0'] = [1/2, 0]
        defined_orient['test1'] = [1/2, 0]
        defined_orient['test2'] = [1, 0]
        defined_orient['test3'] = [3/2, 0]
        defined_orient['test4'] = [0, 1]
        defined_orient['test5'] = [3/2, 0]
        defined_orient['test8'] = [0, 0]
        defined_orient['test9'] = [0, -1/2]
        defined_orient['small-test0'] = [1/2, 0]
        defined_orient['small-test1'] = [1/2, 0]
        defined_orient['small-test2'] = [1, 0]
        defined_orient['small-test3'] = [3/2, 0]
        defined_orient['small-test4'] = [0, 1]
        defined_orient['small-test5'] = [3/2, 0]
        defined_orient['small-test8'] = [0, 0]
        defined_orient['small-test9'] = [0, -1/2]

        # shapenet sem #
        defined_orient['train2-1'] = [1/2, 1/2]
        defined_orient['train2-4'] = [-1/2, 0]
        defined_orient['train2-6'] = [1/2, 0]
        defined_orient['train2-7'] = [1/2, 0]
        defined_orient['train2-9'] = [1/2, 0]
        defined_orient['train2-11'] = [1/2, 0]
        defined_orient['train2-12'] = [1/2, 0]
        defined_orient['small-train2-1'] = [1/2, 1/2]
        defined_orient['small-train2-4'] = [-1/2, 0]
        defined_orient['small-train2-6'] = [1/2, 0]
        defined_orient['small-train2-7'] = [1/2, 0]
        defined_orient['small-train2-9'] = [1/2, 0]
        defined_orient['small-train2-11'] = [1/2, 0]
        defined_orient['small-train2-12'] = [1/2, 0]
        #defined_orient['shapenetsem%d' %6] = [-1/2, 0]

        orient = {}
        for obj_name in defined_orient:
            if obj_name in self.obj_list:
                orient[self.obj_list.index(obj_name)] = defined_orient[obj_name]
        return orient

    def select_objects(self, num=3, idx=-1):
        if idx==-1:
            indices = np.random.choice(range(self.num_objects), num, False)
        else:
            indices = np.arange(num*idx, num*(idx+1)) % 32
        self.selected_objects = list(indices)
        #obj_names_selected = [self.object_names[idx] for idx in self.selected_objects]

    def load_objects(self, num=0):
        if self.dataset=="test":
            obj_list = []
            for o in range(12):
                if self.small:
                    obj_list.append('small-test%d'%o)
                else:
                    obj_list.append('test%d'%o)

        elif self.dataset=="train1":
            obj_list = []
            for o in range(16):
                if self.small:
                    obj_list.append('small-train1-%d'%o)
                else:
                    obj_list.append('train1-%d'%o)

        elif self.dataset=="train2":
            obj_list = []
            for o in range(15):
                if self.small:
                    obj_list.append('small-train2-%d'%o)
                else:
                    obj_list.append('train2-%d'%o)

        self.obj_list = obj_list
        obj_dirpath = 'make_urdf/objects/'
        obj_counts = [0] * len(obj_list)
        lst = []

        if num==0:
            num = len(obj_list)
        for n in range(num):
            #print("spawning %d"%n, obj_list[n])
            rand_obj = n
            #rand_obj = np.random.randint(len(obj_list))
            obj_name = obj_list[rand_obj]
            obj_count = obj_counts[rand_obj]
            obj_xml = MujocoXMLObject(os.path.join(file_path, obj_dirpath, '%s.xml'%obj_name))
            lst.append(("%s_%d"%(obj_name, obj_count), obj_xml))
            obj_counts[rand_obj] += 1
        mujoco_objects = OrderedDict(lst)
        return mujoco_objects

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

        for _ in range(10):
            self.sim.step()
            if self.render: self.sim.render(mode='window')
            #else: self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')

        im_state = self.move_to_pos(get_img=True)
        return im_state

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
                #self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, mode='offscreen')
                if self.camera_depth:
                    im_depth = None
                    while im_depth is None:
                        im_rgb, im_depth = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')
                else:
                    im_rgb = self.sim.render(camera_name=self.camera_name, width=self.camera_width, height=self.camera_height, depth=self.camera_depth, mode='offscreen')

            im_rgb = np.flip(im_rgb, axis=1) / 255.0
            if self.data_format=='NCHW':
                im_rgb = np.transpose(im_rgb, [2, 0, 1])

            if self.camera_depth:
                im_depth = self.calculate_depth(np.flip(im_depth, axis=1))
                return im_rgb, im_depth
            else:
                return im_rgb

    def move_to_pos_slow(self, pos, quat=[0, 1, 0, 0], grasp=0.0):
        control_timestep = 1.5 / self.control_freq # 1.
        cur_time = time.time()
        end_time = cur_time + control_timestep

        ctime = 0.0
        init_pos = copy.deepcopy(self.sim.data.mocap_pos[0])
        pos = np.array(pos)
        while ctime < control_timestep:
            self.sim.data.mocap_pos[0] = (ctime * pos + (control_timestep - ctime) * init_pos) / control_timestep
            self.sim.data.mocap_quat[0] = np.array(quat)

            self.sim.step()
            ctime += self.sim.model.opt.timestep
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
        #if diff_pos + diff_quat > 1e-2:
        #    print('Failed to move slowly to target position.')

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
        cur_pos = copy.deepcopy(self.sim.data.mocap_pos[0])# - self.sim.data.get_body_xpos('box_link')
        target_pos = cur_pos + np.array(posdiff)
        return self.move_to_pos(target_pos, quat, grasp)


if __name__=='__main__':
    env = UR5Env(camera_height=512, camera_width=512, dataset="test", small=True)
    env.move_to_pos()
    '''
    im = env.move_to_pos([0.0, -0.23, 1.4], grasp=1.0)
    import matplotlib
    matplotlib.image.imsave('background.png', im)
    '''
    # place objects #
    x = np.linspace(-0.4, 0.4, 7)
    y = np.linspace(0.4, -0.2, 5)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    from transform_utils import euler2quat

    print(env.object_names)
    for obj_idx in range(len(env.obj_list)): #16
        env.sim.data.qpos[7 * obj_idx + 12: 7 * obj_idx + 15] = [xx[obj_idx], yy[obj_idx], 0.92]
        euler = np.zeros(3)
        if obj_idx in env.obj_orientation:
            euler[:2] = np.pi * np.array(env.obj_orientation[obj_idx])
        x, y, z, w = euler2quat(euler)
        env.sim.data.qpos[7 * obj_idx + 15: 7 * obj_idx + 19] = [w, x, y, z]
        env.sim.forward()
        print(obj_idx, xx[obj_idx], yy[obj_idx])
        print(euler / np.pi)
        print()

    env.move_to_pos()
    for obj_idx in range(len(env.obj_list)): #16
        env.sim.data.qpos[7 * obj_idx + 12: 7 * obj_idx + 15] = [xx[obj_idx], yy[obj_idx], 0.92]
        euler = np.zeros(3)
        if obj_idx in env.obj_orientation:
            euler[:2] = np.pi * np.array(env.obj_orientation[obj_idx])
        x, y, z, w = euler2quat(euler)
        env.sim.data.qpos[7 * obj_idx + 15: 7 * obj_idx + 19] = [w, x, y, z]
        env.sim.forward()
        print(obj_idx, xx[obj_idx], yy[obj_idx])
        print(euler / np.pi)
        print()

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
