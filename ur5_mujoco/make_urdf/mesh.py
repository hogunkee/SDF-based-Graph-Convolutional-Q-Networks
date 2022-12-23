import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import pyrender

class RescalingType:
    FIT_MIN_DIM = 'min'
    FIT_MED_DIM = 'med'
    FIT_MAX_DIM = 'max'
    FIT_DIAG = 'diag'
    RELATIVE = 'relative'

# Load and read the obj file and save mesh data(vertice, triangle, normal) to Mesh3D class
class Mesh3D(object):
    def __init__(self, filepath):
        """Initialize Mesh3D class
        Args:
            filepath: string, directory of mesh .obj file
        """
        self._filepath = filepath
        self._read()
    
    @property    
    def vertices(self):
        return self._vertices
    
    @property
    def triangles(self):
        return self._triangles
    
    @property
    def normals(self):
        return self._normals
    
    def _read(self):
        numVerts = 0
        verts = []
        norms = None
        faces = []
        f = open(self._filepath, 'r') # Read the obj file line by line

        for line in f:
            vals = line.split()
            if len(vals) > 0:
                if vals[0] == 'v':
                    v = list(map(float, vals[1:4]))
                    verts.append(v)
                if vals[0] == 'vn':
                    if norms is None:
                        norms = []
                    n = list(map(float, vals[1:4]))
                    norms.append(n)  
                if vals[0] == 'f':
                    vi = []
                    vti = []
                    nti = []
                    if vals[1].find('/') == -1:
                        vi = list(map(int, vals[1:]))
                        vi = [i - 1 for i in vi]
                    else:
                        for j in range(1, len(vals)):
                            val = vals[j]
                            tokens = val.split('/')
                            for i in range(len(tokens)):
                                if i == 0:
                                    vi.append(int(tokens[i]) - 1) # adjust for python 0 - indexing
                                elif i == 1:
                                    if tokens[i] != '':
                                        vti.append(int(tokens[i]))
                                elif i == 2:
                                    nti.append(int(tokens[i]))
                    faces.append(vi)
        
        f.close()
        
        if verts is not None:
            vertices = np.array(verts)
        self._vertices = vertices

        if faces is not None:
            triangles = np.array(faces)
        self._triangles = triangles

        if norms is not None:
            normals = np.array(norms)
            if normals.shape[0] == 3:
                normals = normals.T
        self._normals = norms
        
        self._bb_center = self._compute_bb_center() # center point of the cube wrapping the mesh model
        self._centroid = self._compute_centroid() # mean of all vertices
        self._trimesh = None
        self._init_trimesh() # save trimesh class in self._trimesh
    
    def _init_trimesh(self):
        if self._trimesh is None:
            self._trimesh = trimesh.Trimesh(vertices=self._vertices,
                                            faces=self._triangles,
                                            vertex_normals=self._normals)
    
    def clean_mesh(self, rescale_mesh=False, scale=1.0, rescaling_type='min'):
        """Remove unreferenced vertices and triangles and scale its mesh (optional)
        Args:
            rescale_mesh: bool, whether rescale mesh or not
            scale: float, a scaling factor
            rescaliing_type: string,
                min: change length of minimum dimension to scale
                med: change length of medium dimension to scale
                max: change length of maximum dimension to scale
                diag: change norm of mesh to scale
                relative: just scaling with scale
        """
        
        self._remove_bad_tris() # remove bad triangle data not included in vertices
        self._remove_unreferenced_vertices() # remove unreferenced vertices according to traingles
        self._center_vertices_bb() # move mean of mesh to the origin
        
        if rescale_mesh: # rescaling mesh model
            self._rescale_vertices(scale, rescaling_type)
            
        self._trimesh = None
        self._init_trimesh() # reset trimesh class
            
    def _remove_bad_tris(self):
        new_tris = []
        num_v = len(self._vertices)
        for t in self._triangles.tolist():
            if (t[0] >= 0 and t[0] < num_v and t[1] >= 0 and t[1] < num_v and t[2] >= 0 and t[2] < num_v and
                t[0] != t[1] and t[0] != t[2] and t[1] != t[2]):
                new_tris.append(t)
        self._triangles = np.array(new_tris)
        
    def _remove_unreferenced_vertices(self):
        vertex_array = self._vertices
        num_v = vertex_array.shape[0]

        reffed_array = np.zeros([num_v, 1])
        for f in self._triangles.tolist():
            if f[0] < num_v and f[1] < num_v and f[2] < num_v:
                reffed_array[f[0]] = 1
                reffed_array[f[1]] = 1
                reffed_array[f[2]] = 1

        reffed_v_old_ind = np.where(reffed_array == 1)
        reffed_v_old_ind = reffed_v_old_ind[0]
        reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1

        try:
            self._vertices = vertex_array[reffed_v_old_ind, :]
            if self._normals is not None:
                normals_array = np.array(self._normals)
        except IndexError:
            return False

        new_triangles = []
        for f in self._triangles:
            new_triangles.append([reffed_v_new_ind[f[0]], reffed_v_new_ind[f[1]], reffed_v_new_ind[f[2]]] )
        self._triangles = np.array(new_triangles)
        return True
    
    def _rescale_vertices(self, scale, rescaling_type=RescalingType.FIT_MIN_DIM):
        vertex_array = self._vertices
        min_vertex_coords = self._min_coords()
        max_vertex_coords = self._max_coords()
        vertex_extent = max_vertex_coords - min_vertex_coords

        if rescaling_type == RescalingType.FIT_MIN_DIM:
            dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.FIT_MED_DIM:
            dim = np.where(vertex_extent == np.median(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.FIT_MAX_DIM:
            dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == RescalingType.RELATIVE:
            relative_scale = 1.0
        elif rescaling_type == RescalingType.FIT_DIAG:
            diag = np.linalg.norm(vertex_extent)
            relative_scale = diag / 3.0 # make the gripper size exactly one third of the diagonal

        scale_factor = scale / relative_scale
        vertex_array = scale_factor * vertex_array
        self._vertices = vertex_array
        self._bb_center = self._compute_bb_center()
        self._centroid = self._compute_centroid()

    def rotate_mesh(self, theta_x, theta_y, theta_z):
        """Rotate mesh with z-axis, y-axis, and x-axis (order)
        Args:
            theta_x: float, rotation angle (radian) referenced to x-axis
            theta_y: float, rotation angle (radian) referenced to y-axis
            theta_z: float, rotation angle (radian) referenced to z-axis
        """
        
        self._center_vertices_bb()
        vertex_array_cent = self._vertices
        
        x_axis = [np.cos(theta_z), np.sin(theta_z), 0]
        y_axis = [-np.sin(theta_z), np.cos(theta_z), 0]
        z_axis = [0, 0, 1.0]
        R_pc_obj = np.c_[x_axis, y_axis, z_axis]
        
        x_axis = [np.cos(theta_y), 0, -np.sin(theta_y)]
        y_axis = [0, 1.0, 0]
        z_axis = [np.sin(theta_y), 0, np.cos(theta_y)]
        R_pc_obj = np.dot(np.c_[x_axis, y_axis, z_axis], R_pc_obj)
        
        x_axis = [1.0, 0, 0]
        y_axis = [0, np.cos(theta_x), np.sin(theta_x)]
        z_axis = [0, -np.sin(theta_x), np.cos(theta_x)]
        R_pc_obj = np.dot(np.c_[x_axis, y_axis, z_axis], R_pc_obj)
        
        vertex_array_rot = np.dot(R_pc_obj, vertex_array_cent.T)
        self._vertices = vertex_array_rot.T
        bb_center = (self._min_coords() + self._max_coords()) / 2.0
        bb_center[2] = self._min_coords()[2]
        self._vertices = self._vertices - bb_center
        
        if self._normals is not None:
            normals_array = np.array(self._normals)
            normals_array_rot = R_pc_obj.dot(normals_array.T)
        
        self._trimesh = None
        self._init_trimesh()

    def move_mesh(self, center_pose):
        #move origin of the mesh
        self._vertices = self._vertices - center_pose
        self._trimesh = None
        self._init_trimesh()
        
    def pose_to_image_pixel(self, pose, camera_pose, image_res, image_width, image_length):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        ratio = z / (camera_pose[2] * image_res)
        x_pixel = int(x * ratio)
        y_pixel = int(y * ratio)
        x_pixel = image_length / 2 + x_pixel
        y_pixel = image_width / 2 - y_pixel
        return x_pixel, y_pixel
        
    def generate_fov_color_and_depth_image(self, pose=np.array([0.0, 0.0, 0.68]), yfov=np.pi / 4.0, image_length=256):
        """Generate color and depth image with a pinhole camera model
        Args:
            pose: 3-D float vector, [x, y, z] coordinate of camear according to the origin
            yfov: float, the vertical field of view in radians
            image_length: int, desired length of rendedered image
        Returns:
            float, resolution of rendered image
            3-D float array, rendered rgb image
            3-D float array, rendered depth image
        """
        
        mesh = pyrender.Mesh.from_trimesh(self._trimesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        
        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=yfov)
        camera_pose = np.array([
               [1.0, 0.0, 0.0, pose[0]],
               [0.0, 1.0, 0.0, pose[1]],
               [0.0, 0.0, 1.0, pose[2]],
               [0.0, 0.0, 0.0, 1.0],
            ])
        scene.add(camera, pose=camera_pose)

        # Set up the light -- a single spot light in the same spot as the camera
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                       innerConeAngle=np.pi/16.0)
        scene.add(light, pose=camera_pose)

        # Render the scene
        focal_length = pose[2]
        sensor_width = np.tan(yfov / 2.0) * 2.0 * focal_length
        #image_length = int(sensor_width / res)
        image_width = image_length
        resolution = sensor_width / image_length
        r = pyrender.OffscreenRenderer(image_length, image_width)
        color_image, depth_image = r.render(scene)
       
        self._camera_pose = pose
        self._image_width = image_width
        self._image_length = image_length
        self._yfov = yfov
        self._sensor_width = sensor_width
        self._sensor_length = sensor_width
        self._image_res = resolution
        
        zero_index = np.where(depth_image == 0.0)
        depth_image[zero_index] = pose[2]
        return resolution, color_image, depth_image
    
    def generate_vertical_depth_image(self, pose=np.array([0.0, 0.0, 5.0]), image_length=400, image_width=400, res=0.05):
        """Generate color and depth image with a vertical camera model
        Args:
            pose: 3-D float vector, [x, y, z] coordinate of camear according to the origin
            image_length: int, desired length of rendedered image
            image_width: int, desired width of rendedered image
            res: float, desired resolution of rendered image
        Returns:
            3-D float array, rendered depth image
        """
        max_length, max_width, max_height = self._max_coords()
        min_length, min_width, min_height = self._min_coords()
        min_length -= pose[0]
        max_length -= pose[0]
        min_width -= pose[1]
        max_width -= pose[1]
        
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        def PointInTriangle(pt, v1, v2, v3):
            d1 = sign(pt, v1, v2)
            d2 = sign(pt, v2, v3)
            d3 = sign(pt, v3, v1)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (has_neg and has_pos)
        
        map_length = length
        map_width = width
        height = pose[2]
        depth_image = np.ones((map_length, map_width), dtype=float) * (height - self._min_coords()[2])
        
        start_x = int(map_length / 2.0 + min_length / res - 0.5) - 5
        stop_x = int(map_length / 2.0 + max_length / res - 0.5) + 5
        start_y = int(map_width / 2.0 + min_width / res - 0.5) - 5  
        stop_y = int(map_width / 2.0 + max_width / res - 0.5) + 5
        
        for i in range(start_x, stop_x):
            for j in range(start_y, stop_y):
                pose_x = (i - map_length / 2) * res + res / 2.0
                pose_y = (j - map_width / 2) * res + res / 2.0
                pose_z = height
                pose = [pose_x, pose_y, pose_z]
                    
                for t in self._triangles.tolist():
                    v1 = self._vertices[t[0]]
                    v2 = self._vertices[t[1]]
                    v3 = self._vertices[t[2]]
                    if PointInTriangle(pose, v1, v2, v3) == True:
                        Np = np.cross(v2 - v1, v3 - v1)
                        d = np.dot(Np, v1)
                        dist = (np.dot(Np, pose) - d) / Np[2]
                        if dist < depth_image[i, j]:
                            depth_image[i, j] = dist
                            
        return depth_image
    
    def signed_distance(self, points):
        """Output signed distance of points
        Args:
            points: (n, 3) float array, 3-D coordinates of points
        Returns:
            (n, 1) float array, signed distance of each points
        """
        points = np.reshape(points, (-1, 3))
        return trimesh.proximity.signed_distance(self._trimesh, points)
        
    def _center_vertices_bb(self):
        self._vertices = self._vertices - self._compute_bb_center()
        
    def _compute_bb_center(self):
        bb_center = (self._min_coords() + self._max_coords()) / 2.0
        return bb_center
    
    def _compute_centroid(self):
        return np.mean(self._vertices, axis=0)
        
    def _min_coords(self):
        """
        Returns:
            3-D float vector, minimum coordinates of vertices
        """
        return np.min(self._vertices, axis=0)

    def _max_coords(self):
        """
        Returns:
            3-D float vector, maximum coordinates of vertices
        """
        return np.max(self._vertices, axis=0)
    
    def visualize(self):
        # Visualize its mesh model
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self._vertices[:, 0], self._vertices[:, 1], self._vertices[:, 2], triangles=self._triangles,
                        linewidth=0.2, antialiased=True)
        ax.axis('equal')
        ax.set_aspect('equal')
