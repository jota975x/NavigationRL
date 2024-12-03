import numpy as np
import math
from scripts.lidar import lidar_sim
from scripts.utils import fit_theta
import matplotlib.pyplot as plt

MAP_RESOLUTION = 0.1    # each cell in the grid map is (MAP_RESOLUTION meters)
MAP_SIZE = 30   # map_size (meters)
GRID_SIZE = int(MAP_SIZE / MAP_RESOLUTION)  # grid_map size
CONFIDENCE_GAIN = 0.5
CONFIDENCE_DECAY = 1 # 0.99 # test with map decay later
MAX_RANGE = 5.0
DELTA_T = 1.0 / 240.0
NOISE = (1e-6, 1e-6, 1e-6, 1e-4)
OCCUPANCY_THRESHOLD = 0.2

        
# SLAM Implementation
class SLAM:
    def __init__(self, robot_id, init_pos=[0, 0, 0], max_range: float=MAX_RANGE,
                 map_size: int=MAP_SIZE, grid_size: int=GRID_SIZE, map_resolution: float=MAP_RESOLUTION, confidence: float=CONFIDENCE_GAIN,
                 confidence_decay=CONFIDENCE_DECAY, delta_t: float=DELTA_T, occupancy_threshold=OCCUPANCY_THRESHOLD, noise=NOISE):
        super().__init__()
        
        # Robot
        self.robot_id = robot_id
        
        # Mapping variables
        self.map_resolution = map_resolution
        self.grid_size = grid_size
        self.map_size = map_size
        self.confidence = confidence
        self.confidence_decay = confidence_decay
        self.max_range = max_range
        self.map = np.zeros((grid_size, grid_size))
        
        # Localization variables
        self.delta_t = delta_t
        self.P = np.eye(3) * 1e-3   # init small uncertainty
        self.noise = noise
        self.Q = np.diag([noise[0], noise[1], noise[2]])
        self.R = np.diag([noise[3]])
        self.occupancy_threshold = occupancy_threshold
        
        # Init robot_pose
        self.robot_pose = init_pos
        self.robot_grid_position = [
            int(self.robot_pose[0] / self.map_resolution),
            int(self.robot_pose[1] / self.map_resolution)
        ]
        
    # Reset robot_pose
    def set_robot_pose(self, x, y, theta):
        self.robot_pose = [x, y, theta]
        
        # Reset noise
        self.P = np.eye(3) * 1e-3   # init small uncertainty
        self.Q = np.diag([self.noise[0], self.noise[1], self.noise[2]])
        self.R = np.diag([self.noise[3]])
    
    # Get mapping data
    def get_local_map(self, lidar_data):
        # create angles
        angles = np.linspace(0, 2*math.pi, 360) + self.robot_pose[2]
        
        # convert lidar_data into from polar coordinate to cartesian coordinates (meters)
        x_local = lidar_data * np.cos(angles)
        y_local = lidar_data * np.sin(angles)
        coords = np.stack((x_local, y_local), axis=0)
        
        x_global = coords[0, :] + self.robot_pose[0]
        y_global = coords[1, :] + self.robot_pose[1]
        
        # transform into mapgrid coordinates
        x_map = ((x_global + self.map_size / 2) / self.map_resolution).astype(int)
        y_map = ((y_global + self.map_size / 2) / self.map_resolution).astype(int)
        
        return np.array(x_map), np.array(y_map)
    
    # update map with mapping data
    def update_map(self, robot_id):
        lidar_data = lidar_sim(robot_id, 2)
        
        x_map, y_map = self.get_local_map(lidar_data)

        # only pass coordinates which are not max_range
        x_map = x_map[(lidar_data < self.max_range)]
        y_map = y_map[(lidar_data < self.max_range)]
        
        # ensure x_map and y_map are inside map bounds
        x_map = np.clip(x_map, 0, self.grid_size - 1)
        y_map = np.clip(y_map, 0, self.grid_size - 1)
        
        # update map with confidence scores
        self.map[y_map, x_map] += self.confidence
        self.map = np.clip(self.map, 0, 1) * self.confidence_decay
        
    # Localization
    def motion_model(self, v, omega):
        # Transform world-relative velocities to robot's frame
        rotation_matrix = np.array([[np.cos(self.robot_pose[2]), -np.sin(self.robot_pose[2]), 0],
                                    [np.sin(self.robot_pose[2]), np.cos(self.robot_pose[2]), 0],
                                    [0, 0, 1]])
        local_v = np.dot(rotation_matrix.T, v)
        local_w = np.dot(rotation_matrix.T, omega)
        
        # Motion model
        x_new = self.robot_pose[0] + local_v[0] * np.cos(self.robot_pose[2]) * self.delta_t
        y_new = self.robot_pose[1] + local_v[0] * np.sin(self.robot_pose[2]) * self.delta_t
        theta_new = fit_theta(self.robot_pose[2] + local_w[2] * self.delta_t)
        
        # update pose
        self.robot_pose = [x_new, y_new, theta_new]

        