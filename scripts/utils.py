import random
import torch
from geometry_msgs.msg import Pose
import math

MAP_DIMS_X = [-10, 14]
MAP_DIMS_Y = [-8, 7]
DIST_FACTOR = 2

def get_goal(map_dims_x=MAP_DIMS_X, map_dims_y=MAP_DIMS_Y):
    x = random.uniform(*map_dims_x)
    y = random.uniform(*map_dims_y)
    
    return torch.tensor([x, y], dtype=torch.float32)

def get_reward(pose, goal, distance_factor=DIST_FACTOR):
    reward = -10
    reward -= distance_factor * math.sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)
    
    return reward