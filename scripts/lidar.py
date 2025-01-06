import pybullet as p
import math
import numpy as np
import matplotlib.pyplot as plt

def lidar_sim(robot_id, lidar_idx, num_beams=360, range=5.0):

    # Get LiDAR position and orientation (quaternion) - relative to world
    lidar_pos, lidar_orientation = p.getLinkState(robot_id, lidar_idx)[:2]
    lidar_orientation = p.getEulerFromQuaternion(lidar_orientation)
    angles = np.linspace(0, 2 * math.pi, num_beams, endpoint=False) + lidar_orientation[2]
    
    # set ray_starts and ray_ends
    ray_starts = np.tile(lidar_pos, (num_beams, 1))
    ray_ends = np.array([
        lidar_pos[0] + range * np.cos(angles),
        lidar_pos[1] + range * np.sin(angles),
        lidar_pos[2] * np.ones(len(angles))
    ]).T
    
    # get ray results
    ray_results = p.rayTestBatch(ray_starts, ray_ends)

    # Plot lines (debugging)
    # p.removeAllUserDebugItems()
    # for start, end, result in zip(ray_starts, ray_ends, ray_results):
    #     color = [0, 1, 0] if result[0] != -1 else [1, 0, 0]
    #     p.addUserDebugLine(start, result[3] if result[0] != -1 else end, color, 1.0)
    
    lidar_ranges = []
    for result in ray_results:
        if result[0] == -1: # no hit
            lidar_ranges.append(range)
        else: # hit
            hit_fraction = result[2]
            distance = hit_fraction * range if hit_fraction < 1.0 else range
            lidar_ranges.append(distance)
    
    return np.array(lidar_ranges)