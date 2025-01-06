import torch
import pybullet as p
import numpy as np
import random
import math
from scripts.lidar import lidar_sim

STATE_DIM = 7
WHEEL_SEPARATION = 0.2
WHEEL_RADIUS = 0.05

WALL_THICKNESS = 0.1
WALL_HEIGHT = 1.0
ROOM_SIZE = 25
WALL_THRESHOLD = 0.2
GOAL_THRESHOLD = 0.2
DIST_FACTOR = 2
CELL_SIZE = 8  # Size of each cell in the maze grid
WALL_THICKNESS = 0.2
WALL_HEIGHT = 2
PASSAGE_WIDTH = 4  # Width of the passageways
SEGMENT_LENGTH = 6  # Length of individual wall segments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dict_to_tensors(dict, dtype=torch.float32):
    tensor_dict = {key: torch.tensor(value, dtype=dtype).unsqueeze(0).to(device) for key, value in dict.items()}
    return tensor_dict

def states_to_stack(list_dict):
    maps_state = [d['map'] for d in list_dict]
    robot_state = [d['robot'] for d in list_dict]
    lidar_state = [d['lidar'] for d in list_dict]
    maps_state = torch.stack(maps_state).squeeze(1)
    robot_state = torch.stack(robot_state).squeeze(1)
    lidar_state = torch.stack(lidar_state).squeeze(1)
    
    return maps_state, robot_state, lidar_state

def diff_drive_control(linear_vel=0.0, angular_vel=0.0):
    v_right = linear_vel + (angular_vel * WHEEL_SEPARATION / 2)
    v_left = linear_vel - (angular_vel * WHEEL_SEPARATION / 2)
    
    return [- v_right / WHEEL_RADIUS, v_left / WHEEL_RADIUS]

def fit_theta(theta):
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta < -math.pi:
        theta += 2 * math.pi
    return theta

def get_state(robot_id, goal, slam_node):
    state = {}
    
    # get robot position and velocities
    robot_pose = slam_node.robot_pose

    robot_lin_vel, robot_w_vel = p.getBaseVelocity(robot_id)

    # transform linear and angular velocities to robot local frame
    rot_matrix = np.array([
        [np.cos(robot_pose[2]), -np.sin(robot_pose[2]), 0],
        [np.sin(robot_pose[2]), np.cos(robot_pose[2]), 0],
        [0, 0, 1]
    ])
    local_v = np.dot(rot_matrix.T, robot_lin_vel)
    local_w = np.dot(rot_matrix.T, robot_w_vel)

    lin_vel = local_v[0]
    ang_vel = local_w[2]

    # obtain relative position to the goal
    diff_x = goal[0] - robot_pose[0]
    diff_y = goal[1] - robot_pose[1]
    dist_to_goal = np.sqrt(diff_x**2 + diff_y**2)
    angle_to_goal = np.arctan2(diff_y, diff_x) - robot_pose[2]

    state['robot'] = np.array([robot_pose[0], robot_pose[1], robot_pose[2], lin_vel, ang_vel, dist_to_goal, angle_to_goal])
    state['lidar'] = lidar_sim(robot_id, 2)
    state['map'] = slam_node.map[np.newaxis, :, :]
    return state

def get_spawn(radius=(ROOM_SIZE - 2)):
    
    spawn_x = random.uniform(-radius/2, radius/2)
    spawn_y = random.uniform(-radius/2, radius/2)
    spawn_theta = random.uniform(-math.pi, math.pi)
    
    return [spawn_x, spawn_y, spawn_theta]

def get_goal(spawn, radius=10.0):
    dist = random.uniform(0, radius)
    angle = random.uniform(0, 2* math.pi)
    
    goal_x = spawn[0] + dist * math.cos(angle)
    goal_y = spawn[1] + dist * math.sin(angle)
    
    goal = np.array([goal_x, goal_y])
    goal = np.clip(goal, -ROOM_SIZE/2, ROOM_SIZE/2)

    return goal.tolist()
    

def compute_reward_done(state, action, next_state):
    k1, k2, k3, k4, k5, k6 = 5.0, 0.05, 0.1, 0.001, 0.01, 0.5
    done = False

    # goal proximity reward
    dist_reward = -k1 * (1 / next_state['robot'][5])
    if next_state['robot'][5] < 0.05:
        dist_reward += 500.0
        done = True

    # angular deviation
    ang_reward = -k2 * np.abs(next_state['robot'][6])

    # smooth motion penalty
    smoothness_penalty = -k3 * np.abs(action[1])

    # energy penalty
    energy_penalty = -k4 * (action[0]**2 + action[1]**2)

    # time penalty
    time_penalty = -k5
    
    # obstacle penalty
    obstacle_penalty = 0.0
    min_lidar = np.min(next_state['lidar'])
    if min_lidar < 0.25:
        obstacle_penalty += -k6 * (1 - min_lidar)
    if min_lidar < 0.05:
        obstacle_penalty = -1000
        done = True

    total_reward = dist_reward + ang_reward + smoothness_penalty + energy_penalty + time_penalty + obstacle_penalty

    return total_reward, done


def setupEnvironment():
    
    # Create the outer boundary walls
    boundary_wall_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[ROOM_SIZE / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2])
    # Create the random wall
    random_wall_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[ROOM_SIZE / 4, WALL_THICKNESS / 2, WALL_HEIGHT / 2])
    
    # Front wall
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=boundary_wall_collision,
                      basePosition=[0, ROOM_SIZE / 2, WALL_HEIGHT / 2])
    # Back wall
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=boundary_wall_collision,
                      basePosition=[0, -ROOM_SIZE / 2, WALL_HEIGHT / 2])
    # Left wall
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=boundary_wall_collision,
                      basePosition=[-ROOM_SIZE / 2, 0, WALL_HEIGHT / 2],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]))
    # Right wall
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=boundary_wall_collision,
                      basePosition=[ROOM_SIZE / 2, 0, WALL_HEIGHT / 2],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]))
    
    # Random wall
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=random_wall_collision,
                      basePosition=[ROOM_SIZE / 10, 0, WALL_HEIGHT / 2],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]))
