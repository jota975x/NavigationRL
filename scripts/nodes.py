import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import tf_transformations
import numpy as np
from std_srvs.srv import Empty

MAP_SIZE = 550
MAXIMUM_RANGE = 5.0


class SlamIntegrationNode(Node):
    def __init__(self):
        super().__init__('slam_integration_node')

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('Created LidarScanner Subsciptor')
        self.lidar_data = None

        # Subscribe to the map topic
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.processed_map = None

        # Create DRL map (this ensures map passed into DRL agent has same size always)
        self.drl_map = np.zeros(shape=[MAP_SIZE, MAP_SIZE])
        self.wall_threshold = 0.3

        # Initialize Transform Listener for localization (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("SLAM Integration Node Initialized.")

    def scan_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)
        self.lidar_data[np.isinf(self.lidar_data)] = MAXIMUM_RANGE

    def is_goal_reached(self, goal, pose):
        """
        Check if the robot has reached the goal.

        Args:
            goal (torch.Tensor): The goal coordinates as a tensor [x, y].

        Returns:
            bool: True if the goal is reached, False otherwise.
        """
        if pose is None:
            return False  # Cannot determine pose

        x_robot, y_robot, _ = pose
        x_goal, y_goal = goal[0].item(), goal[1].item()

        # Calculate Euclidean distance to the goal
        distance = np.sqrt((x_goal - x_robot) ** 2 + (y_goal - y_robot) ** 2)

        # Check if within the threshold
        threshold = 0.1  # Adjust as needed
        return distance < threshold

    def is_wall_close(self):
        """
        Check if a wall is within the threshold distance in the front of the robot.
        Returns:
            bool: True if a wall is detected, False otherwise.
        """
        if self.lidar_data is None:
            return False  # No data available

        # Define the range of angles for the "front" section
        front_angle_start = 350  # Degrees near the front-left (in LIDAR indices)
        front_angle_end = 10  # Degrees near the front-right (wrap-around case)
        
        # Define the range of angles for the "front" section (robot can also go backwards)
        back_angle_start = 170
        back_angle_end = 190

        # Extract ranges for the front section
        lidar_ranges = np.concatenate((
            self.lidar_data[front_angle_start:],
            self.lidar_data[:front_angle_end],
            self.lidar_data[back_angle_start:back_angle_end]
        ))

        # Check if any range in the front is below the threshold
        return np.any(lidar_ranges < self.wall_threshold)

    def map_callback(self, msg):
        """
        Callback to process the map received from SLAM Toolbox.
        """
        # self.get_logger().info(f"Map received: {msg.info.width} x {msg.info.height}")
        # Convert OccupancyGrid data to a 2D numpy array
        map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        # Normalize map values (-1: unknown, 0: free, 100: occupied)
        # Normalize to range [0, 1] where 0 = free, 1 = occupied
        map_data = np.clip(map_data, 0, 100) / 100.0

        self.processed_map = map_data  # Store for use in the DRL state

    def get_robot_pose(self):
        """
        Get the robot's pose in the map frame using tf2.
        """
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                'map',  # Source frame (global map)
                'base_footprint',  # Target frame (robot base)
                now
            )
            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            euler = tf_transformations.euler_from_quaternion(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            )
            x, y, theta = translation.x, translation.y, euler[2]
            # self.get_logger().info(f"Robot Pose: x={x}, y={y}, theta={theta}")
            return x, y, theta
        except Exception as e:
            # self.get_logger().warn(f"Failed to get robot pose: {e}")
            return None

    def get_drl_state(self):
        """
        Combine map, pose, and sensor data into a single state for the DRL algorithm.
        Returns:
            state (dict): A dictionary containing map, pose, and sensor data.
        """
        state = {}

        # Map data
        if hasattr(self, 'processed_map') and self.processed_map is not None:
            # Paste SLAM map into DRL map
            drl_map_rows, drl_map_cols = self.drl_map.shape
            slam_map_rows, slam_map_cols = self.processed_map.shape

            # Assertion to ensure DRL map matrix is larger than SLAM map
            assert drl_map_rows >= slam_map_rows and drl_map_cols >= slam_map_cols, (
                "The empty matrix must be larger than or equal to the map matrix in both dimensions."
            )

            start_row = (drl_map_rows - slam_map_rows) // 2
            start_col = (drl_map_cols - slam_map_cols) // 2
            self.drl_map[start_row:start_row + slam_map_rows, start_col:start_col + slam_map_cols] = self.processed_map

            state['map'] = self.drl_map
        else:
            state['map'] = np.zeros((MAP_SIZE, MAP_SIZE))  # Placeholder if map is unavailable

        # Robot pose
        pose = self.get_robot_pose()
        if pose:
            state['pose'] = pose
        else:
            state['pose'] = (0.0, 0.0, 0.0)  # Default pose if unavailable

        # LIDAR data
        if self.lidar_data is not None:
            # Handle 'inf' values
            self.lidar_data[np.isinf(self.lidar_data)] = MAXIMUM_RANGE
            state['lidar'] = self.lidar_data
        else:
            state['lidar'] = np.zeros(360)  # Placeholder for 360-degree LIDAR
        # self.get_logger().info(f"DRL state: {state}")
        return state


# TODO: add logic (preprocessing to position and lidar data)

class velNode(Node):
    def __init__(self):
        super().__init__('velNode')

        self.velPub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.get_logger().info('Created Velocity Command Publisher')
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.command = None

        self.subscription = self.create_subscription(Twist, '/diff_cont/cmd_vel_unstamped', self.vel_callback, 10)
        self.get_logger().info('Created Velocity Command Subscriber')
        self.latest_vel = None

    # Publisher functions
    def timer_callback(self):
        if self.command:
            self.velPub.publish(self.command)

    def step(self, linear=0.0, angular=0.0):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular

        self.command = command

    # Subscription functions
    def vel_callback(self, msg):
        self.latest_vel = msg


class posNode(Node):
    def __init__(self):
        super().__init__('posNode')

        self.subscription = self.create_subscription(Odometry, '/diff_cont/odom', self.pos_callback, 10)
        self.get_logger().info('Created Robot Position Subscriber')
        self.latest_pos = None

    def pos_callback(self, msg):
        self.latest_pos = msg.pose.pose
        
        
class ResetWorldClient(Node):
    def __init__(self):
        super().__init__('reset_world_client')
        
        self.client = self.create_client(Empty, '/reset_world')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /gazebo/reset_world not available, waiting...')
        self.get_logger().info('Created World Reset client')
        
    def reset_world(self):
        request = Empty.Request()
        future = self.client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Successfully reset the world!')
        else:
            self.get_logger().error('Failed to reset the world.')
