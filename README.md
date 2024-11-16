# Deep Reinforcement Learning + SLAM for Autonomous Robot Navigation
This project integrates Deep Reinforcement Learning (DRL) with Simultaneous Localization and Mapping (SLAM) to enable autonomous navigation for robots. The system utilizes DRL for decision-making and path planning, while SLAM ensures accurate environmental mapping and localization. Together, they enable the robot to explore unknown environments, avoid obstacles, and reach specified goals efficiently. This approach leverages the strengths of both techniques to create a robust, adaptive navigation system for real-world applications.

After building the package and sourcing the workspace, to be able to run the training loop you need to: 
1. Initialize the Gazebo simulation
```
ros2 launch navigationrl gazebo.launch.py
```
2. Initialize the `slam_toolbox` module
```
ros2 launch slam_toolbox online_async_launch.py params_file:=./src/navigationrl/config/mapper_params_online_async.yaml
```
3. Run the `robot_control.py` file
```
ros2 run navigationrl robot_control.py
```
