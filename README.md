A ROS package to setup a Parrot AR drone OpenAI Gym environment for Reinforcement Learning experiments

#### Building Workspace in python3:
After cloning the repository, follow the following steps:
```
cd ParrotDrone-RL-Experiments
wstool up
rosdep install --from-paths src --ignore-src -y -r
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```
A sample test file can be found at ```src/parrot_ardrone_rl/scripts```. Currently I am subscribing to ```gt_pose```, ```gt_vel``` and ```front_camera/image_raw``` topics as the observation tuple (Dimensions are ```[(13,),(360,640,3)]```).

For the action space, I am publishing ```cmd_vel``` with angular x and y velocities set to zero. You can edit these settings in ```parrot_gym/parrotdrone_env.py```.

The tasks are defined in ```parrot_gym/parrotdrone_tasks```.

Note: I recommend creating a virtual python3 environment and using the test file as a guide to build your own training/testing project.
