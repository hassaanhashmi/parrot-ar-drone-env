# ParrotDrone-RL-Experiments
A ROS package to setup a Parrot AR done OpenAI Gym environment for Reinforcement Learning experiments

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