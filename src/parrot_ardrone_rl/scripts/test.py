#!/usr/bin/env python3
import os
import sys
import gym
import time
import numpy as np
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import rosparam
# import training environment
from parrot_gym.roscore_handler import Roscore
from parrot_gym.make_gym_env import GymMake


if __name__ == '__main__':
    roscore = Roscore()
    roscore.run()
    time.sleep(1)
    rospy.init_node('parrotdrone_test', anonymous=True, log_level=rospy.WARN)

    # Init Gym ENV
    task_env = 'ParrotDroneGoto-v0'
    env = GymMake(task_env)
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('parrot_ardrone_rl')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    
    nepisodes = 500
    nsteps = 1000

    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))
        observation = env.reset()
        print(observation[0].shape)
        print(observation[1].shape)
        for i in range(nsteps):
            time.sleep(0.001)
    env.close()
    roscore.terminate()