#!/usr/bin/env python3
import os
import numpy as np
import gym
import numpy
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import rosparam
# import training environment
from gym_link.make_gym_env import GymMake


if __name__ == '__main__':
    
    rospy.init_node('parrotdrone_goto_ddpg', anonymous=True, log_level=rospy.WARN)

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

    last_time_steps = numpy.ndarray(0)

    
    nepisodes = 500
    nsteps = 1000

    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))
        observation = env.reset()
        print(observation[1].shape)
        for i in range(nsteps):
            rospy.sleep(0.01)
    env.close()
