#!/usr/bin/env python3
import gym
from gym_link.reg_gym_env import Register_Gym_Env
import rospy
import os


def GymMake(robot_task_env_name, max_steps_per_ep=10000):
    """
    Register the targeted env, and call the gym.make() function
    """
    rospy.logwarn("Env: {} will be imported".format(robot_task_env_name))
    result = Register_Gym_Env(task_env=robot_task_env_name,\
                                    max_episode_steps_per_episode=max_steps_per_ep)

    if result:
        #rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        env = gym.make(robot_task_env_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env