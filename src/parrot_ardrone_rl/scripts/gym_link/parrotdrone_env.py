#!/usr/bin/env python3
import numpy
import rospy
import time
from gym_link import robot_gazebo_env
from std_msgs.msg import Float64, Empty
from sensor_msgs.msg import JointState, Image, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Range, Imu
from gym_link.roslauncher import ROSLauncher



class ParrotDroneEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all PX4 MavDrone environments.
    """
    def __init__(self):
        """Initializes a new MavROS environment. \\
        To check the ROS topics, unpause the paused simulation \\
        or reset the controllers if simulation is running.

        Sensors for RL observation space (by topic list): \\
        
        * /drone/gt_pose
        * /drone/ge_vel
        }
        
        Actuations for RL action space (by topic list):
        * /cmd_vel: Move the Drone Around when you have taken off.
        * /drone/takeoff: Takeoff drone from current position
        * /drone/land: Land drone at current xy position

        Args:
        """

        rospy.logdebug("Start ParrotDroneEnv INIT...")

        #Sleep rate for when trying for Publishers
        self._rate = rospy.Rate(10)

        self.controllers_list = []
        self.robot_name_space = ""
        
        super(ParrotDroneEnv, self).__init__(controllers_list=self.controllers_list,
                                             robot_name_space=self.robot_name_space,
                                             reset_controls=False,
                                             start_init_physics_parameters=False,
                                             reset_world_or_sim="WORLD")
        self.gazebo.unpauseSim()

        self.ros_launcher = ROSLauncher(rospackage_name="drone_construct",\
                            launch_file_name="put_drone_in_worlds.launch")
        
        
        self._CheckAllSensors()
        rospy.Subscriber('/drone/gt_pose', Pose , callback=self._gt_pose_cb)
        rospy.Subscriber('/drone/gt_vel' , Twist, callback=self._gt_vel_cb)
        
        self._publish_cmd_vel = rospy.Publisher('/cmd_vel',       Twist, queue_size=1)
        self._publish_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._publish_land    = rospy.Publisher('/drone/land',    Empty, queue_size=1)
        self._CheckAllPublishers()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished ParrotDroneEnv INIT...")

#Callback functions for Topic Subscribers uesd by TASK environments
    def _gt_pose_cb(self, msg):
        self._current_gt_pose = msg

    def _gt_vel_cb(self, msg):
        self._current_gt_vel = msg

# Methods needed by the RobotGazeboEnv
    # ----------------------------

    

    def _CheckAllRLTopics(self):
        """
        Checks that all the sensors, publishers, services and other simulation systems are
        operational.
        """
        self._CheckAllSensors()
        self._CheckAllPublishers()
        return True
    

    def _CheckAllSensors(self):
        #rospy.logdebug("CHECK ALL SENSORS CONNECTION:")
        self._CheckCurrentGtPose()
        self._CheckCurrentGtVel()
        #rospy.logdebug("All Sensors CONNECTED and READY!")
    
    def _CheckCurrentGtPose(self):

        self._current_gt_pose = None
        while self._current_gt_pose is None and not rospy.is_shutdown():
            try:
                self._current_gt_pose = rospy.wait_for_message("/drone/gt_pose", Pose, timeout=5.0)
            except:
                rospy.logdebug("Current /drone/gt_pose not ready, retrying for getting Global Pose")

        return self._current_gt_pose
    
    def _CheckCurrentGtVel(self):

        self._current_gt_vel = None
        while self._current_gt_vel is None and not rospy.is_shutdown():
            try:
                selself._ratevel = rospy.wait_for_message("/drone/gt_vel", Pose, timeout=5.0)
            except:
                rospy.logdebug("Current /drone/gt_vel not ready, retrying for getting Global Velocity")

        return self._current_gt_vel


    def _CheckAllPublishers(self):
        """
        Checks that all the publishers are working
        :return:
        """
        #rospy.logdebug("CHECK ALL PUBLISHERS CONNECTION:")
        self._CheckCmdVelConnection()
        self._CheckTakeoffConnection()
        self._CheckLandConnection()
        
    def _CheckCmdVelConnection(self):
        while self._publish_cmd_vel.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_publish_cmd_vel Publisher Connected")

    def _CheckTakeoffConnection(self):
        while self._publish_take.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_local_takeoff Publisher Connected")

    def _CheckLandConnection(self):
        while self._publish_land.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_publish_land Publisher Connected")
    



    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
     
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def publish_vel(self, vel_msg, epsilon=0.05, update_rate=20):
        """
        Execute the velocity command
        """
        self._CheckCmdVelConnection()
        self._publish_cmd_vel.publish(vel_msg)

    def ExecuteTakeoff(self, altitude=0.8):
        """
        Sends the takeoff command and waits for the drone to takeoff
        Gazebo Pause and Unpause to make it a self-contained action
        """
        self.gazebo.unpauseSim()
        self._CheckTakeoffConnection()
        self._publish_takeoff.publish(Empty())
        self.wait_for_height(desired_height=altitude, to_land=False, epsilon=0.05, update_rate=10)
        self.gazebo.pauseSim()

    def ExecuteLand(self, land_height=0.6):
        """
        Sends the land command and waits for the drone to land
        Gazebo Pause and Unpause to make it a self-contained action
        """
        self.gazebo.unpauseSim()
        self._CheckLandConnection()
        self._publish_land.publish(Empty())
        self.wait_for_height(desired_height=land_height, to_land=True, epsilon=0.05, update_rate=10)
        self.gazebo.pauseSim()

    def wait_for_height(self, desired_height, to_land, epsilon, update_rate):
        """
        Checks if current height is smaller or bigger than a value
        :param: to_land: If True, we will wait until value is smaller than the one given
        """
        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        while not rospy.is_shutdown():
            current_height = self._CheckCurrentGtPose().position.z

            if to_land:
                takeoff_height_achieved = (current_height <= heigh_value_to_check)
            else:
                takeoff_height_achieved = (current_height >= heigh_value_to_check)

            if takeoff_height_achieved:
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rate.sleep()
    


    def get_current_gt_pose(self):
        return self._current_gt_pose

    def get_current_gt_vel(self):
        return self._current_gt_vel

