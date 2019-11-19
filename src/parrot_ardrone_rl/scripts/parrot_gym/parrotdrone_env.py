#!/usr/bin/env python3
import time
import rospy
import numpy as np
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge, CvBridgeError
from parrot_gym.robot_gazebo_env import RobotGazeboEnv


class ParrotDroneEnv(RobotGazeboEnv):
    """Superclass for all PX4 MavDrone environments.
    """
    def __init__(self):
        """Initializes a new MavROS environment. \\
        To check the ROS topics, unpause the paused simulation \\
        or reset the controllers if simulation is running.

        Sensors for RL observation space (by topic list): \\
        * /drone/front_camera/image_raw: RGB Camera facing front. 640x360
        * /drone/gt_pose: Get position and orientation in Global space
        * /drone/gt_vel: Get the linear velocity , the angular doesnt record anything.
        }
        
        Actuations for RL action space (by topic list):
        * /cmd_vel: Move the Drone Around when you have taken off.
        * /drone/takeoff: Takeoff drone from current position
        * /drone/land: Land drone at current xy position
        """
        rospy.logdebug("Start ParrotDroneEnv INIT...")

        #Spawn Parrot AR Drone through launch file
        self.ros_pkg_name="drone_construct"
        self.launch_file_name="put_drone_in_world.launch"
        
        super(ParrotDroneEnv, self).__init__(
            ros_pkg_name=self.ros_pkg_name,
            launch_file=self.launch_file_name,
            start_init_physics_parameters=True,
            reset_world_or_sim='WORLD')

        rospy.logdebug("Finished ParrotDroneEnv INIT...")

#Callback functions for Topic Subscribers uesd by TASK environments
    def _front_camera_cb(self, msg):
        self._current_front_camera = Image()
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError as cv_error:
            print(cv_error)
        self._current_front_camera = cv_image
    
    def _gt_pose_cb(self, msg):
        self._current_gt_pose = msg

    def _gt_vel_cb(self, msg):
        self._current_gt_vel = msg
    

# Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _setup_subscribers(self):
        rospy.Subscriber('/drone/front_camera/image_raw', Image, self._front_camera_cb)
        rospy.Subscriber('/drone/gt_pose', Pose , callback=self._gt_pose_cb)
        rospy.Subscriber('/drone/gt_vel' , Twist, callback=self._gt_vel_cb)

    @property
    def current_gt_pose(self):
        return self._current_gt_pose

    @property
    def current_gt_vel(self):
        return self._current_gt_vel

    @property
    def current_front_camera(self):
        return self._current_front_camera

    
    def _check_all_subscribers_ready(self):
        #rospy.logdebug("CHECK ALL Subscribers:")
        self._check_subscriber_ready('/drone/front_camera/image_raw', Image)
        self._check_subscriber_ready('/drone/gt_pose', Pose)
        self._check_subscriber_ready('/drone/gt_vel', Twist)
        #rospy.logdebug("All Sensors CONNECTED and READY!")
    
    def _setup_publishers(self):
        #rospy.logdebug("CHECK ALL PUBLISHERS CONNECTION:")
        self._publish_cmd_vel = rospy.Publisher('/cmd_vel',       Twist, queue_size=1)
        self._publish_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._publish_land    = rospy.Publisher('/drone/land',    Empty, queue_size=1)
        #rospy.logdebug("All Publishers CONNECTED and READY!")
    
    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        #rospy.logdebug("CHECK ALL PUBLISHERS CONNECTION:")
        self._check_publisher_ready(self._publish_cmd_vel.name,self._publish_cmd_vel)
        self._check_publisher_ready(self._publish_takeoff.name,self._publish_takeoff)
        self._check_publisher_ready(self._publish_land.name,self._publish_land)
     


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
        self._check_publisher_ready(self._publish_cmd_vel.name,self._publish_cmd_vel)
        self._publish_cmd_vel.publish(vel_msg)

    def ExecuteTakeoff(self, altitude=0.8):
        """
        Sends the takeoff command and waits for the drone to takeoff
        Gazebo Pause and Unpause to make it a self-contained action
        """
        self.gazebo.unpauseSim()
        self._check_publisher_ready(self._publish_takeoff.name,self._publish_takeoff)
        self._publish_takeoff.publish(Empty())
        self.wait_for_height(desired_height=altitude, to_land=False, epsilon=0.05, update_rate=10)
        self.gazebo.pauseSim()

    def ExecuteLand(self, land_height=0.6):
        """
        Sends the land command and waits for the drone to land
        Gazebo Pause and Unpause to make it a self-contained action
        """
        self.gazebo.unpauseSim()
        self._check_publisher_ready(self._publish_land.name,self._publish_land)
        self._publish_land.publish(Empty())
        self.wait_for_height(desired_height=land_height, to_land=True, epsilon=0.05, update_rate=10)
        self.gazebo.pauseSim()

    def wait_for_height(self, desired_height, to_land, epsilon, update_rate):
        """
        Checks if current height is smaller or bigger than a value
        :param: to_land: If True, we will wait until value is smaller than the one given
        """
        rate = rospy.Rate(update_rate)
        while not rospy.is_shutdown():
            current_height = self._check_subscriber_ready('/drone/gt_pose', Pose).position.z

            if to_land:
                takeoff_height_achieved = (current_height <= desired_height)
            else:
                takeoff_height_achieved = (current_height >= desired_height)

            if takeoff_height_achieved:
                break
            rate.sleep()
    


