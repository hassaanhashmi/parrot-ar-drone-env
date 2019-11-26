#!/usr/bin/env python3
import rospy
import rospkg
import rosparam
import numpy as np
from gym.spaces import Box, Tuple
from parrotdrone_env import ParrotDroneEnv
from gym.envs.registration import register
from geometry_msgs.msg import Point, Pose, Twist, Vector3
from geometry_msgs.msg import Vector3
from tf.transformations import quaternion_to_euler
from roslauncher import ROSLauncher
import os

class ParrotDroneGotoEnv(ParrotDroneEnv):
    def __init__(self):
        """
        Make parrotdrone learn how to go to a point in world
        """

        ROSLauncher(rospackage_name="drone_construct", 
                    launch_file_name="start_world.launch")
        self._load_config_params()

        super(ParrotDroneGotoEnv, self).__init__()


        self.vel_msg = Twist()
        self._rate = rospy.Rate(10.0) # ros run rate




    def _load_config_params(self):

        # Load Params from the desired Yaml file
        rospkg_path = rospkg.RosPack().get_path("parrot_ardrone_rl")
        config_file_name = "parrotdrone_goto.yaml"
        config_file_path = os.path.join(rospkg_path,
                            "scripts/parrot_gym/parrotdrone_tasks/config/"
                            + str(config_file_name))
        parameters_list=rosparam.load_file(config_file_path)
        for params, namespace in parameters_list:
            rosparam.upload_params(namespace,params)

        # Continuous action space
        hv_range = rospy.get_param('/parrotdrone/lxy_vel_range')
        vv_range = rospy.get_param('/parrotdrone/lz_vel_range')
        rv_range = rospy.get_param('/parrotdrone/rot_vel_range')
        
        self.action_low= np.array([-hv_range, -hv_range, -vv_range, -rv_range])
        self.action_high = np.array([hv_range, hv_range, vv_range, rv_range])
        self.action_space=Box(low=self.action_low, high=self.action_high,
                                    dtype=np.float32)

        self.reward_range = (-np.inf, np.inf)

        self.init_vel_vec = Twist()
        self.init_vel_vec.linear.x  = rospy.get_param(
                                  '/parrotdrone/init_velocity_vector/linear_x')
        self.init_vel_vec.linear.y  = rospy.get_param(
                                  '/parrotdrone/init_velocity_vector/linear_y')
        self.init_vel_vec.linear.z  = rospy.get_param(
                                  '/parrotdrone/init_velocity_vector/linear_z')
        self.init_vel_vec.angular.x = rospy.get_param(
                                 '/parrotdrone/init_velocity_vector/angular_x')
        self.init_vel_vec.angular.y = rospy.get_param(
                                 '/parrotdrone/init_velocity_vector/angular_y')
        self.init_vel_vec.angular.z = rospy.get_param(
                                 '/parrotdrone/init_velocity_vector/angular_z')

        # Get WorkSpace Cube Dimensions
        self.work_space_x_max= rospy.get_param("/parrotdrone/work_space/x_max")
        self.work_space_x_min= rospy.get_param("/parrotdrone/work_space/x_min")
        self.work_space_y_max= rospy.get_param("/parrotdrone/work_space/y_max")
        self.work_space_y_min= rospy.get_param("/parrotdrone/work_space/y_min")
        self.work_space_z_max= rospy.get_param("/parrotdrone/work_space/z_max")
        self.work_space_z_min= rospy.get_param("/parrotdrone/work_space/z_min")

        # Maximum Quaternion values
        self.max_qw = rospy.get_param("/parrotdrone/max_orientation/w")
        self.max_qx = rospy.get_param("/parrotdrone/max_orientation/x")
        self.max_qy = rospy.get_param("/parrotdrone/max_orientation/y")
        self.max_qz = rospy.get_param("/parrotdrone/max_orientation/z")

        # Maximum velocity values
        self.max_vel_lin_x = rospy.get_param(
                                   "/parrotdrone/max_velocity_vector/linear_x")
        self.max_vel_lin_y = rospy.get_param(
                                   "/parrotdrone/max_velocity_vector/linear_y")
        self.max_vel_lin_z = rospy.get_param(
                                   "/parrotdrone/max_velocity_vector/linear_z")
        self.max_vel_ang_x = rospy.get_param(
                                  "/parrotdrone/max_velocity_vector/angular_x")
        self.max_vel_ang_y = rospy.get_param(
                                  "/parrotdrone/max_velocity_vector/angular_y")
        self.max_vel_ang_z = rospy.get_param(
                                  "/parrotdrone/max_velocity_vector/angular_z")

        #Front camera resolution
        self.front_cam_h = rospy.get_param("/parrotdrone/front_cam_res/height")
        self.front_cam_w = rospy.get_param("/parrotdrone/front_cam_res/width")

        # Get Desired Point to Get
        self.desired_pose = Pose()
        self.desired_pose.position.x = rospy.get_param(
                                             "/parrotdrone/desired_position/x")
        self.desired_pose.position.y = rospy.get_param(
                                             "/parrotdrone/desired_position/y")
        self.desired_pose.position.z = rospy.get_param(
                                             "/parrotdrone/desired_position/z")
        self.desired_pose.orientation.w = rospy.get_param(
                                          "/parrotdrone/desired_orientation/w")
        self.desired_pose.orientation.x = rospy.get_param(
                                          "/parrotdrone/desired_orientation/x")
        self.desired_pose.orientation.y = rospy.get_param(
                                          "/parrotdrone/desired_orientation/y")
        self.desired_pose.orientation.z = rospy.get_param(
                                          "/parrotdrone/desired_orientation/z")


        self.desired_pose_epsilon = rospy.get_param(
                                          "/parrotdrone/desired_point_epsilon")
        
        self.geo_distance = rospy.get_param("/parrotdrone/geodesic_distance")


        # We place the Maximum and minimum values of the X,Y,Z,W,X,Y,Z 
        # of the pose

        numeric_high = np.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.max_qw,
                            self.max_qx,
                            self.max_qy,
                            self.max_qz,
                            self.max_vel_lin_x,
                            self.max_vel_lin_y,
                            self.max_vel_lin_z,
                            self.max_vel_ang_x,
                            self.max_vel_ang_y,
                            self.max_vel_ang_z])

        numeric_low = np.array([self.work_space_x_min,
                        self.work_space_y_min,
                        self.work_space_z_min,
                        -1*self.max_qw,
                        -1*self.max_qx,
                        -1*self.max_qy,
                        -1*self.max_qz,
                        -1*self.max_vel_lin_x,
                        -1*self.max_vel_lin_y,
                        -1*self.max_vel_lin_z,
                        -1*self.max_vel_ang_x,
                        -1*self.max_vel_ang_y,
                        -1*self.max_vel_ang_z])

        self.numeric_obs_space = Box(numeric_low, numeric_high, 
                                     dtype=np.float32)
        self.image_obs_space = Box(low=0, high=255, 
                                  shape=(self.front_cam_h,self.front_cam_w, 3),
                                  dtype=np.uint8)
        self.observation_space = Tuple([self.numeric_obs_space, 
                                        self.image_obs_space])


        # rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        # rospy.logdebug("OBSERVATION SPACES TYPE===>" +
        #             str(self.observation_space))

        # Rewards
        self.closer_to_point_reward = rospy.get_param(
                                        "/parrotdrone/closer_to_point_reward")
        self.not_ending_point_reward = rospy.get_param(
                                        "/parrotdrone/not_ending_point_reward")
        self.end_episode_points = rospy.get_param(
                                        "/parrotdrone/end_episode_points")
        self.cumulated_steps = 0.0

    def _set_init_pose(self):
        """
        Sets the Robot in its init linear and angular speeds.
        Its preparing it to be reseted in the world.
        """
        self.publish_vel(self.init_vel_vec, epsilon=0.05, update_rate=10)
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at 
        the start of an episode.
        :return:
        """
        self.gazebo.unpauseSim()
        
        self.ExecuteTakeoff(altitude=0.8)

        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to measure the distance from 
        #the desired point.
        curr_position = self.current_gt_pose.position
        self.prev_dist_des_point = self.get_dist_des_point(curr_position)

        # self.prev_diff_des_orient = \
        # self.get_diff_des_orient(self.current_gt_pose.orientation)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the 
        drone based on the action number given.
        :param action: The action integer that set s what movement to do
         next.
        """
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the 
        # parent class of Parrot
        action_vel = TwistStamped()
        action_vel.twist.linear.x   = action[0]
        action_vel.twist.linear.y   = action[1]
        action_vel.twist.linear.z   = action[2]
        action_vel.twist.angular.x  = 0.0
        action_vel.twist.angular.y  = 0.0
        action_vel.twist.angular.z  = action[3]
        
        # We tell drone the linear and angular velocities to set to 
        # execute
        self.publish_vel(action_vel, epsilon=0.05, update_rate=20)
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        #rospy.logdebug("Start Get Observation ==>")
        # We get the global pose and velocity data as observation
        curr_gt_pose = self.current_gt_pose
        curr_gt_vel = self.current_gt_vel
        curr_front_cam= np.asarray(self.current_front_camera)

        numeric_obs = np.array([curr_gt_pose.position.x,
                        curr_gt_pose.position.y, 
                        curr_gt_pose.position.z,
                        curr_gt_pose.orientation.w,
                        curr_gt_pose.orientation.x,
                        curr_gt_pose.orientation.y,
                        curr_gt_pose.orientation.z,
                        curr_gt_vel.linear.x,
                        curr_gt_vel.linear.y,
                        curr_gt_vel.linear.z,
                        curr_gt_vel.angular.x,
                        curr_gt_vel.angular.y,
                        curr_gt_vel.angular.z])
        # rospy.logdebug("Observations==>"+str(observations))
        # rospy.logdebug("END Get Observation ==>")
        return [numeric_obs, curr_front_cam]

    def _is_done(self, observations):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It detected something with the sonar that is too close
        3) It flipped due to a crash or something
        4) It has reached the desired point
        """

        episode_done = False
        current_pose = observations[:7]
        current_position = observations[:3]
        current_orientation = observations[3:7]
        

        is_inside_workspace_now = self.is_inside_workspace(current_position)
        too_close_to_grnd       = self.too_close_to_ground(current_position[2])
        drone_flipped           = self.drone_has_flipped(current_orientation)
        has_reached_des_pose    = self.is_in_desired_pose(current_pose, 
                                                     self.desired_pose_epsilon)

        rospy.logwarn(">>>>>> DONE RESULTS <<<<<")

        if not is_inside_workspace_now:
            rospy.logerr("Drone is outside workspace")

        if too_close_to_grnd:
            rospy.logerr("Drone is too close to ground")
        

        if drone_flipped:
            rospy.logerr("Drone has flipped")
        

        if has_reached_des_pose:
            rospy.logerr("Drone has reached the desired pose")

        # We see if we are outside the Learning Space
        episode_done = not(is_inside_workspace_now) or\
                        too_close_to_grnd or\
                        drone_flipped or\
                        has_reached_des_pose

        if episode_done:
            rospy.logerr("episode_done====>"+str(episode_done))
        else:
            rospy.logwarn("episode running! \n")

        return episode_done

    def _compute_reward(self, observations, done):
        
        current_pose = PoseStamped()
        current_pose.pose.position.x    = observations[0]
        current_pose.pose.position.y    = observations[1]
        current_pose.pose.position.z    = observations[2]
        current_pose.pose.orientation.w = observations[3]
        current_pose.pose.orientation.x = observations[4]
        current_pose.pose.orientation.y = observations[5]
        current_pose.pose.orientation.z = observations[6]
        
        curr_position = current_pose.pose.position
        curr_orientation = current_pose.pose.orientation

        dist_des_point = self.get_dist_des_point(curr_position)
        
        diff_des_orientation = self.get_diff_des_orient(curr_orientation)
        
        distance_difference = dist_des_point - self.prev_dist_des_point + \
                           2*(diff_des_orientation - self.prev_diff_des_orient)

        if not done:

            # If there has been a decrease in the distance to the 
            # desired point, we reward it
            if distance_difference < 0.0:
                # rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = self.closer_to_point_reward
            else:
                # rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward = 0

        else:

            if self.is_in_desired_pose(current_position, epsilon=0.5):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points
        
        self.cumulated_reward += reward
        self.cumulated_steps += 1

        self.prev_dist_des_point = dist_des_point
        self.prev_diff_des_orient = diff_des_orientation
        return reward

    # Internal TaskEnv Methods

    def is_in_desired_pose(self, current_pose, epsilon=0.05):
        """
        It return True if the current position is similar to the desired
        poistion
        """

        is_in_desired_pose = False
        current_pose = np.array(current_pose)
        desired_pose = np.array([self.desired_pose.pose.position.x,\
                        self.desired_pose.pose.position.y,\
                        self.desired_pose.pose.position.z,\
                        self.desired_pose.pose.orientation.w,\
                        self.desired_pose.pose.orientation.x,\
                        self.desired_pose.pose.orientation.y,\
                        self.desired_pose.pose.orientation.z])
        
        desired_pose_plus = desired_pose + epsilon
        desired_pose_minus= desired_pose - epsilon

        pose_are_close = np.all(current_pose <= desired_pose_plus) and \
                         np.all(current_pose >  desired_pose_minus)


        return is_in_desired_pose

    def is_inside_workspace(self, current_position):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False

        if current_position[0] > self.work_space_x_min and \
            current_position[0] <= self.work_space_x_max:
            if current_position[1] > self.work_space_y_min and \
                current_position[1] <= self.work_space_y_max:
                if current_position[2] > self.work_space_z_min and \
                    current_position[2] <= self.work_space_z_max:
                    is_inside = True

        return is_inside

    def too_close_to_ground(self, current_position_z):
        """
        Detects if there is something too close to the drone front
        """
        too_close = current_position_z < self.min_height
        return too_close

    def drone_has_flipped(self, curr_orient):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True

        curr_roll, curr_pitch, curr_yaw = quaternion_to_euler([curr_orient[1],
                                                               curr_orient[2],
                                                               curr_orient[3],
                                                               curr_orient[0]])
        self.max_roll = rospy.get_param("/parrotdrone/max_roll")
        self.max_pitch = rospy.get_param("/parrotdrone/max_pitch")

        rospy.logwarn("#### HAS FLIPPED? ########")
        rospy.logwarn("RPY current_orientation"+
                       str(curr_roll, curr_pitch, curr_yaw))
        rospy.logwarn("max_roll"+str(self.max_roll) +
                      ",min_roll="+str(-1*self.max_roll))
        rospy.logwarn("max_pitch"+str(self.max_pitch) +
                      ",min_pitch="+str(-1*self.max_pitch))
        rospy.logwarn("############")

        if curr_roll > -1*self.max_roll and curr.roll <= self.max_roll:
            if curr_pitch > -1*self.max_pitch and curr_pitch <= self.max_pitch:
                has_flipped = False

        return has_flipped

    def self.get_dist_des_point(self, current_pos):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        curr_position = np.array([current_position.x, 
                                  current_position.y, 
                                  current_position.z])
        des_position = np.array([self.desired_pose.position.x,\
                                self.desired_pose.position.y,\
                                self.desired_pose.position.z])
        dist = self.get_distance_between_points(curr_position, des_position)

        return dist

    def get_distance_between_points(self, p_start, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array(p_start)
        b = np.array(p_end)

        distance = np.linalg.norm(a - b)

        return distance

    def get_diff_des_orient(self, current_orientation):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        curr_orientation = np.array([current_orientation.w, 
                                     current_orientation.x, 
                                     current_orientation.y, 
                                     current_orientation.z])
        des_orientation = np.array([self.desired_pose.pose.orientation.w,\
                                self.desired_pose.pose.orientation.x,\
                                self.desired_pose.pose.orientation.y,\
                                self.desired_pose.pose.orientation.z])
        diff = self.get_diff_btw_orient(curr_orientation, des_orientation)

        return difference
    
    def get_diff_btw_orient(self, ostart, o_end):
        """
        Given an orientation Object, get difference from current orientation
        :param p_end:
        :return:
        """
        ostart_norm_sq = np.dot(ostart, ostart)
        if self.geo_distance == True:   #<-- Geodesic distance
            if ostart_norm_sq > 0:
                ostart_conj=np.array((ostart[0],-1*ostart[1:4]))/ostart_norm_sq
            else:
                rospy.logerr("can not compute for a quaternion with 0 norm")
                return float('NaN')
        
            o_product = ostart_conj * o_end
            o_product_vector = o_product[1:4]
        
            v_product_norm = np.linalg.norm(o_product_vector)
            o_product_norm = sqrt(np.dot(o_product, o_product))

            tolerance = 1e-17
            if o_product_norm < tolerance:
                # 0 quaternion - undefined
                o_diff= np.array([-float('inf'),float('nan')*o_product_vector])
            if v_product_norm < tolerance:
                # real quaternions - no imaginary part
                o_diff = np.array([log(o_product_norm),0,0,0])
            vec = o_product_vector / v_product_norm
            o_diff = np.array(log(o_product_norm), 
                              acos(o_product[0]/o_product_norm)*vec)

            diff = sqrt(np.dot(o_diff, o_diff))
            return diff

        else: #<-- Absolute distance
            ostart_minus_o_end = ostart - o_end
            ostart_plus_o_end  = ostart + o_end
            d_minus = sqrt(np.dot(ostart_minus_o_end, ostart_minus_o_end))
            d_plus  = sqrt(np.dot(ostart_plus_o_end, ostart_plus_o_end))
            if (d_minus < d_plus):
                return d_minus
            else:
                return d_plus
