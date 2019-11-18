#!/usr/bin/env python
import rospy
from gym_link.robot_gazebo_env import RobotGazeboEnv


class ROSRobotEnv(RobotGazeboEnv):
    def __init__(self):
        # list of controllers
        self.controllers_list = []
        
        # robot namespace
        self.robot_name_space = ''

        # launch connection to gazebo
        super(ROSRobotEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=False,
            start_init_physics_parameters=True,
            reset_world_or_sim='WORLD')

        self.gazebo.unpauseSim()
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()
        self._check_all_systems_ready()
        self._setup_services()
        self.gazebo.pauseSim()

    def _check_all_systems_ready(self):
        """
        Checks that all the subscribers, publishers, services and other simulation systems are
        operational.
        """
        self._check_all_subscribers_ready()
        self._check_all_publishers_ready()
        self._check_all_services_ready()
        return True
    
    def _check_all_subscribers_ready(self):
        """
        Checks that all the subscribers are ready for connection
        """
        raise NotImplementedError()

    def _check_all_publishers_ready(self):
        """
        Checks that all the sensors are ready for connection
        """
        raise NotImplementedError()
    
    def _setup_subscribers(self):
        """
        Sets up all the subscribers relating to robot state
        """
        raise NotImplementedError()

    def _setup_publishers(self):
        """
        Sets up all the publishers relating to robot state
        """
        raise NotImplementedError()

    def _check_subscriber_ready(self, name, type, timeout=5.0):
        """
        Waits for a sensor topic to get ready for connection
        """
        var = None        
        while var is None and not rospy.is_shutdown():
            try:
                var = rospy.wait_for_message(name, type, timeout)
            except:
                rospy.logfatal('Sensor topic "%s" is not available. Waiting...', name)
        return var

    def _check_publisher_ready(self, name, obj, timeout=5.0):
        """
        Waits for a publisher to get response
        """
        start_time = rospy.Time.now()
        while obj.get_num_connections() == 0 and not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() >= timeout:
                rospy.logfatal('No subscriber found for the publisher %s. Exiting...', name)
    
    def _check_service_ready(self, name, timeout=5.0):
        """
        Waits for a service to get ready
        """
        try:
            rospy.wait_for_service(name, timeout)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logfatal("Service %s unavailable.", name)