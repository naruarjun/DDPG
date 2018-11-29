import numpy as np
from util import info, error, warn

import rospy
from std_msgs.msg import Int8
from stage_ros.srv import ResetPose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan
from rospy import Publisher, Subscriber, ServiceProxy
from tf.transformations import euler_from_quaternion as q2e


class AgentStage(object):
    def __init__(self, _id):
        self._id = _id
        self.reset_agent = ServiceProxy("/reset_positions", ResetPose)
        self.vel_pub = Publisher("/cmd_vel", Twist, queue_size=1)

    def reset(self, pose=None):
        pose = list(pose)
        if len(pose) == 2:
            pose.append(0)
        if pose is not None:
            pose = Pose2D(*pose)
            rospy.wait_for_service('/reset_positions')
        else:
            pose = Pose2D()
        try:
            self.reset_agent(pose)
            self.update()
            self.stall = False
        except (rospy.ServiceException) as e:
            error.out("/reset_positions service call failed")

    def step(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x, vel_cmd.angular.z = action
        self.vel_pub.publish(vel_cmd)
        self.update()

    def update(self):
        odom = rospy.wait_for_message("/base_pose_ground_truth", Odometry)
        scan = rospy.wait_for_message("/base_scan", LaserScan)
        stall = rospy.wait_for_message("/stall", Int8)
        x, y = odom.pose.pose.position.x, odom.pose.pose.position.y
        orin = odom.pose.pose.orientation
        theta = q2e([orin.x, orin.y, orin.z, orin.w])[-1]
        self.stall = bool(stall.data)
        self.pose = np.array([x, y, theta])
        self.scan = np.array(scan.ranges)*scan.intensities/scan.range_max
