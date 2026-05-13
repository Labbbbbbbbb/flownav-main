import numpy as np
import yaml
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4
WAYPOINT_ARRIVAL_THRESHOLD = 0.05  # meters, threshold to consider a waypoint reached

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, queue_size=8, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	data = np.array(waypoint_msg.data)
	# Expect a flattened trajectory: [x1,y1, x2,y2, ...] or [x1,y1,hx1,hy1, x2,y2,hx2,hy2, ...]
	waypoint.set(data)
	print(f"[CALLBACK] 收到整条轨迹 waypoint (扁平数组长度={len(data)}): {data}")
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data


def main():
	global vel_msg, reverse_mode
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("[*] PD Controller 就绪。等待 waypoint 序列...")

	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif waypoint.is_valid(verbose=True):
			data = waypoint.get()
			if data is None:
				vel_out.publish(vel_msg)
				continue
			# parse flattened trajectory into Nx2 or Nx4
			n = len(data)
			if n >= 4 and n % 4 == 0:
				waypoints = data.reshape(-1, 4)
			elif n >= 2 and n % 2 == 0:
				waypoints = data.reshape(-1, 2)
			else:
				print(f"[WARN] 无法解析收到的 waypoint 数据，长度为 {n}")
				vel_out.publish(vel_msg)
				continue
			idx = waypoint.current_waypoint_index
			if idx >= len(waypoints):
				# 已完成整条轨迹
				print("[INFO] 轨迹已完成，清空 waypoint 数据")
				waypoint.pop_head()
				vel_out.publish(vel_msg)
				continue
			current_wp = waypoints[idx]
			# 使用 xy 距离判断到达
			dist = np.linalg.norm(current_wp[:2])
			print(f"Current idx={idx}, waypoint={current_wp}, dist={dist:.3f}")
			if dist < WAYPOINT_ARRIVAL_THRESHOLD:
				# 到达当前 waypoint，前进到下一个
				waypoint.current_waypoint_index += 1
				if waypoint.current_waypoint_index >= len(waypoints):
					print("[INFO] 最后一个 waypoint 已到达，清空 waypoint 数据")
					waypoint.pop_head()
					vel_out.publish(vel_msg)
					continue
				# 否则在下一次循环继续追踪新的 idx
			else:
				v, w = pd_controller(current_wp)
				if reverse_mode:
					v *= -1
				vel_msg.linear.x = v
				vel_msg.angular.z = w
				print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
