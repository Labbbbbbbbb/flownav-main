import numpy as np
import yaml
from typing import Tuple
from scipy.interpolate import splprep, splev

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC,
       					FITTED_WAYPOINT_TOPIC)
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
WAYPOINT_TIMEOUT =1.2# seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4
WAYPOINT_ARRIVAL_THRESHOLD = 0.05  # meters, threshold to consider a waypoint reached
BSPLINE_SAMPLE_MULTIPLIER = 4
BASE_WAYPOINT_INTERVAL_MS = 150.0

# GLOBALS
vel_msg = Twist()
reached_goal = False
reverse_mode = False
current_yaw = None
waypoint = None

def get_delta(actions: np.ndarray) -> np.ndarray:
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def fit_bspline_waypoints(waypoints: np.ndarray, sample_multiplier: int = BSPLINE_SAMPLE_MULTIPLIER) -> np.ndarray:
	"""Fit a B-spline through waypoints and resample a denser trajectory."""
	pts = np.asarray(waypoints, dtype=np.float32)
	if pts.ndim != 2 or pts.shape[0] < 3:
		return pts

	num_samples = max(int(pts.shape[0] * sample_multiplier), pts.shape[0])
	num_samples = max(num_samples, 2)
	curve_dim = pts.shape[1]
	k = min(3, pts.shape[0] - 1)

	try:
		diffs = np.diff(pts[:, :2], axis=0)
		distances = np.linalg.norm(diffs, axis=1)
		u = np.concatenate(([0.0], np.cumsum(distances)))
		if u[-1] < EPS:
			return np.repeat(pts[:1], num_samples, axis=0)
		u = u / u[-1]

		coords = [pts[:, dim] for dim in range(curve_dim)]
		tck, _ = splprep(coords, u=u, s=0.0, k=k)
		sample_u = np.linspace(0.0, 1.0, num_samples)
		sampled = np.asarray(splev(sample_u, tck), dtype=np.float32).T
		return sampled
	except Exception:
		sample_u = np.linspace(0.0, 1.0, num_samples)
		t_src = np.linspace(0.0, 1.0, pts.shape[0])
		resampled = np.column_stack([
			np.interp(sample_u, t_src, pts[:, dim]) for dim in range(curve_dim)
		]).astype(np.float32)
		return resampled


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
	if waypoint is not None:
		waypoint.set(data)
	print(f"[CALLBACK] Received waypoint message with data length {len(data)}: {data}")
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data


def main():
	global vel_msg, reverse_mode,waypoint
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint = ROSData(WAYPOINT_TIMEOUT, queue_size=8, name="waypoint")

	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	fitted_waypoint_pub = rospy.Publisher(FITTED_WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
	rate = rospy.Rate(RATE)

	print("[*] PD Controller 就绪。等待 waypoint 序列...")

	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif waypoint.is_valid(verbose=True):
			data = waypoint.get()		#data是一个列表，包含了queue_size个waypoint点，每个点是一个numpy数组，形状为(2,)或(4,)
			if data is None:
				vel_out.publish(vel_msg)
				waypoints=np.zeros((8, 2))
				continue
			# parse flattened trajectory into Nx2 or Nx4
			n = len(data)
			print(f"Received waypoint data length: {n}, data: {data}")
			if n >= 2 and n % 2 == 0:
				waypoints = np.array(data).reshape(-1, 2)
			else:
				print(f"[WARN] 无法解析收到的 waypoint 数据，长度为 {n}")
				waypoints=np.zeros((8, 2))
				vel_out.publish(vel_msg)
				continue
			
			# with no B-spline, directly compute delta_waypoint from raw waypoints
			# delta_waypoint=get_delta(np.asarray(waypoints).reshape(1, -1, waypoints.shape[-1])).squeeze()	# 计算相邻waypoint之间的差值，得到每个waypoint相对于前一个waypoint的增量
			# cur_waypoint_elasped_time = (rospy.get_time() - waypoint.last_time_received ) * 1000.0	#单位ms
			# waypoint.current_waypoint_index= int(int(cur_waypoint_elasped_time //150))	#每200ms切换到下一个waypoint，待调整

			
			# B-spline
			smoothed_waypoints = fit_bspline_waypoints(waypoints)
			delta_waypoint = np.diff(smoothed_waypoints, axis=0, prepend=smoothed_waypoints[:1])
			cur_waypoint_elasped_time = (rospy.get_time() - waypoint.last_time_received ) * 1000.0	#单位ms
			sample_interval_ms = BASE_WAYPOINT_INTERVAL_MS * len(waypoints) / float(len(smoothed_waypoints))
			waypoint.current_waypoint_index= int(cur_waypoint_elasped_time // sample_interval_ms)	# 按样条密采样后的时间步长切换
   			
   
			print("[DEBUG] cur_waypoint_elasped_time: {:.2f}ms, current_waypoint_index: {}".format(cur_waypoint_elasped_time, waypoint.current_waypoint_index))
			waypoint.current_waypoint_index=min(waypoint.current_waypoint_index, len(delta_waypoint) - 1)
			current_wp = delta_waypoint[waypoint.current_waypoint_index]

			print(f"Current idx={waypoint.current_waypoint_index},elasped={cur_waypoint_elasped_time:.2f}ms , current_wp={current_wp}")
			
			v, w = pd_controller(current_wp)
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v
			vel_msg.angular.z = w
			print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		waypoint_msg = Float32MultiArray()
		# waypoint_msg.data = waypoints.astype(np.float32).reshape(-1).tolist()
		waypoint_msg.data = smoothed_waypoints.astype(np.float32).reshape(-1).tolist() # Bspline
		fitted_waypoint_pub.publish(waypoint_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
