# # topic names for ROS communication


# # Robot name space to append to all topics
# # Change this to match your robot's namespace
# # For example, if your robot is named "turtle1", you can set it to "/turtle1"
# ROBOT_NAMESPACE = "/turtle1"    

# # Image observation topics
# IMAGE_TOPIC = f"{ROBOT_NAMESPACE}/image_compressed"

# # exploration topics
# WAYPOINT_TOPIC = f"{ROBOT_NAMESPACE}/waypoint"
# REACHED_GOAL_TOPIC = f"{ROBOT_NAMESPACE}/topoplan/reached_goal"
# SAMPLED_ACTIONS_TOPIC = f"{ROBOT_NAMESPACE}/sampled_actions"

# # move the robot
# VEL_TOPIC = f"{ROBOT_NAMESPACE}/cmd_vel"

# topic names for ROS 1 communication (MeanFlow on Scout Robot)

# 图像观测话题 (来自你的 Realsense)
IMAGE_TOPIC = "/camera/color/image_raw"

# 导航相关话题
WAYPOINT_TOPIC = "/waypoint"
FITTED_WAYPOINT_TOPIC = "/fitted_waypoint"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"

# 控制话题 (Scout 机器人底盘接收的话题)
# 注意：请确认你的 Scout 底盘驱动监听的是 /cmd_vel 还是 /scout/cmd_vel
VEL_TOPIC = "/cmd_vel" 

# 其他辅助话题 (保持与 NoMaD 一致，方便调试)
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"