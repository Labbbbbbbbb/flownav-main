# topic names for ROS communication


# Robot name space to append to all topics
# Change this to match your robot's namespace
# For example, if your robot is named "turtle1", you can set it to "/turtle1"
#ROBOT_NAMESPACE = "/turtle1"    
ROBOT_NAMESPACE = ""   #zyt 

# Image observation topics
#IMAGE_TOPIC = f"{ROBOT_NAMESPACE}/image_compressed"
IMAGE_TOPIC = f"{ROBOT_NAMESPACE}/oakd/rgb/preview/image_raw"  #zyt
# exploration topics
WAYPOINT_TOPIC = f"{ROBOT_NAMESPACE}/waypoint"
REACHED_GOAL_TOPIC = f"{ROBOT_NAMESPACE}/topoplan/reached_goal"
SAMPLED_ACTIONS_TOPIC = f"{ROBOT_NAMESPACE}/sampled_actions"

# move the robot
VEL_TOPIC = f"{ROBOT_NAMESPACE}/cmd_vel"
#VEL_TOPIC = "/diffdrive_controller/cmd_vel_unstamped"  #zyt, for turtlebot4  这俩都可以好像，但是都是刚开始不知道为啥很长一段时间不动，后面突然蹭一下,并且第二个超级慢