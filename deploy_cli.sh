#ps 不是脚本，只是把命令写出来方便粘贴
#terminal1
conda activate nomad_deployment
cd /home/zyt/flownav-main/deployment/src/navigation
PYTHONPATH=/home/zyt/flownav-main:/home/zyt/flownav-main/consistency-policy:/home/zyt/flownav-main/py-meanflow:$PYTHONPATH

python navigate_ros1.py --model flownav --dir ../../topomaps/images --ckpt /home/zyt/flownav-main/meanflownav_2026_04_15_20_23_25/latest.pth

#terminal2
conda activate nomad_deployment
cd /home/zyt/flownav-main/deployment/src
PYTHONPATH=/home/zyt/flownav-main:/home/zyt/flownav-main/consistency-policy:/home/zyt/flownav-main/py-meanflow:$PYTHONPATH

python pd_controller.py

#terminal3
cd ~/catkin_ws
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch color_width:=640 color_height:=480 color_fps:=15
roslaunch realsense2_camera rs_camera.launch

#
#terminal4
cd ~/catkin_ws
source devel/setup.bash
sudo modprobe gs_usb
rosrun scout_bringup bringup_can2usb.bash
roslaunch scout_bringup scout_robot_base.launch 

# new vechile
# 关闭接口
sudo ip link set can0 down
# 设置波特率为 1M
sudo ip link set can0 up type can bitrate 1000000
# 使能接口：
sudo modprobe peak_usb
# 检查状态
ifconfig can0
roscore #if not already running
cd /home/zyt/MMChassis/MASMM-Mobile-Manipulator-Chassis-main
python3  chassis_can_ros1.py

#another terminal
rosservice call /chassis/enable "data: true"
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

#ouster
#clear proxy settings
export ROS_IP=169.254.220.200                       
export ROS_MASTER_URI=http://localhost:11311
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export NO_PROXY=localhost,127.0.0.1,169.254.220.149,169.254.220.200
cd /home/zyt/Ouster_DLIO/ouster
source devel/setup.zsh
roslaunch ouster_ros sensor.launch
#Dlio
cd /home/zyt/Ouster_DLIO/lidar
source devel/setup.zsh
roslaunch direct_lidar_inertial_odometry dlio.launch
