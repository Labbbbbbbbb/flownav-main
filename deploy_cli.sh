#ps 不是脚本，只是把命令写出来方便粘贴
#terminal1
conda activate nomad_deployment
cd /home/zyt/flownav-main/deployment/src/navigation
PYTHONPATH=/home/zyt/flownav-main:/home/zyt/flownav-main/consistency-policy:/home/zyt/flownav-main/py-meanflow:$PYTHONPATH

python navigate_ros1.py --model flownav --dir ../../topomaps/images --ckpt ../../../weights/flownav_weights.pth
#terminal2
conda activate nomad_deployment
cd /home/zyt/flownav-main/deployment/src
PYTHONPATH=/home/zyt/flownav-main:/home/zyt/flownav-main/consistency-policy:/home/zyt/flownav-main/py-meanflow:$PYTHONPATH

python pd_controller.py

#terminal3
cd ~/catkin_ws
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch

#terminal4
cd ~/catkin_ws
source devel/setup.bash
sudo modprobe gs_usb
rosrun scout_bringup bringup_can2usb.bash
roslaunch scout_bringup scout_robot_base.launch 
