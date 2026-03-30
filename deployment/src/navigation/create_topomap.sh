#!/bin/bash

# Create a new tmux session
session_name="turtlebot_$(date +%s)"
tmux new-session -d -s $session_name

tmux set-option -g mouse on

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 0
# Enabling Discovery Server mode creates this file
tmux send-keys "source /etc/turtlebot4/setup.zsh" Enter
# Source venv
tmux send-keys "source ../../../.venv/bin/activate" Enter #zyt  本来是 .venv/bin/activate 
tmux send-keys "export PYTHONPATH=/media/zhanyt/Data/Zytisworking/flownav-main/deployment/src:$PYTHONPATH" Enter
tmux send-keys "python create_topomap.py --dt 1 --dir $1" Enter

tmux select-pane -t 1
# Enabling Discovery Server mode creates this file
tmux send-keys "source /etc/turtlebot4/setup.zsh" Enter
# Source venv
tmux send-keys "source ../../../.venv/bin/activate" Enter #zyt
tmux send-keys "export PYTHONPATH=/media/zhanyt/Data/Zytisworking/flownav-main/deployment/src:$PYTHONPATH" Enter
tmux send-keys "ros2 run teleop_twist_keyboard teleop_twist_keyboard" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
