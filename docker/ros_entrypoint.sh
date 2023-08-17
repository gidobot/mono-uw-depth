#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/melodic/setup.bash"
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/knfu_slam/Examples/ROS
#exec "$@"
