cmake_minimum_required(VERSION 3.0.2)
project(yolo_world_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  vision_msgs
)

catkin_package()

install(PROGRAMS
  nodes/yolo_world_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
