cmake_minimum_required(VERSION 3.0.2)
project(yolo_world_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  vision_msgs
  dynamic_reconfigure
  std_msgs
  message_generation
)

generate_dynamic_reconfigure_options(
  cfg/YOLOWorld.cfg
)

add_message_files(
  FILES
  LabelSet.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs dynamic_reconfigure
)

install(PROGRAMS
  nodes/yolo_world_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
