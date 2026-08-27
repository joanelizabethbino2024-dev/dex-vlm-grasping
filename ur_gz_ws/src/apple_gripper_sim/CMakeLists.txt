cmake_minimum_required(VERSION 3.8)
project(apple_gripper_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  models
  worlds
  launch
  scripts
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
