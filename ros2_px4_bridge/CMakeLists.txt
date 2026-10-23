cmake_minimum_required(VERSION 3.8)
project(aeroharvest_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# 1. Disturbance Observer Node (50 Hz C++ Telemetry Observer)
add_executable(disturbance_observer_node src/disturbance_observer_node.cpp)
ament_target_dependencies(disturbance_observer_node
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
)
target_link_libraries(disturbance_observer_node Eigen3::Eigen)

# 2. MPPI Controller Node
add_executable(mppi_controller_node src/mppi_controller_node.cpp)
ament_target_dependencies(mppi_controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
)
target_link_libraries(mppi_controller_node Eigen3::Eigen)

# 3. PX4 Offboard Bridge Node
add_executable(px4_offboard_bridge_node src/px4_offboard_bridge_node.cpp)
ament_target_dependencies(px4_offboard_bridge_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
)

install(TARGETS
  disturbance_observer_node
  mppi_controller_node
  px4_offboard_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
