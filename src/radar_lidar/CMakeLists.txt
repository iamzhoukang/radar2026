cmake_minimum_required(VERSION 3.8)
project(radar_lidar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(radar_interfaces REQUIRED)

# 寻找 OpenMP 多线程加速库
find_package(OpenMP REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 添加所有三个节点的源文件
add_library(${PROJECT_NAME} SHARED
  src/localization.cpp
  src/dynamic_cloud.cpp
  src/cluster.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  pcl_ros
  radar_interfaces
)

# 同时链接 PCL 库和 OpenMP 库
target_link_libraries(${PROJECT_NAME} 
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

# ==========================================
# 🌟 核心修复：必须将所有组件放在同一个宏指令中注册
# ==========================================
rclcpp_components_register_nodes(${PROJECT_NAME} 
  "radar_lidar::Localization"
  "radar_lidar::DynamicCloud"
  "radar_lidar::ClusterNode"
)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()