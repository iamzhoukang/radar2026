cmake_minimum_required(VERSION 3.10)
project(radar_lidar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------------------------------------------------
# 1. 寻找依赖 
# -------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(radar_interfaces REQUIRED)

# 寻找系统库 PCL 和 Eigen
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# -------------------------------------------------------------------
# 2. 包含目录
# -------------------------------------------------------------------
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

# -------------------------------------------------------------------
# 3. 注册 Localization 节点 (定位) 
# -------------------------------------------------------------------
add_library(localization_node SHARED
  src/localization.cpp
)
target_link_libraries(localization_node
  ${PCL_LIBRARIES}
  -Wl,--no-as-needed -lusb-1.0 -Wl,--as-needed  
)
ament_target_dependencies(localization_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
  pcl_ros
)
# 注册组件插件
rclcpp_components_register_node(localization_node
  PLUGIN "radar_lidar::Localization"
  EXECUTABLE localization_node_exe
)

# -------------------------------------------------------------------
# 4. 注册 DynamicCloud 节点 (背景减除)
# -------------------------------------------------------------------
add_library(dynamic_cloud_node SHARED
  src/dynamic_cloud.cpp
)
target_link_libraries(dynamic_cloud_node
  ${PCL_LIBRARIES}
  tf2_eigen::tf2_eigen
  -Wl,--no-as-needed -lusb-1.0 -Wl,--as-needed
)
ament_target_dependencies(dynamic_cloud_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  pcl_ros
)
# 注册组件插件
rclcpp_components_register_node(dynamic_cloud_node
  PLUGIN "radar_lidar::DynamicCloud"
  EXECUTABLE dynamic_cloud_node_exe
)

# -------------------------------------------------------------------
# 5. 注册 Cluster 节点 (欧几里得聚类)
# -------------------------------------------------------------------
add_library(cluster_node SHARED
  src/cluster.cpp
)
target_link_libraries(cluster_node
  ${PCL_LIBRARIES}
)
ament_target_dependencies(cluster_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  pcl_conversions
  pcl_ros
  radar_interfaces
)
# 注册组件插件
rclcpp_components_register_node(cluster_node
  PLUGIN "radar_lidar::ClusterNode"
  EXECUTABLE cluster_node_exe
)

# -------------------------------------------------------------------
# 6. 安装与打包
# -------------------------------------------------------------------
install(TARGETS
  localization_node
  dynamic_cloud_node
  cluster_node
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

ament_package()