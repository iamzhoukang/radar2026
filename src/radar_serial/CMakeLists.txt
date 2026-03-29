cmake_minimum_required(VERSION 3.10)
project(radar_serial)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 寻找系统与 ROS2 依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(radar_interfaces REQUIRED)

# ==========================================
# 2. 编译可执行文件节点
# ==========================================
add_executable(serial_node src/serial_node.cpp)

# ==========================================
# 3. 暴露头文件路径
# ==========================================
# 这里直接将 include/ 目录暴露给编译器
target_include_directories(serial_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# ==========================================
# 4. 链接 ROS2 目标依赖
# ==========================================
ament_target_dependencies(serial_node
  rclcpp
  rclcpp_components
  radar_interfaces
)

# ==========================================
# 5. 安装规则 (打包供 Launch 调用)
# ==========================================
install(TARGETS serial_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()