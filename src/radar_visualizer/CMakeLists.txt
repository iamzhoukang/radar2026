cmake_minimum_required(VERSION 3.10)
project(radar_visualizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# ==========================================
# 开启 Qt 的核心编译工具
# ==========================================
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# ==========================================
# 寻找依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

# 寻找 Qt5 库
find_package(Qt5 COMPONENTS Widgets REQUIRED)

# ==========================================
# 编译可执行文件
# ==========================================
add_executable(visualizer_node src/main.cpp)

target_include_directories(visualizer_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(visualizer_node
  rclcpp
  sensor_msgs
  cv_bridge
  std_srvs
)

# 链接 Qt5 和 OpenCV
target_link_libraries(visualizer_node
  Qt5::Widgets
  ${OpenCV_LIBS}
)

# ==========================================
# 安装与打包
# ==========================================
install(TARGETS visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()