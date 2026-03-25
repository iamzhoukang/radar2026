cmake_minimum_required(VERSION 3.8)
project(radar_visualizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 开启 Qt 自动元对象编译魔法 (极其重要)
# 这一步让 CMake 自动处理 Q_OBJECT 宏，生成 moc 文件
# ==========================================
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# ==========================================
# 2. 寻找 ROS 2 与系统依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 寻找 Qt5 核心组件
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

# ==========================================
# 3. 编译可执行文件
# 注意：名字 visualizer_node 必须与 launch 文件中配置的 executable 严格一致
# ==========================================
add_executable(visualizer_node
  src/main.cpp
  src/main_window.cpp
  src/ros_worker.cpp
  include/main_window.hpp   # 强制触发 MOC
  include/ros_worker.hpp    # 强制触发 MOC
)

# ==========================================
# 4. 指定头文件路径
# ==========================================
target_include_directories(visualizer_node PRIVATE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
)

# ==========================================
# 5. 链接依赖库
# ==========================================
ament_target_dependencies(visualizer_node
  rclcpp
  sensor_msgs
  std_srvs
  cv_bridge
)

# 将 Qt5 和 OpenCV 链接到我们的节点上
target_link_libraries(visualizer_node
  Qt5::Core
  Qt5::Gui
  Qt5::Widgets
  ${OpenCV_LIBS}
)

# ==========================================
# 6. 安装规则 (供 ros2 run 和 launch 调用)
# ==========================================
install(TARGETS visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()