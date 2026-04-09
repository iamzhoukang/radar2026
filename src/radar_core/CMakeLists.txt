cmake_minimum_required(VERSION 3.10)
project(radar_core)

# 加入 -Wno-deprecated-declarations 忽略 TensorRT 弃用警告
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)
endif()

# 强制全项目（包含被引入的静态库）生成 -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 开启 CUDA 语言支持 (必须位于寻找包之前)
# ==========================================
enable_language(CUDA)

# ==========================================
# 1. 外部依赖系统路径配置
# ==========================================
include_directories(/opt/MVS/include) 
link_directories(/opt/MVS/lib/64) 

# 严格对齐 TensorRT 的本地绝对路径
find_path(TRT_INC NvInfer.h PATHS /home/lzhros/TensorRT-10.11.0.33/include /usr/local/TensorRT-10.11.0.33/include /usr/include)
find_library(TRT_LIB nvinfer PATHS /home/lzhros/TensorRT-10.11.0.33/lib /usr/local/TensorRT-10.11.0.33/lib /usr/lib/x86_64-linux-gnu)

# ==========================================
# 2. 寻找系统与 ROS2 依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(radar_interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED) # 【新增】追踪器依赖

# ==========================================
# 3. 添加头文件路径
# ==========================================
include_directories(
  include
  src                  # 允许跨文件夹直接引用头文件 (如 "utils/model.hpp")
  src/rb26SDK/include  # 海康大恒相机 SDK 头文件
  ${OpenCV_INCLUDE_DIRS}
  ${TRT_INC}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # CUDA 头文件路径
)

# 编译海康大恒相机的底层驱动 SDK 库
add_subdirectory(src/rb26SDK)

# ==========================================
# 4. 编译工具库 (CUDA预处理, 模型推理)
# ==========================================
add_library(utils_lib
  src/utils/model.cpp
  src/utils/classifier.cpp
  src/utils/cuda_preprocess.cu
)

# ==========================================
# 5. 编译追踪器库 (HKUST Cascade Tracker)
# ==========================================
add_library(tracker_lib
  src/tracker/hungarian.cpp
  src/tracker/kalman.cpp
  src/tracker/cascade_tracker.cpp
)
target_link_libraries(tracker_lib Eigen3::Eigen ${OpenCV_LIBS})

# ==========================================
# 6. 编译各个核心组件 (Component)
# ==========================================
# 相机组件
add_library(camera_one_component SHARED src/camera/cameraone_component.cpp)
ament_target_dependencies(camera_one_component rclcpp sensor_msgs cv_bridge rclcpp_components)

add_library(video_component SHARED src/camera/video_component.cpp)
ament_target_dependencies(video_component rclcpp sensor_msgs cv_bridge rclcpp_components)

# 神经网络检测组件
add_library(netdetector_component SHARED src/detector/netdetector_component.cpp)
ament_target_dependencies(netdetector_component rclcpp sensor_msgs cv_bridge radar_interfaces rclcpp_components)

# PNP解算组件
add_library(solvepnp_component SHARED 
  src/solver/solvepnp_component.cpp
  src/solver/pnp_solver.cpp
  src/solver/calibration_ui.cpp
)
ament_target_dependencies(solvepnp_component rclcpp sensor_msgs cv_bridge radar_interfaces rclcpp_components std_srvs)

# 地图解算组件
add_library(map_component SHARED 
  src/map/map_component.cpp
  src/map/transform.cpp
  src/map/raycaster.cpp
)
ament_target_dependencies(map_component rclcpp sensor_msgs cv_bridge radar_interfaces rclcpp_components std_srvs)

# ==========================================
# 7. 链接具体的系统库
# ==========================================
target_link_libraries(camera_one_component ${OpenCV_LIBS} rb26SDK MvCameraControl dl pthread)

target_link_libraries(video_component ${OpenCV_LIBS} pthread)

# 【修正】为神经网络挂载 utils_lib 依赖
target_link_libraries(netdetector_component utils_lib ${OpenCV_LIBS} ${TRT_LIB} yaml-cpp::yaml-cpp pthread)

target_link_libraries(solvepnp_component ${OpenCV_LIBS} yaml-cpp::yaml-cpp)

# 【修正】为地图挂载 tracker_lib 和 Eigen3 依赖
target_link_libraries(map_component ${OpenCV_LIBS} yaml-cpp::yaml-cpp pthread Open3D::Open3D tracker_lib Eigen3::Eigen)

# ==========================================
# 8. 注册组件 (向 ROS2 容器暴露接口)
# ==========================================
rclcpp_components_register_nodes(camera_one_component "radar_core::CameraOneComponent")
rclcpp_components_register_nodes(video_component "radar_core::VideoComponent")
rclcpp_components_register_nodes(netdetector_component "radar_core::NetDetectorComponent")
rclcpp_components_register_nodes(solvepnp_component "radar_core::SolvePnPComponent")
rclcpp_components_register_nodes(map_component "radar_core::MapComponent")

# ==========================================
# 9. 安装规则 (打包 .so 文件供 Launch 调用)
# ==========================================
install(TARGETS 
  utils_lib 
  tracker_lib 
  camera_one_component 
  video_component 
  netdetector_component 
  solvepnp_component 
  map_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装所有的头文件
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()