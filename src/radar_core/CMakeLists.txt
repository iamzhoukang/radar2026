cmake_minimum_required(VERSION 3.10)
project(radar_core)

#加入 -Wno-deprecated-declarations 忽略 TensorRT 弃用警告
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
find_package(CUDA REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Open3D REQUIRED)     #依赖libc++1 libc++abi1

# ==========================================
# 3. 添加 SDK 子目录
# ==========================================
add_subdirectory(src/rb26SDK) 

# ==========================================
# 4. 编译动态链接库 (Component 的核心)
# ==========================================
add_library(camera_one_component SHARED src/camera/cameraone_component.cpp)
add_library(video_component SHARED src/camera/video_component.cpp)

add_library(netdetector_component SHARED 
  src/detector/netdetector_component.cpp
  src/utils/model.cpp
  src/utils/classifier.cpp
  src/utils/cuda_preprocess.cu
)

add_library(solvepnp_component SHARED 
  src/solver/solvepnp_component.cpp
  src/solver/pnp_solver.cpp
  src/solver/calibration_ui.cpp
)

add_library(map_component SHARED src/map/map_component.cpp)

# ==========================================
# 5. 暴露头文件路径 
# ==========================================
target_include_directories(camera_one_component PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS})
target_include_directories(video_component PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS})
target_include_directories(netdetector_component PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS} ${TRT_INC} ${CUDA_INCLUDE_DIRS})
target_include_directories(solvepnp_component PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${OpenCV_INCLUDE_DIRS})

# 【修正】：为 map_component 添加 Open3D 的头文件路径包含
target_include_directories(map_component PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS})

# ==========================================
# 6. 链接 ROS2 目标依赖 (仅放 ament_package)
# ==========================================
ament_target_dependencies(camera_one_component rclcpp rclcpp_components sensor_msgs cv_bridge radar_interfaces)
ament_target_dependencies(video_component rclcpp rclcpp_components sensor_msgs cv_bridge)
ament_target_dependencies(netdetector_component rclcpp rclcpp_components sensor_msgs cv_bridge radar_interfaces)
ament_target_dependencies(solvepnp_component rclcpp rclcpp_components sensor_msgs cv_bridge std_srvs)

# map_component 的 ament 依赖保持不变
ament_target_dependencies(map_component rclcpp rclcpp_components sensor_msgs cv_bridge radar_interfaces std_srvs)

# ==========================================
# 7. 链接第三方底层库 (OpenCV, TensorRT, Open3D 等)
# ==========================================
target_link_libraries(camera_one_component ${OpenCV_LIBS} rb26SDK MvCameraControl dl pthread)
target_link_libraries(video_component ${OpenCV_LIBS} pthread)
target_link_libraries(netdetector_component ${OpenCV_LIBS} ${TRT_LIB} ${CUDA_INCLUDE_DIRS} yaml-cpp::yaml-cpp pthread)
target_link_libraries(solvepnp_component ${OpenCV_LIBS} yaml-cpp::yaml-cpp)


target_link_libraries(map_component ${OpenCV_LIBS} yaml-cpp::yaml-cpp pthread Open3D::Open3D)

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
install(TARGETS camera_one_component video_component netdetector_component solvepnp_component map_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()