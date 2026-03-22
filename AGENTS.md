# RadarStation - 雷达站视觉系统

## Project Overview

RadarStation 是一个基于 ROS 2 Humble 的机器人雷达站视觉系统，专为 RoboMaster 机甲大师赛设计。系统通过高分辨率工业相机采集图像，利用深度学习进行目标检测，并通过 PnP 解算和 3D 射线碰撞检测实现战场目标定位与地图映射。

核心功能包括：
- 高分辨率工业相机（海康威视 CS200）图像采集
- 基于 TensorRT 的双级神经网络检测（机器人检测 + 装甲板识别 + 数字分类）
- 单帧 PnP 标定与解算
- Open3D 物理射线碰撞检测（支持场地高低差地形）
- Qt5 可视化客户端
- 零拷贝（Zero-Copy）进程内通信优化

## Technology Stack

- **Framework**: ROS 2 Humble (rclcpp)
- **Language**: C++17
- **CUDA**: GPU 加速预处理与神经网络推理
- **TensorRT 10.11.0**: 神经网络推理引擎
- **OpenCV 4.x**: 图像处理
- **Open3D**: 3D 射线碰撞检测与场地网格加载
- **Qt5**: 可视化界面
- **YAML-CPP**: 配置文件解析
- **Camera SDK**: 海康威视 MVS SDK (工业相机驱动)

## Project Structure

```
RadarStation/
├── src/
│   ├── radar_interfaces/      # 自定义消息接口定义
│   │   ├── msg/DetectResult.msg      # 单个检测结果
│   │   ├── msg/DetectResults.msg     # 检测结果数组
│   │   ├── msg/RadarMap.msg          # 小地图官方数据格式
│   │   ├── msg/GimbalCmd.msg         # 云台控制指令
│   │   └── msg/VisionTargetAngle.msg # 视觉目标角度
│   │
│   ├── radar_core/            # 核心功能组件（主要开发包）
│   │   ├── src/camera/               # 图像采集组件
│   │   │   ├── cameraone_component.cpp   # 海康相机组件
│   │   │   └── video_component.cpp       # 视频离线测试组件
│   │   ├── src/detector/             # 神经网络检测组件
│   │   │   └── netdetector_component.cpp # YOLO 检测器
│   │   ├── src/solver/               # PnP 解算组件
│   │   │   └── solvepnp_component.cpp    # 单帧标定与解算
│   │   ├── src/map/                  # 地图映射组件
│   │   │   └── map_component.cpp         # 3D 射线投影与小地图生成
│   │   ├── src/utils/                # 工具类
│   │   │   ├── model.hpp/cpp           # TensorRT YOLO 推理类
│   │   │   ├── classifier.hpp/cpp      # TensorRT 数字分类器
│   │   │   ├── cuda_preprocess.cu/.cuh # CUDA 预处理核函数
│   │   │   └── ...
│   │   └── src/rb26SDK/              # 相机 SDK 封装库
│   │       ├── include/                # SDK 头文件（海康/大恒）
│   │       └── src/                    # SDK 实现
│   │
│   ├── radar_visualizer/      # Qt5 可视化客户端
│   │   └── src/main.cpp              # 视频流与小地图显示
│   │
│   ├── radar_serial/          # 串口通信包（预留）
│   │
│   ├── radar_bringup/         # 系统启动包
│   │   └── launch/startall.py        # 系统启动 Launch 文件
│   │
│   └── watchdog/
│       └── start_radar.sh            # 一键启动脚本
│
├── config/                    # 配置文件目录
│   ├── camera/hik_cs200.yaml         # 相机参数配置
│   ├── detector/yolo.yaml            # 神经网络模型路径配置
│   ├── solver/cs200_calibration.yaml # 相机标定参数（含 rvec/tvec）
│   ├── solver/keypoint_6.txt         # 6 点标定世界坐标
│   └── map/                          # 小地图相关配置
│
├── model/                     # 神经网络模型
│   └── engine/                       # TensorRT Engine 文件
│       ├── robot_only.engine         # 机器人检测模型
│       ├── armor_little.engine       # 装甲板检测模型
│       ├── armor_classifier.engine   # 数字分类模型
│       └── plane_best.engine         # _plane 检测模型（备用）
│
├── video/                     # 离线测试视频
├── build/                     # 构建目录（colcon）
└── install/                   # 安装目录
```

## Package Details

### 1. radar_interfaces
- **类型**: 接口定义包
- **功能**: 定义 ROS 2 消息格式
- **构建依赖**: rosidl_default_generators
- **关键消息**:
  - `DetectResult`: 目标编号 (B1, R3 等) + 像素坐标
  - `DetectResults`: 带 Header 的检测结果数组
  - `RadarMap`: 官方比赛要求的小地图坐标格式

### 2. radar_core
- **类型**: 主功能包（核心）
- **架构**: ROS 2 Component 插件化架构，支持零拷贝
- **组件列表**:

| 组件名 | 插件类名 | 功能描述 |
|--------|----------|----------|
| camera_one_component | `radar_core::CameraOneComponent` | 海康相机图像采集 |
| video_component | `radar_core::VideoComponent` | 离线视频测试 |
| netdetector_component | `radar_core::NetDetectorComponent` | YOLO 神经网络检测 |
| solvepnp_component | `radar_core::SolvePnPComponent` | PnP 标定与解算 |
| map_component | `radar_core::MapComponent` | 3D 投影与小地图生成 |

- **外部依赖**:
  - TensorRT 10.11.0 (路径: `/home/lzhros/TensorRT-10.11.0.33`)
  - 海康威视 MVS SDK (路径: `/opt/MVS`)
  - Open3D (依赖 libc++1 libc++abi1)

### 3. radar_visualizer
- **类型**: Qt5 GUI 应用
- **功能**: 
  - 显示处理后的视频流（processed_video）
  - 显示小地图（map/image）
  - 触发标定服务（solvepnp/start）
  - 一键翻转阵营视角（180° 旋转）
- **通信方式**: ROS 2 话题订阅 + 服务调用

### 4. radar_bringup
- **类型**: 启动包
- **Launch 文件**: `startall.py`
  - 使用 `ComposableNodeContainer` 实现零拷贝进程内通信
  - 支持两种模式：`use_video:=true/false`
  - 独立启动 Qt 可视化节点

### 5. radar_serial
- **类型**: 预留包
- **状态**: 当前无源码，仅框架

## Build Instructions

### Prerequisites

1. **ROS 2 Humble** (完整桌面版)
2. **CUDA Toolkit** (与 TensorRT 兼容版本)
3. **TensorRT 10.11.0** 安装于 `/home/lzhros/TensorRT-10.11.0.33`
4. **海康威视 MVS SDK** 安装于 `/opt/MVS`
5. **Open3D** (系统包或源码编译)
6. **依赖安装**:
   ```bash
   sudo apt install libyaml-cpp-dev libc++1 libc++abi1
   ```

### Build Commands

```bash
# 进入工作空间
cd ~/Code/RadarStation

# 清理旧构建（可选）
rm -rf build install log

# 完整构建
colcon build --symlink-install

# 仅构建指定包
colcon build --packages-select radar_core

# 构建并输出详细日志
colcon build --symlink-install --event-handlers console_direct+
```

### Sourcing

```bash
# 每次新终端必须执行
source ~/Code/RadarStation/install/setup.bash
```

## Run Instructions

### Method 1: 使用启动脚本（推荐）

```bash
# 真实相机模式（实战）
./src/watchdog/start_radar.sh

# 离线视频测试模式
./src/watchdog/start_radar.sh video
```

### Method 2: 使用 ros2 launch

```bash
# 加载环境
source install/setup.bash

# 真实相机模式
ros2 launch radar_bringup startall.py use_video:=false

# 离线视频模式
ros2 launch radar_bringup startall.py use_video:=true
```

### Method 3: 单独启动组件（调试）

```bash
# 启动容器
ros2 run rclcpp_components component_container

# 加载组件（在另一个终端）
ros2 component load /radar_vision_container radar_core radar_core::CameraOneComponent
ros2 component load /radar_vision_container radar_core radar_core::NetDetectorComponent
```

## Runtime Architecture

### 数据流图

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Camera/Video   │────▶│  NetDetector     │────▶│  SolvePnP        │
│  Component      │     │  (YOLO+TRT)      │     │  (PnP Solver)    │
│  (cs200_topic)  │     │  (detector/      │     │  (calibration)   │
│                 │     │   results)       │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │  processed_video │     │  map_component   │
                        │  (可视化用)       │     │  (3D射线投影)     │
                        └──────────────────┘     └────────┬─────────┘
                               │                          │
                               ▼                          ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │  Qt Visualizer   │     │  map/image       │
                        │  (视频显示)       │     │  map/official_   │
                        │                  │     │  data            │
                        └──────────────────┘     └──────────────────┘
```

### 通信机制

1. **零拷贝通信**: 使用 `rclcpp_components` + `use_intra_process_comms: True`
   - 核心组件（相机/检测/PnP/地图）运行在同一进程内
   - 通过 `std::unique_ptr` 转移图像所有权，避免内存拷贝

2. **话题列表**:
   - `cs200_topic`: 原始图像（sensor_msgs/Image）
   - `processed_video`: 画框后的视频（用于显示）
   - `detector/results`: 检测结果（DetectResults）
   - `map/image`: 小地图图像
   - `map/official_data`: 官方格式地图数据（RadarMap）

3. **服务列表**:
   - `solvepnp/start`: 触发标定流程（Trigger）
   - `map/reload_config`: 热重载标定参数（Trigger）

## Configuration System

### 主要配置文件

| 文件路径 | 用途 |
|----------|------|
| `config/detector/yolo.yaml` | 神经网络模型路径、输入尺寸、置信度阈值 |
| `config/camera/hik_cs200.yaml` | 相机序列号、曝光时间、增益 |
| `config/solver/cs200_calibration.yaml` | 相机内参 (K, D)、外参 (rvec, tvec) |
| `config/solver/keypoint_6.txt` | 6 个标定点的世界坐标 |
| `config/map/field_image.yaml` | 场地物理尺寸、小地图分辨率 |

### 标定流程

1. 启动系统后，点击 Qt 界面 "🎯 触发标定" 按钮
2. 在弹出的 "Calibration Tool" 窗口中：
   - 左键点击 → 进入 600x600 放大模式
   - 放大模式中左键 → 确认选点
   - 右键 → 撤销上一点
   - 's' → 保存标定结果
   - 'c' → 清空所有点
3. 选中 6 个点后按 's'，自动计算并保存到 YAML

## Code Style Guidelines

### 命名规范
- **命名空间**: `radar_core`, `radar_interfaces`, `sdk`
- **类名**: `CamelCase` (如 `CameraOneComponent`, `NetDetectorComponent`)
- **变量名**: `snake_case_` (成员变量带下划线后缀)
- **函数名**: `camelCase` (如 `captureLoop`, `imageCallback`)
- **宏定义**: `SCREAMING_SNAKE_CASE`
- **文件命名**: 全小写 + 下划线 (如 `solvepnp_component.cpp`)

### 代码注释规范
- 主要使用中文注释
- 组件关键逻辑需标注 "【零拷贝构造】"、"【极致丝滑逻辑】" 等标识
- 复杂算法需说明物理意义（如 "物理射线碰撞检测"）

### 组件开发规范
1. 必须继承 `rclcpp::Node`
2. 构造函数必须使用 `const rclcpp::NodeOptions & options`
3. 必须在文件末尾注册组件：
   ```cpp
   #include "rclcpp_components/register_node_macro.hpp"
   RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::ComponentName)
   ```

## Testing Strategy

### 离线测试
- 使用 `video_component` 加载录制视频
- 通过 `use_video:=true` 参数切换
- 视频文件路径在 Launch 文件中配置

### 性能监控
- `NetDetectorComponent` 内置 FPS 和延迟统计
- 日志格式: `[Perf] 吞吐率: XX.XX FPS | 神经网络端到端延迟: XX.XX ms`

### 调试工具
```bash
# 查看话题列表
ros2 topic list

# 查看检测结果的频率
ros2 topic hz /detector/results

# 可视化图像话题
ros2 run rqt_image_view rqt_image_view

# 触发标定服务
ros2 service call /solvepnp/start std_srvs/srv/Trigger
```

## Important Notes

### TensorRT 路径
CMake 中通过 `find_path` 和 `find_library` 查找 TensorRT，搜索顺序：
1. `/home/lzhros/TensorRT-10.11.0.33`
2. `/usr/local/TensorRT-10.11.0.33`
3. `/usr/include`

### 相机 SDK 路径
海康 MVS SDK 默认路径 `/opt/MVS`，如安装位置不同需修改：
- `src/radar_core/CMakeLists.txt` 第 20-21 行
- `src/radar_core/src/rb26SDK/CMakeLists.txt` 第 21 行

### Open3D 依赖
运行时需要 `libc++1` 和 `libc++abi1`：
```bash
sudo apt install libc++1 libc++abi1
```

### 绝对路径依赖
多处配置使用绝对路径 `/home/lzhros/Code/RadarStation`，迁移项目时需批量替换。

## License

TODO: 待添加开源许可证声明
