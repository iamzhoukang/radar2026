#!/bin/bash

# ==========================================
# Radar Station 一键启动脚本 
# ==========================================

# 1. 绝对路径定位工作空间
WORKSPACE_DIR="$HOME/Code/RadarStation"

echo "[Radar Watchdog] 正在进入工作空间: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR" || { echo "找不到工作空间目录！"; exit 1; }

# 2. 刷新 ROS 2 环境变量
echo "[Radar Watchdog] 正在加载环境变量..."
source install/setup.bash

#优先使用系统 libusb，避免 PCL 被海康 MVS SDK 的旧版 libusb 污染
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 3. 解析启动参数 (默认启动真实相机)
MODE="false" # false 代表启动真实相机
MODE_NAME="真实相机实战模式"

if [ "$1" == "video" ]; then
    MODE="true"
    MODE_NAME="离线视频测试模式"
fi

# ==========================================
# 4. 硬件层：启动 Livox 激光雷达驱动
# ==========================================
echo "⚡ [Radar Watchdog] 正在拉起 Livox 激光雷达硬件驱动..."
# 使用 & 将其放入后台运行，防止阻塞脚本向下执行
ros2 launch livox_ros2_driver livox_lidar_launch.py &
LIVOX_PID=$!

# 注册退出清理陷阱：当脚本收到退出信号(如 Ctrl+C)时，安全关闭后台的雷达驱动
trap "echo -e '\n[Radar Watchdog] 收到退出信号，正在关闭 Livox 雷达驱动...'; kill $LIVOX_PID" EXIT INT TERM

# 给雷达建立 UDP 通信和点云预热预留 3 秒时间
echo "[Radar Watchdog] 等待雷达点云数据流建立 (3秒)..."
sleep 3

# ==========================================
# 5. 算法层：点火启动核心容器
# ==========================================
echo "⚡ [Radar Watchdog] 启动核心算法模式: $MODE_NAME"
echo "---------------------------------------------------"
ros2 launch radar_bringup startall.py use_video:=$MODE