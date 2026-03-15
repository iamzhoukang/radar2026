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

# 3. 解析启动参数 (默认启动真实相机)
MODE="false" # false 代表启动真实相机
MODE_NAME="真实相机实战模式"

if [ "$1" == "video" ]; then
    MODE="true"
    MODE_NAME="离线视频测试模式"
fi

# 4. 点火启动
echo "⚡ [Radar Watchdog] 启动模式: $MODE_NAME"
echo "---------------------------------------------------"
ros2 launch radar_bringup startall.py use_video:=$MODE

#radar_start video