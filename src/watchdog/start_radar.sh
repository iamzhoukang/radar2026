#!/bin/bash

WORKSPACE_DIR="$HOME/Code/RadarStation"
cd "$WORKSPACE_DIR" || { echo "找不到工作空间目录！"; exit 1; }

# 【修复】将系统库路径放在 MVS 库之前，避免 PCL 加载错误的 libusb
# PCL 需要 libusb_set_option，而 MVS 自带的 libusb 版本较旧缺少此符号
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

source install/setup.bash

# 解析参数
MODE="false"
if [ "$1" == "video" ]; then
    MODE="true"
fi

# ==========================================
# 1. 安全退出机制
# ==========================================
cleanup() {
    echo -e "\n[Watchdog] 正在安全退出，清理雷达驱动进程..."
    if [ -n "$LIVOX_PID" ]; then
        kill -9 $LIVOX_PID 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# ==========================================
# 2. 启动官方雷达驱动 (不管有没有连接，直接后台强拉)
# ==========================================
echo "[Watchdog] 正在后台拉起 Livox 雷达驱动..."
ros2 launch livox_ros2_driver livox_lidar_launch.py > /dev/null 2>&1 &
LIVOX_PID=$!

sleep 2 # 给雷达驱动留2秒初始化时间

# ==========================================
# 3. 启动核心管线
# ==========================================
echo "[Watchdog] 启动核心管线 (use_video: $MODE)..."
ros2 launch radar_bringup startall.py use_video:=$MODE