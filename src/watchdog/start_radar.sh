#!/bin/bash

WORKSPACE_DIR="$HOME/Code/RadarStation"
cd "$WORKSPACE_DIR" || { echo "找不到工作空间目录！"; exit 1; }

# 【系统库路径修复】将系统库路径放在 MVS 库之前，避免 PCL 加载错误的 libusb
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 加载 ROS 2 雷达工作空间环境
source install/setup.bash

# 解析参数 (支持传参开启视频模式)
MODE="false"
if [ "$1" == "video" ]; then
    MODE="true"
fi

# ==========================================
# 1. 联合安全退出机制 (Ctrl+C 拦截)
# ==========================================
cleanup() {
    echo -e "\n[Watchdog] 收到退出信号，正在安全清理所有进程..."
    
    # 杀掉自瞄进程
    if [ -n "$AIM_PID" ]; then
        echo "[Watchdog] 正在关闭自瞄系统..."
        kill -9 $AIM_PID 2>/dev/null
    fi

    # 杀掉雷达驱动进程
    if [ -n "$LIVOX_PID" ]; then
        echo "[Watchdog] 正在关闭 Livox 雷达驱动..."
        kill -9 $LIVOX_PID 2>/dev/null
    fi
    
    echo "[Watchdog] 清理完毕，安全退出。"
    exit 0
}
# 捕获 Ctrl+C (SIGINT) 和 kill 信号 (SIGTERM)
trap cleanup SIGINT SIGTERM

# ==========================================
# 2. 启动自瞄系统 (后台运行)
# ==========================================
echo "[Watchdog] 正在后台拉起自瞄系统..."
# 先切入到 build 文件夹内部，保证 ../configs 这种相对路径能被正确解析
pushd /home/lzhros/Code/sp_vision_25_rbclone/build > /dev/null

# 直接运行当前目录下的可执行文件
./rb_auto_drone_debug &
AIM_PID=$!

# 启动完后迅速退回雷达工作空间
popd > /dev/null

# ==========================================
# 3. 启动官方雷达驱动 (后台运行)
# ==========================================
echo "[Watchdog] 正在后台拉起 Livox 雷达驱动..."
ros2 launch livox_ros2_driver livox_lidar_launch.py > /dev/null 2>&1 &
LIVOX_PID=$!

# 给自瞄系统加载模型和雷达驱动初始化留出 2 秒缓冲时间
sleep 2 

# ==========================================
# 4. 启动核心管线 (前台阻断运行)
# ==========================================
echo "[Watchdog] 启动核心管线 (use_video: $MODE)..."
ros2 launch radar_bringup startall.py use_video:=$MODE