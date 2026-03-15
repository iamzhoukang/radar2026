# VisionTargetAngle.msg
# 通用的视觉目标相对夹角输出

std_msgs/Header header

bool is_detected      # 是否在当前画面中检测到了目标 (非常重要，用于状态机切换)

# 以当前相机光心为原点的相对夹角 (弧度)
# 对于广角：这是大角度引导量
# 对于长焦：这是偏离画面中心的小角度误差量
float32 yaw_relative
float32 pitch_relative