# GimbalCmd.msg
# 下发给云台电控的最终指令

std_msgs/Header header

# 追踪状态机 (让电控知道当前处于什么模式)
uint8 MODE_SEARCH = 0      # 搜索模式：两者都没看到，云台可以按预设轨迹巡逻
uint8 MODE_COARSE = 1      # 粗瞄模式：听从广角引导 (大开大合)
uint8 MODE_FINE = 2        # 精瞄模式：听从长焦闭环 (微调锁定)
uint8 current_mode

# 发送给电控的目标角度
# 如果是 MODE_COARSE，这里是广角给出的绝对目标位姿
# 如果是 MODE_FINE，这里是结合长焦误差算出的高频微调量，或是让电控做直接的 PID 闭环误差输入
float32 yaw_cmd
float32 pitch_cmd