# DetectResult.msg
string number        # 目标编号 (如 "B1", "R3", "Drone")
int32 class_id       # 类别ID (0-9是兵种，-1是未知装甲板，-2是无人机)
float32 class_conf   # 分类置信度 (数字分类器的概率)
float32 x            # 框中心像素坐标 x
float32 y            # 框中心像素坐标 y
float32 width        # 框宽
float32 height       # 框高
float32 car_conf     # 车辆检测置信度 (YOLO检测的概率)
int32 bot_id         # 底层追踪ID (目前预留，填-1)