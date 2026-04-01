# LidarCluster.msg
geometry_msgs/Point center    # 质心坐标 (x, y, z)
float32 cluster_size         # 点云簇的大小（用于辅助判定是否为无人机）
int32 id                     # 簇 ID（用于后续卡尔曼追踪）