#include "localization.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2/LinearMath/Quaternion.h>

namespace radar_lidar {

Localization::Localization(const rclcpp::NodeOptions &options)
: Node("localization_node", options) {
    
    // 获取参数
    std::string pcd_path = this->declare_parameter("map_pcd_path", "/home/lzhros/Code/RadarStation/config/lidar/RB2026_rmuc.pcd");
    
    // 极其重要的测试开关：实验室模式下跳过 GICP 匹配，直接强行下发原点 TF
    bool lab_mode = this->declare_parameter("lab_mode", true); 

    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // 加载地图
    if (loadMap(pcd_path)) {
        is_initialized_ = true;
    } else {
        RCLCPP_ERROR(this->get_logger(), "定位地图加载失败，请检查路径: %s", pcd_path.c_str());
    }

    if (lab_mode) {
        current_pose_ = Eigen::Matrix4f::Identity(); // 位姿矩阵设为单位矩阵 (无旋转，无平移)
        has_aligned_ = true;
        RCLCPP_WARN(this->get_logger(), "!!! [实验室模式] 已开启：强制锁定原点 TF,跳过 GICP 对齐 !!!");
    } else {
        init_guess_ = Eigen::Matrix4f::Identity(); 
    }

    sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10, std::bind(&Localization::lidarCallback, this, std::placeholders::_1));
}

bool Localization::loadMap(const std::string & map_path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *raw_map) == -1) return false;

    // 对庞大的 PCD 地图进行体素滤波，这一步能让 GICP 的匹配速度提升 10 倍！
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel_filter.setInputCloud(raw_map);
    voxel_filter.filter(*map_cloud_);
    return true;
}

void Localization::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!is_initialized_) return;

    // 赛场上雷达是固定的，如果已经对齐成功，只需高频广播 TF，直接跳过计算
    if (has_aligned_) {
        broadcastTransform(current_pose_);
        return;
    }

    // 将 ROS 消息转换为 PCL 格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    // 必须降采样当前帧，否则 10 万个点跑 GICP 会导致系统卡死
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel_filter.setInputCloud(cloud);
    voxel_filter.filter(*downsampled_cloud);

    // GICP 匹配核心逻辑
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(downsampled_cloud);
    gicp.setInputTarget(map_cloud_);
    gicp.setMaxCorrespondenceDistance(1.0);
    gicp.setMaximumIterations(50);

    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    gicp.align(aligned_cloud, init_guess_);

    // 评估对齐结果
    if (gicp.hasConverged() && gicp.getFitnessScore() < 0.25) {
        current_pose_ = gicp.getFinalTransformation();
        has_aligned_ = true; // 锁定状态
        RCLCPP_INFO(this->get_logger(), ">>> 定位成功！初始位姿已锁定, GICP 得分: %f", gicp.getFitnessScore());
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            "等待对齐中... 当前匹配得分: %f (需要 < 0.25)", gicp.getFitnessScore());
    }
}

void Localization::broadcastTransform(const Eigen::Matrix4f & transform) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = "map";           // 世界坐标系
    t.child_frame_id = "livox_frame";    // 雷达传感器坐标系

    // 提取平移向量
    t.transform.translation.x = transform(0, 3);
    t.transform.translation.y = transform(1, 3);
    t.transform.translation.z = transform(2, 3);

    // 提取旋转矩阵并转为四元数
    Eigen::Matrix3f rot_matrix = transform.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rot_matrix);

    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(t); // 广播 TF
}

} // namespace radar_lidar

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::Localization)