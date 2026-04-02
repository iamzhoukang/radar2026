#include "localization.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

namespace radar_lidar {

Localization::Localization(const rclcpp::NodeOptions & options)
: Node("localization_node", options)
{
    RCLCPP_INFO(this->get_logger(), "定位节点启动中...");

    // 1. 初始化 TF 广播器
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // 2. 加载 PCD 地图 (请确保路径与你实际存放位置一致)
    std::string map_path = "/home/lzhros/Code/RadarStation/config/lidar/RB2026_rmuc.pcd";
    if (!loadMap(map_path)) {
        RCLCPP_ERROR(this->get_logger(), "无法读取 PCD 地图文件，请检查路径！");
        return;
    }

    // 3. 订阅雷达点云
    sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10, std::bind(&Localization::lidarCallback, this, std::placeholders::_1));

    // 4. 初始化位姿矩阵为单位阵
    current_pose_ = Eigen::Matrix4f::Identity();
    
    RCLCPP_INFO(this->get_logger(), "定位节点就绪，等待 Livox 点云输入...");
}

void Localization::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    //基础数据转换：ROS 消息 -> PCL 格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud_raw);

    // 定位对齐逻辑
    // 如果尚未完成初始对齐，则执行高负载的 GICP 算法
    if (!has_aligned_) {
        // 1. 体素滤波：降采样以提升计算效率
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setLeafSize(0.15f, 0.15f, 0.15f); // 15cm 体素网格
        vox.setInputCloud(cloud_raw);
        vox.filter(*cloud_filtered);

        // 配置广义 ICP 算法 (GICP)
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(cloud_filtered);
        gicp.setInputTarget(map_cloud_);
        
        // 算法收敛参数设置
        gicp.setTransformationEpsilon(1e-8);
        gicp.setMaxCorrespondenceDistance(1.0); 

        pcl::PointCloud<pcl::PointXYZ> output;
        gicp.align(output, current_pose_);

        //3. 结果评估
        if (gicp.hasConverged() && gicp.getFitnessScore() < 0.15) {
            current_pose_ = gicp.getFinalTransformation();
            has_aligned_ = true; // 锁定状态，后续不再重复计算对齐
            RCLCPP_INFO(this->get_logger(), ">>> 定位成功！初始位姿已锁定，匹配得分: %f", gicp.getFitnessScore());
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                 "等待对齐中... 当前匹配得分: %f", gicp.getFitnessScore());
        }

    }

    // 坐标广播逻辑 (核心微调部分)
    // 只要完成过对齐，每收到一帧点云就发布一次 TF，确保坐标树(TF Tree)不中断
    if (has_aligned_) {
        broadcastTransform(current_pose_);
    }
}

bool Localization::loadMap(const std::string & map_path)
{
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *map_cloud_) == -1) {
        return false;
    }
    is_initialized_ = true;
    return true;
}

void Localization::broadcastTransform(const Eigen::Matrix4f & transform)
{
    geometry_msgs::msg::TransformStamped tf_msg;
    
    // 填充标准 TF 消息头
    tf_msg.header.stamp = this->now();
    tf_msg.header.frame_id = "rm_frame";    // 世界坐标系 (地图)
    tf_msg.child_frame_id = "livox_frame";  // 子坐标系 (雷达)

    // 提取平移分量
    tf_msg.transform.translation.x = transform(0, 3);
    tf_msg.transform.translation.y = transform(1, 3);
    tf_msg.transform.translation.z = transform(2, 3);

    // 提取旋转分量并转换为四元数
    Eigen::Matrix3f rotation_matrix = transform.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rotation_matrix);
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    // 发布坐标变换
    tf_broadcaster_->sendTransform(tf_msg);
}

} // namespace radar_lidar

// 注册为 ROS 2 插件组件
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::Localization)