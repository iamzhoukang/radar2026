#include "dynamic_cloud.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <tf2_eigen/tf2_eigen.hpp> 
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace radar_lidar {

DynamicCloud::DynamicCloud(const rclcpp::NodeOptions & options) 
: Node("dynamic_cloud_node", options), 
  tf_buffer_(this->get_clock()), 
  tf_listener_(tf_buffer_) 
{
    // 1. 声明并获取参数
    this->declare_parameter("map_path", "/home/lzhros/Code/RadarStation/config/lidar/RB2026_rmuc.pcd");
    this->declare_parameter("threshold", 0.2);
    
    std::string map_path = this->get_parameter("map_path").as_string();
    threshold_ = this->get_parameter("threshold").as_double();

    // 2. 加载静态地图并构建空间索引
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *map_cloud_) == -1) {
        RCLCPP_ERROR(this->get_logger(), "动态节点无法加载地图: %s", map_path.c_str());
        return;
    }
    
    // 构建 KD-Tree，用于执行极速最近邻搜索
    kd_tree_.setInputCloud(map_cloud_);

    // 3. 通信接口初始化
    sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10, std::bind(&DynamicCloud::lidarCallback, this, std::placeholders::_1));
    
    pub_dynamic_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar_dynamic", 10);
    
    RCLCPP_INFO(this->get_logger(), "动态点云提取节点已就绪，背景阈值: %.2fm", threshold_);
}

void DynamicCloud::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1. 坐标变换查找：获取从地图(rm_frame)到当前雷达帧的变换
    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
        // 查找最新的变换矩阵
        tf_stamped = tf_buffer_.lookupTransform("rm_frame", msg->header.frame_id, tf2::TimePointZero);
    } catch (tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "等待定位节点发布 TF...");
        return;
    }

    // 2. 数据转换与空间对齐
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_map(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud_raw);
    
    // 核心转换：Isometry3d -> Matrix4d -> Matrix4f
    Eigen::Matrix4f transform = tf2::transformToEigen(tf_stamped).matrix().cast<float>();
    pcl::transformPointCloud(*cloud_raw, *cloud_in_map, transform);

    // 3. 背景减除：提取动态点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dynamic(new pcl::PointCloud<pcl::PointXYZ>());
    filterBackground(cloud_in_map, cloud_dynamic);

    // 4. 发布结果：坐标系设为 rm_frame
    if (!cloud_dynamic->empty()) {
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud_dynamic, output_msg);
        output_msg.header.frame_id = "rm_frame";
        output_msg.header.stamp = msg->header.stamp;
        pub_dynamic_->publish(output_msg);
    }
}

void DynamicCloud::filterBackground(const pcl::PointCloud<pcl::PointXYZ>::Ptr input, 
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr output) 
{
    float sq_threshold = threshold_ * threshold_;
    for (const auto& point : input->points) {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        // 如果在地图中找不到足够近的点，则判定为动态目标（如机器人）
        if (kd_tree_.nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            if (pointNKNSquaredDistance[0] > sq_threshold) {
                output->points.push_back(point);
            }
        }
    }
}

} // namespace radar_lidar

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::DynamicCloud)