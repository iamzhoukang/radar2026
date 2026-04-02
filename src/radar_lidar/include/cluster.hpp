#ifndef RADAR_LIDAR__CLUSTER_HPP_
#define RADAR_LIDAR__CLUSTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// 引入你的自定义消息接口
#include "radar_interfaces/msg/lidar_cluster.hpp"
#include "radar_interfaces/msg/lidar_cluster_results.hpp"

// PCL 核心
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

namespace radar_lidar {

class ClusterNode : public rclcpp::Node {
public:
    explicit ClusterNode(const rclcpp::NodeOptions & options);

private:
    // 回调函数：接收动态点云，输出聚类结果
    void dynamicCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // ROS 2 对象
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_dynamic_;
    rclcpp::Publisher<radar_interfaces::msg::LidarClusterResults>::SharedPtr pub_results_;

    // 聚类核心参数 (建议后续放入 YAML 配置文件)
    float cluster_tolerance_; // 聚类距离阈值
    int min_cluster_size_;    // 簇最少点数
    int max_cluster_size_;    // 簇最多点数
};

} // namespace radar_lidar

#endif