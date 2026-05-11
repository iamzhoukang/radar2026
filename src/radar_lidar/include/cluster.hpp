#ifndef RADAR_LIDAR__CLUSTER_HPP_
#define RADAR_LIDAR__CLUSTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "radar_interfaces/msg/lidar_cluster.hpp"
#include "radar_interfaces/msg/lidar_cluster_results.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

namespace radar_lidar {

class ClusterNode : public rclcpp::Node {
public:
    explicit ClusterNode(const rclcpp::NodeOptions & options);

private:
    void dynamicCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_dynamic_;
    rclcpp::Publisher<radar_interfaces::msg::LidarClusterResults>::SharedPtr pub_results_;

    // 聚类与过滤参数
    float cluster_tolerance_; // 聚类距离阈值
    int min_cluster_size_;    // 地面机器人最少点数
    int max_cluster_size_;    // 簇最多点数

    // 防空专用参数
    int drone_min_size_;      // 无人机最少点数 (点数要求可放宽)
    double drone_min_height_; // 判定为“空中目标”的最低高度阈值
};

} // namespace radar_lidar

#endif // RADAR_LIDAR__CLUSTER_HPP_