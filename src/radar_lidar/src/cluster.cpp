#include "cluster.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>

namespace radar_lidar {

ClusterNode::ClusterNode(const rclcpp::NodeOptions & options)
: Node("cluster_node", options) {
    // 1. 参数声明（默认值适配 2026 赛季大场地）
    cluster_tolerance_ = this->declare_parameter("cluster_tolerance", 0.6); // 60cm 宽的机器人
    min_cluster_size_ = this->declare_parameter("min_cluster_size", 15);
    max_cluster_size_ = this->declare_parameter("max_cluster_size", 2000);

    // 2. 初始化订阅与发布
    sub_dynamic_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10, std::bind(&ClusterNode::dynamicCloudCallback, this, std::placeholders::_1));
    
    pub_results_ = this->create_publisher<radar_interfaces::msg::LidarClusterResults>(
        "/radar/lidar_clusters", 10);

    RCLCPP_INFO(this->get_logger(), "聚类节点已拉起，监听动态点云流...");
}

void ClusterNode::dynamicCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 数据格式转换
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) return;

    // 构建 KdTree 空间索引
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // 执行欧几里得聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_); 
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // 封装自定义消息结果
    radar_interfaces::msg::LidarClusterResults results_msg;
    results_msg.header = msg->header;

    int current_id = 0;
    for (const auto& indices : cluster_indices) {
        // 计算该簇的几何中心 (Centroid)
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, indices, centroid);

        // 2. 填充单体簇消息
        radar_interfaces::msg::LidarCluster single_cluster;
        single_cluster.id = current_id++;
        single_cluster.center.x = centroid[0];
        single_cluster.center.y = centroid[1];
        single_cluster.center.z = centroid[2];
        
        //计算簇的大小（此处用点数表示，也可根据 AABB 框计算体积）
        single_cluster.cluster_size = static_cast<float>(indices.indices.size());

        results_msg.clusters.push_back(single_cluster);
    }

    // 发布结果供卡尔曼滤波或视觉融合使用
    pub_results_->publish(results_msg);
}

} // namespace radar_lidar

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::ClusterNode)