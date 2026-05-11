#include "cluster.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h> // 【性能优化】体素滤波
#include <Eigen/Core>

namespace radar_lidar {

ClusterNode::ClusterNode(const rclcpp::NodeOptions & options)
: Node("cluster_node", options) {
    // 加载参数
    cluster_tolerance_ = this->declare_parameter("cluster_tolerance", 0.6); 
    max_cluster_size_ = this->declare_parameter("max_cluster_size", 2000); 
    min_cluster_size_ = this->declare_parameter("min_cluster_size", 15);
    
    drone_min_size_ = this->declare_parameter("drone_min_size", 5);       
    drone_min_height_ = this->declare_parameter("drone_min_height", 1.2); 

    sub_dynamic_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10, std::bind(&ClusterNode::dynamicCloudCallback, this, std::placeholders::_1));
    
    pub_results_ = this->create_publisher<radar_interfaces::msg::LidarClusterResults>(
        "/radar/lidar_clusters", 10);

    RCLCPP_INFO(this->get_logger(), "【点云聚类】引擎启动，防空甄别高度: %.1fm", drone_min_height_);
}

void ClusterNode::dynamicCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) return;

    // 【性能优化】：聚类前必须降采样，防止多点造成 KD-Tree 搜索崩溃
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.filter(*downsampled_cloud);

    if (downsampled_cloud->empty()) return;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(downsampled_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_); 
    ec.setMinClusterSize(drone_min_size_); // 底层放行最小的无人机点数
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(downsampled_cloud);
    ec.extract(cluster_indices);

    radar_interfaces::msg::LidarClusterResults results_msg;
    results_msg.header = msg->header;

    int current_id = 0;
    for (const auto& indices : cluster_indices) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*downsampled_cloud, indices, centroid);
        
        float z_height = centroid[2];
        int pts_count = indices.indices.size();

        // 核心逻辑：身份二次审查
        bool is_valid_ground = (z_height < drone_min_height_ && pts_count >= min_cluster_size_);
        bool is_valid_drone  = (z_height >= drone_min_height_ && pts_count >= drone_min_size_);

        if (!is_valid_ground && !is_valid_drone) continue; 

        radar_interfaces::msg::LidarCluster single_cluster;
        single_cluster.id = current_id++;
        single_cluster.center.x = centroid[0];
        single_cluster.center.y = centroid[1];
        single_cluster.center.z = centroid[2];
        single_cluster.cluster_size = static_cast<float>(pts_count);

        results_msg.clusters.push_back(single_cluster);
    }

    pub_results_->publish(results_msg);
}

} // namespace radar_lidar

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::ClusterNode)