#ifndef RADAR_LIDAR__DYNAMIC_CLOUD_HPP_
#define RADAR_LIDAR__DYNAMIC_CLOUD_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

// 【新增】TF2 坐标变换监听相关头文件
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace radar_lidar {

class DynamicCloud : public rclcpp::Node {
public:
    explicit DynamicCloud(const rclcpp::NodeOptions & options);

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    bool loadStaticMap(const std::string& pcd_path);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_raw_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dynamic_;

    // 背景扣除核心工具
    pcl::PointCloud<pcl::PointXYZ>::Ptr static_map_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_map_;
    
    // 动态提取阈值 (点离静态地图多远才算动态点)
    double distance_threshold_;

    // 【新增】TF 监听器对象，用于处理实验室或实战中的坐标对齐
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

} // namespace radar_lidar

#endif