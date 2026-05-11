#ifndef RADAR_LIDAR__LOCALIZATION_HPP_
#define RADAR_LIDAR__LOCALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>

namespace radar_lidar {

class Localization : public rclcpp::Node {
public:
    explicit Localization(const rclcpp::NodeOptions &options);

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    bool loadMap(const std::string & map_path);
    void broadcastTransform(const Eigen::Matrix4f & transform);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_; 

    bool is_initialized_ = false;   
    bool has_aligned_ = false;      
    Eigen::Matrix4f init_guess_;    
    Eigen::Matrix4f current_pose_;  
};

} // namespace radar_lidar

#endif // RADAR_LIDAR__LOCALIZATION_HPP_