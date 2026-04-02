#ifndef RADAR_LIDAR__DYNAMIC_CLOUD_HPP_
#define RADAR_LIDAR__DYNAMIC_CLOUD_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h> // 核心：用于快速空间检索
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>


namespace radar_lidar{

    class DynamicCloud:public rclcpp::Node{
        public:
            explicit DynamicCloud(const rclcpp::NodeOptions & options);

        private:
            // 处理函数
            void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
            // 核心算法：背景过滤
            void filterBackground(const pcl::PointCloud<pcl::PointXYZ>::Ptr input, 
                         pcl::PointCloud<pcl::PointXYZ>::Ptr output);

            // ROS 2 对象 
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dynamic_;

            // TF 监听器：获取由 Localization 发布的坐标变换
            tf2_ros::Buffer tf_buffer_;
            tf2_ros::TransformListener tf_listener_;

            // PCL 静态地图与检索树 
            pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
            pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_; // 用于执行高效的最近邻搜索
    
            float threshold_ = 0.2f; // 距离阈值：超过此距离的点被视为动态点
    };


}// namespace radar_lidar

#endif 