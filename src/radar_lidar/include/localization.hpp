#ifndef RADAR_LIDAR__LOCALIZATION_HPP_
#define RADAR_LIDAR__LOCALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// TF2 坐标变换（雷达定位的最终输出就是 TF）
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// PCL 点云库
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>

namespace radar_lidar{
    class Localization : public rclcpp::Node
    {
        public:
            explicit Localization(const rclcpp::NodeOptions &options);
        private:
            // 回调函数：处理接收到的每一帧点云
            void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

            // 初始化函数：加载场地地图
            bool loadMap(const std::string & map_path);

            // TF 广播函数：将解算的位姿发出去
            void broadcastTransform(const Eigen::Matrix4f & transform);

            // --- ROS 2 对象 ---
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
            std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

            // --- PCL 对象 ---
            pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_; // 静态地图（目标点云）

            // --- 算法状态 ---
            bool is_initialized_ = false;   // 标记是否成功加载了地图
            bool has_aligned_ = false;      // 标记是否完成了第一次成功定位
            Eigen::Matrix4f init_guess_;    // 初始猜测位姿（非常重要）
            Eigen::Matrix4f current_pose_;  // 当前雷达在地图中的位姿

    };

}// namespace radar_lidar



#endif // RADAR_LIDAR__LOCALIZATION_HPP_