#pragma once

#include <QObject>
#include <QImage>
#include <memory>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace radar_visualizer{

    //多继承即是qt信号源,也是ros节点
    class RosWorker : public QObject, public rclcpp::Node{
        Q_OBJECT //qt元对象系统的核心宏,必须放在私有区第一行

    public:
        RosWorker();
        ~RosWorker() override = default;
    
    //qt槽函数,专门用来接受来自ui线程的鼠标点击信号
    public slots:
        void triggerCalibration();
        void toggleMapFlip();
        void configOutpostROI();  // 【新增】配置前哨站ROI

    signals:// Qt 信号，专门向 UI 线程发送图像
        void videoReady(const QImage &img);
        void mapReady(const QImage &img);

    private:
        // ROS 回调函数
    void video_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void map_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_video_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_map_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_calib_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_outpost_config_;  // 【新增】前哨站配置客户端
    
    std::atomic<bool> is_map_flipped_{false};
    };
}// namespace radar_visualizer