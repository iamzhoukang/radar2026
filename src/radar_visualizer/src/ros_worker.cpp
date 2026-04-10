#include "ros_worker.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace radar_visualizer {

RosWorker::RosWorker() : QObject(), Node("radar_vis_node") {
    // 1. 订阅检测后的画框视频 (使用 SensorDataQoS 适应高频丢包网络)
    sub_video_ = this->create_subscription<sensor_msgs::msg::Image>(
        "processed_video", 10,
        std::bind(&RosWorker::video_callback, this, std::placeholders::_1)
    );

    // 2. 订阅小地图
    sub_map_ = this->create_subscription<sensor_msgs::msg::Image>(
        "map/image", 10,
        std::bind(&RosWorker::map_callback, this, std::placeholders::_1)
    );

    // 3. 标定服务客户端
    client_calib_ = this->create_client<std_srvs::srv::Trigger>("solvepnp/start");
    
    // 4. 前哨站ROI配置客户端
    client_outpost_config_ = this->create_client<std_srvs::srv::Trigger>("detector/config_outpost_roi");
}

void RosWorker::toggleMapFlip() {
    is_map_flipped_ = !is_map_flipped_;
    RCLCPP_INFO(this->get_logger(), "地图翻转状态: %s", is_map_flipped_ ? "ON (180°)" : "OFF");
}

void RosWorker::triggerCalibration() {
    if (!client_calib_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "SolvePnP 服务未响应，请检查后台组件！");
        return;
    }
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    client_calib_->async_send_request(req);
    RCLCPP_INFO(this->get_logger(), "已发送 PnP 标定触发请求！");
}

void RosWorker::configOutpostROI() {
    if (!client_outpost_config_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "前哨站配置服务未响应，请检查 detector 组件！");
        return;
    }
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    client_outpost_config_->async_send_request(req);
    RCLCPP_INFO(this->get_logger(), "已发送前哨站 ROI 配置请求，请在 OpenCV 窗口中操作！");
}

void RosWorker::video_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try { cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); } catch (...) { return; }

    cv::Mat mat = cv_ptr->image;
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB); // OpenCV是BGR，Qt是RGB

    // 【核心细节】：必须调用 copy() 进行深拷贝！
    // 因为 mat 是局部变量，函数结束就会销毁。如果浅拷贝，UI 线程去读取时会发生野指针崩溃。
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    emit videoReady(qimg.copy()); 
}

void RosWorker::map_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try { cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); } catch (...) { return; }

    cv::Mat mat = cv_ptr->image;
    
    // 核心新增：一键翻转阵营
    if (is_map_flipped_) {
        cv::rotate(mat, mat, cv::ROTATE_180);
    }

    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    emit mapReady(qimg.copy());
}

} // namespace radar_visualizer