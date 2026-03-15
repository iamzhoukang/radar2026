#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h> 
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <thread> 

#include "radar_interfaces/msg/vision_target_angle.hpp"
#include "utils/model.hpp"

namespace radar_core 
{

class AirDetectorComponent : public rclcpp::Node 
{
public:
    explicit AirDetectorComponent(const rclcpp::NodeOptions & options) 
    : Node("air_detector_component", options) 
    {
        // 1. 声明所有参数
        this->declare_parameter<std::string>("camera_config_path", "/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml");
        this->declare_parameter<std::string>("model_config_path", "/home/lzhros/Code/RadarStation/config/detector/yolo.yaml");
        
        // 【新增】：声明 Debug 图像渲染开关，默认关闭 (false) 保护算力
        this->declare_parameter<bool>("enable_debug_stream", false);
        enable_debug_stream_ = this->get_parameter("enable_debug_stream").as_bool();

        // 2. 加载参数与模型
        load_camera_params(this->get_parameter("camera_config_path").as_string());
        init_models(this->get_parameter("model_config_path").as_string());

        // 3. 通信话题初始化
        pub_angle_ = this->create_publisher<radar_interfaces::msg::VisionTargetAngle>("wide_camera/angle", 10);
        pub_debug_img_ = this->create_publisher<sensor_msgs::msg::Image>("wide_camera/debug_img", 5);
        
        sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 
            5,
            std::bind(&AirDetectorComponent::imageCallback, this, std::placeholders::_1)
        );

        if (enable_debug_stream_) {
            RCLCPP_INFO(this->get_logger(), "防空组件已启动 (3抽1降帧开启 | Debug渲染流: 开启)。");
        } else {
            RCLCPP_INFO(this->get_logger(), "防空组件已启动 (3抽1降帧开启 | Debug渲染流: 关闭 [极致性能模式])。");
        }
    }

private:
    cv::Mat K_; 
    cv::Mat D_; 
    std::unique_ptr<Model> air_model_; 
    
    int frame_counter_ = 0; 
    bool enable_debug_stream_; // 【新增】：开关状态存储

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Publisher<radar_interfaces::msg::VisionTargetAngle>::SharedPtr pub_angle_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_img_;

    void load_camera_params(const std::string& yaml_path) {
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (!config["camera"]) throw std::runtime_error("缺失 camera 根节点");
        
        auto camera_node = config["camera"];
        K_ = cv::Mat::zeros(3, 3, CV_64F);
        auto K_node = camera_node["K"];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                K_.at<double>(i, j) = K_node[i * 3 + j].as<double>(); 

        auto D_node = camera_node["dist"];
        D_ = cv::Mat::zeros(1, D_node.size(), CV_64F);
        for (size_t i = 0; i < D_node.size(); ++i)
            D_.at<double>(0, i) = D_node[i].as<double>();
    }

    void init_models(const std::string& yaml_path) {
        YAML::Node config = YAML::LoadFile(yaml_path);
        air_model_ = std::make_unique<Model>(
            config["plane_modelpath"].as<std::string>(), 
            config["plane_inputSize"].as<int>(), 
            config["plane_scoreThresh"].as<float>(), 
            config["plane_nmsThresh"].as<float>(), 
            true 
        );
    }

    void imageCallback(sensor_msgs::msg::Image::UniquePtr msg) 
    {
        frame_counter_++;
        if (frame_counter_ % 3 != 0) return; 

        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());

        radar_interfaces::msg::VisionTargetAngle angle_msg;
        angle_msg.header = msg->header; 
        angle_msg.is_detected = false;
        angle_msg.yaw_relative = 0.0;
        angle_msg.pitch_relative = 0.0;

        bool is_drone_found = false;
        cv::Rect best_box;

        if (air_model_->Detect(frame) && !air_model_->detectResults.empty()) {
            auto best_drone = std::max_element(
                air_model_->detectResults.begin(),
                air_model_->detectResults.end(),
                [](const Result& a, const Result& b) { return a.confidence < b.confidence; }
            );

            best_box = best_drone->box;
            cv::Point2f target_center(
                best_box.x + best_box.width / 2.0f,
                best_box.y + best_box.height / 2.0f
            );

            std::vector<cv::Point2f> src_pts = { target_center }, dst_pts;
            cv::undistortPoints(src_pts, dst_pts, K_, D_);
            
            angle_msg.yaw_relative = std::atan2(dst_pts[0].x, 1.0);
            angle_msg.pitch_relative = std::atan2(-dst_pts[0].y, 1.0); 
            angle_msg.is_detected = true;
            is_drone_found = true;
        }

        // 核心角度数据永远发布
        pub_angle_->publish(angle_msg);

        // ==========================================
        // 【核心控制流】：只有开关开启时，才拉起渲染线程
        // ==========================================
        if (enable_debug_stream_) {
            std::thread([this, msg_ptr = std::move(msg), is_drone_found, best_box, 
                         yaw = angle_msg.yaw_relative, pitch = angle_msg.pitch_relative]() mutable {
                
                cv::Mat raw_frame(msg_ptr->height, msg_ptr->width, CV_8UC3, msg_ptr->data.data());
                publish_debug_video(raw_frame, is_drone_found, best_box, yaw, pitch, msg_ptr->header);
                
            }).detach();
        }
        // 如果开关关闭，msg 对象会在这里随着作用域结束而被自动安全销毁，无任何额外开销
    }

    void publish_debug_video(cv::Mat& raw_frame, bool is_detected, const cv::Rect& rect, float yaw, float pitch, const std_msgs::msg::Header& header)
    {
        cv::Mat vis_frame;
        float scale = 1280.0f / raw_frame.cols; 
        cv::resize(raw_frame, vis_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);

        if (is_detected) {
            cv::Rect scaled_rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale);
            cv::Point scaled_center(scaled_rect.x + scaled_rect.width / 2.0f, scaled_rect.y + scaled_rect.height / 2.0f);

            cv::rectangle(vis_frame, scaled_rect, cv::Scalar(0, 0, 255), 2);
            cv::circle(vis_frame, scaled_center, 4, cv::Scalar(0, 255, 0), -1);

            char text[64];
            snprintf(text, sizeof(text), "Y:%.2f P:%.2f", yaw, pitch);
            cv::putText(vis_frame, text, cv::Point(scaled_rect.x, std::max(20, scaled_rect.y - 10)), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        auto out_msg = cv_bridge::CvImage(header, "bgr8", vis_frame).toImageMsg();
        pub_debug_img_->publish(*out_msg);
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::AirDetectorComponent)