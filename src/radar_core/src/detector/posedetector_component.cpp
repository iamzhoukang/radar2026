#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>

// Linux 底层串口通信头文件
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "utils/pose.hpp"

namespace radar_core {

class PoseDetectorComponent : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;   

    std::unique_ptr<PoseModel> pose_model_;

    std::vector<cv::Point3f> object_points_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // ==========================================
    // 1. 多线程架构核心：回调组隔离
    // ==========================================
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
    rclcpp::CallbackGroup::SharedPtr sub_cb_group_;

    // ==========================================
    // 2. 串口与解耦变量
    // ==========================================
    int serial_fd_ = -1; 
    bool is_offline_mode_ = false;

    rclcpp::TimerBase::SharedPtr serial_timer_; 
    rclcpp::TimerBase::SharedPtr stats_timer_;

    std::atomic<float> current_pitch_{0.0f};
    std::atomic<float> current_yaw_{0.0f};
    std::atomic<float> current_dist_{0.0f};  // 【新增】用于记录距离的原子变量

    std::atomic<int> send_count_{0};       
    std::atomic<int> current_tx_hz_{0};    
    
    int frame_count_ = 0;                  
    double current_fps_ = 0.0;             
    std::chrono::steady_clock::time_point last_fps_time_;

    #pragma pack(push, 1)
    struct SerialPacket {
        uint8_t head = 0x38;  
        float pitch;          
        float yaw;            
        uint8_t tail = 0x83;  
    };
    #pragma pack(pop)

    void init_serial(const std::string& port_name) {
        serial_fd_ = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (serial_fd_ == -1) {
            is_offline_mode_ = true;
            RCLCPP_WARN(this->get_logger(), "未检测到串口 %s，已切入离线模拟模式", port_name.c_str());
            return;
        }

        struct termios options;
        tcgetattr(serial_fd_, &options);
        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        tcsetattr(serial_fd_, TCSANOW, &options);
        
        is_offline_mode_ = false;
        RCLCPP_INFO(this->get_logger(), "长焦控制串口已就绪!");
    }

    void serial_send_loop() {
        float p = current_pitch_.load(std::memory_order_relaxed);
        float y = current_yaw_.load(std::memory_order_relaxed);

        if (!is_offline_mode_ && serial_fd_ != -1) {
            SerialPacket packet;
            packet.pitch = p;
            packet.yaw   = y;
            int bytes_written = write(serial_fd_, &packet, sizeof(SerialPacket));
            if (bytes_written == sizeof(SerialPacket)) {
                send_count_.fetch_add(1, std::memory_order_relaxed); 
            }
        } else {
            send_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void stats_loop() {
        current_tx_hz_.store(send_count_.exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        frame_count_++;
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - last_fps_time_).count();
        if (elapsed_sec >= 1.0) {
            current_fps_ = frame_count_ / elapsed_sec;
            frame_count_ = 0;
            last_fps_time_ = now;
        }

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            return;
        }

        bool target_found = false;

        if (pose_model_->Detect(frame)) {
            for (const auto& target : pose_model_->detectResults) {
                
                std::vector<cv::Point2f> image_points;
                std::vector<cv::Point3f> valid_obj_points;

                for (int i = 0; i < 8; ++i) {
                    if (target.keypoints[i].visibility > 0.5f) {
                        image_points.push_back(target.keypoints[i].pt);
                        valid_obj_points.push_back(object_points_[i]); 
                        cv::circle(frame, target.keypoints[i].pt, 4, cv::Scalar(0, 0, 255), -1);
                    }
                }
                cv::rectangle(frame, target.box, cv::Scalar(0, 255, 0), 2);

                if (image_points.size() >= 4) {
                    cv::Mat rvec, tvec;
                    bool success = cv::solvePnP(valid_obj_points, image_points, 
                                                camera_matrix_, dist_coeffs_, 
                                                rvec, tvec, false, cv::SOLVEPNP_EPNP);

                    if (success) {
                        double x = tvec.at<double>(0, 0);
                        double y = tvec.at<double>(1, 0);
                        double z = tvec.at<double>(2, 0);

                        double raw_yaw   = atan2(x, z) * 180.0 / M_PI;  
                        double raw_pitch = atan2(-y, z) * 180.0 / M_PI; 
                        
                        // 】根据 x, y, z 求解直线物理距离 (米)
                        double dist = std::sqrt(x * x + y * y + z * z);

                        current_pitch_.store(static_cast<float>(raw_pitch) - 1.1f, std::memory_order_relaxed);
                        current_yaw_.store(static_cast<float>(raw_yaw) - 4.0f, std::memory_order_relaxed);
                        current_dist_.store(static_cast<float>(dist), std::memory_order_relaxed); // 保存距离
                        
                        target_found = true;
                    }
                }
            }
        }

        // ==========================================
        // 3. OSD 渲染区
        // ==========================================
        float p_sent = current_pitch_.load(std::memory_order_relaxed);
        float y_sent = current_yaw_.load(std::memory_order_relaxed);
        float d_sent = current_dist_.load(std::memory_order_relaxed); // 【新增】读取距离
        int tx_hz = current_tx_hz_.load(std::memory_order_relaxed);

        char text_buffer[128];
        int y_offset = 35; 
        int line_height = 30; 

        snprintf(text_buffer, sizeof(text_buffer), "Vision FPS : %.1f", current_fps_);
        cv::putText(frame, text_buffer, cv::Point(20, y_offset), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 165, 0), 2);
        y_offset += line_height;

        snprintf(text_buffer, sizeof(text_buffer), "Serial TX  : %d Hz", tx_hz);
        cv::putText(frame, text_buffer, cv::Point(20, y_offset), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        y_offset += line_height;

        // 【修改】去掉了 Raw 显示，直接把 P、Y、D 打印在同一行
        if (target_found) {
            snprintf(text_buffer, sizeof(text_buffer), "Target     : P:%.1f Y:%.1f D:%.2fm", p_sent, y_sent, d_sent);
            cv::putText(frame, text_buffer, cv::Point(20, y_offset), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        } else {
            snprintf(text_buffer, sizeof(text_buffer), "Target     : LOST");
            cv::putText(frame, text_buffer, cv::Point(20, y_offset), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        debug_pub_->publish(*debug_msg);
    }

public:
    explicit PoseDetectorComponent(const rclcpp::NodeOptions & options)
    : Node("pose_detector_node", options)
    {
        init_serial("/dev/ttyUSB0");

        object_points_ = {
            cv::Point3f(-0.025f,   0.01f,   0.0f),
            cv::Point3f(-0.01535f, 0.01f,   0.0146f),
            cv::Point3f( 0.01535f, 0.01f,   0.0146f),
            cv::Point3f( 0.025f,   0.01f,   0.0f),
            cv::Point3f(-0.025f,  -0.01f,   0.0f),
            cv::Point3f(-0.01535f,-0.01f,   0.0146f),
            cv::Point3f( 0.01535f,-0.01f,   0.0146f),
            cv::Point3f( 0.025f,  -0.01f,   0.0f)
        };

        std::string yaml_path = "/home/lzhros/Code/RadarStation/config/camera/cs016.yaml";
        try {
            YAML::Node config = YAML::LoadFile(yaml_path);
            std::vector<double> K_vec = config["camera"]["K"].as<std::vector<double>>();
            std::vector<double> dist_vec = config["camera"]["dist"].as<std::vector<double>>();
            camera_matrix_ = cv::Mat(3, 3, CV_64F, K_vec.data()).clone();
            dist_coeffs_ = cv::Mat(1, 5, CV_64F, dist_vec.data()).clone();
        } catch (...) {
            camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
            dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
        }

        try {
            std::string engine_path = "/home/lzhros/Code/RadarStation/model/engine/module_s_400.engine"; 
            pose_model_ = std::make_unique<PoseModel>(engine_path, 640, 0.5f, 0.45f, 1, 8, true);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "模型加载失败: %s", e.what());
            return;
        }

        // ==========================================
        // 4. 初始化多线程回调组
        // ==========================================
        timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        sub_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        last_fps_time_ = std::chrono::steady_clock::now();

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = sub_cb_group_;
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs016_topic", 10, 
            std::bind(&PoseDetectorComponent::image_callback, this, std::placeholders::_1),
            sub_options);

        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("pose_debug_image", 10);
        
        serial_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&PoseDetectorComponent::serial_send_loop, this),
            timer_cb_group_
        );

        stats_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000),
            std::bind(&PoseDetectorComponent::stats_loop, this),
            timer_cb_group_
        );

        RCLCPP_INFO(this->get_logger(), "姿态检测与 1000Hz 无锁并发串口发送引擎已启动...");
    }
    
    ~PoseDetectorComponent() {
        if (serial_fd_ != -1) {
            close(serial_fd_);
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::PoseDetectorComponent)