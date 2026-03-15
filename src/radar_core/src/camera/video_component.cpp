#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <memory>
#include <chrono>

namespace radar_core
{

class VideoComponent : public rclcpp::Node
{
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    cv::VideoCapture cap_;
    
    // 多线程控制变量
    std::thread capture_thread_;
    std::atomic<bool> is_running_;
    
    // 视频帧率控制
    double fps_;
    int frame_delay_ms_;

    void captureLoop()
    {   
        // 只要 ROS 正常运行且标志位为 true，就持续读取
        while(rclcpp::ok() && is_running_)
        {   
            auto start_time = std::chrono::steady_clock::now();

            cv::Mat frame;
            cap_ >> frame;

            // 【Debug 利器：循环播放】如果视频读完，把进度条拉回第 0 帧
            if(frame.empty()) {
                RCLCPP_WARN(this->get_logger(), "视频播放结束，自动重头循环播放...");
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }

            // 【零拷贝构造】
            auto msg = std::make_unique<sensor_msgs::msg::Image>();
            msg->header.stamp = this->now();
            
            // 注意：这里为了后续和真实相机无缝切换，通常把 frame_id 设为一样
            // 或者在 Launch 文件中统一配置，这里先用 video_frame
            msg->header.frame_id = "video_frame";

            // 将 cv::Mat 数据拷贝到 msg
            cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg(*msg);

            // 零拷贝发布
            pub_->publish(std::move(msg));

            // 【核心逻辑：帧率控制补偿】
            auto end_time = std::chrono::steady_clock::now();
            auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            // 计算需要休眠的时间 = 理论每帧耗时 - 刚刚处理这张图的耗时
            int sleep_time = frame_delay_ms_ - process_time;
            if (sleep_time > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            }
        }
    }

public:
    explicit VideoComponent(const rclcpp::NodeOptions & options)
    : Node("video_component", options), is_running_(false)
    {
        // 1. 声明并获取视频路径参数
        this->declare_parameter<std::string>("video_path", "/home/lzhros/Code/RadarStation/video/hdlg.mp4");
        std::string video_path = this->get_parameter("video_path").as_string();

        // 2. 打开视频
        cap_.open(video_path);
        if(!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开视频文件: %s", video_path.c_str());
            return;
        }

        // 3. 获取视频原生属性
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0) fps_ = 30.0; // 防止获取失败给出默认值
        frame_delay_ms_ = static_cast<int>(1000.0 / fps_);
        
        RCLCPP_INFO(this->get_logger(), "视频加载成功 | 分辨率: %dx%d | 原生FPS: %.2f", 
                    (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH),
                    (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT),
                    fps_);

        // 4. 创建发布者 (注意：为了让你下游的神经网络不用改代码，你可以直接发布到 video_topic)
        // 这样神经网络根本不知道数据是来自相机还是视频
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("video_topic", 10);

        // 5. 启动读取线程
        is_running_ = true;
        capture_thread_ = std::thread(&VideoComponent::captureLoop, this);
    }

    ~VideoComponent() 
    {
        is_running_ = false;
        if(capture_thread_.joinable()) {
            capture_thread_.join();
        }
        if(cap_.isOpened()) {
            cap_.release();
        }
        RCLCPP_INFO(this->get_logger(), "视频流子线程已安全关闭。");
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::VideoComponent)