#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <sstream>

// 哈基旭师兄开发的海康sdk
#include "rb26SDK/include/CamreaExmple.hpp"

namespace radar_core
{

class CameraOneComponent : public rclcpp::Node
{
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    sdk::CameraExmple<sdk::HikCamera> cap_;

    std::thread capture_thread_;
    std::thread record_thread_;
    std::atomic<bool> is_running_;

    // 录制相关缓冲池与锁
    std::queue<cv::Mat> record_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    cv::VideoWriter writer_;
    const size_t MAX_QUEUE_SIZE = 60; // 防止内存撑爆的队列上限
    
    bool enable_recording_ = false;

    void recordLoop()
    {
        while (is_running_ || !record_queue_.empty()) {
            cv::Mat frame_to_write;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { 
                    return !record_queue_.empty() || !is_running_; 
                });

                if (record_queue_.empty() && !is_running_) break;

                frame_to_write = std::move(record_queue_.front());
                record_queue_.pop();
            }

            if (!frame_to_write.empty() && writer_.isOpened()) {
                writer_.write(frame_to_write);
            }
        }
        if(writer_.isOpened()) writer_.release();
        RCLCPP_INFO(this->get_logger(), "录制已停止，MP4 (H.265) 文件已安全关闭。");
    }

    void captureLoop()
    {
        while (rclcpp::ok() && is_running_) {
            cv::Mat frame = cap_.getFrame(0, 0);
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            // 1. 如果开启了内录，才进行深拷贝并压入队列
            if (enable_recording_) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (record_queue_.size() < MAX_QUEUE_SIZE) {
                    record_queue_.push(frame.clone()); // 必须 clone
                }
                queue_cv_.notify_one();
            }

            // 2. 发布 ROS 消息
            auto msg = std::make_unique<sensor_msgs::msg::Image>();
            msg->header.stamp = this->now();
            msg->header.frame_id = "cs200_frame";
            cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg(*msg);
            pub_->publish(std::move(msg));
        }
    }

    // 获取当前时间字符串用于命名
    std::string getCurrentTimeString() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }

public:
    explicit CameraOneComponent(const rclcpp::NodeOptions &options)
        : Node("camera_one_node", options), is_running_(false)
    {
        // 1. 声明并获取参数
        this->declare_parameter<bool>("enable_recording", false); 
        this->declare_parameter<std::string>("record_path", "/home/lzhros/Code/RadarStation/recording/");
        
        enable_recording_ = this->get_parameter("enable_recording").as_bool();
        std::string base_path = this->get_parameter("record_path").as_string();

        // 提前将 is_running_ 设为 true，防止录制线程因为 false 直接退出
        is_running_ = true; 

        // 2. 初始化海康相机
        sdk::CameraExmple<sdk::HikCamera>::CameraSDKInit();
        const char *sn = "DA7831910";
        if (!cap_.CameraInit(const_cast<char *>(sn), true, 10000, 0.7, 0.3)) {
            RCLCPP_ERROR(this->get_logger(), "相机一初始化失败！请检查连接或是否被占用。");
            is_running_ = false; // 初始化失败，重置状态
            return;
        }

        // 3. 根据开关决定是否初始化录制
        if (enable_recording_) {
            if (!base_path.empty() && base_path.back() != '/') {
                base_path += "/";
            }
            
            std::string full_file_path = base_path + "cam1_" + getCurrentTimeString() + ".mp4";

            int frame_width = 5472;
            int frame_height = 3648;
            
            // 【核心修复】：升级为 H.265 (nvh265enc) 编码器，完美吞吐 5.4K 超高分辨率
            std::string pipeline = "appsrc ! videoconvert ! video/x-raw, format=I420 ! "
                                   "nvh265enc bitrate=80000 preset=default rc-mode=cbr ! "
                                   "h265parse ! mp4mux ! filesink location=" + full_file_path;

            writer_.open(pipeline, cv::CAP_GSTREAMER, 0, 20.0, cv::Size(frame_width, frame_height));
            
            if (!writer_.isOpened()) {
                RCLCPP_WARN(this->get_logger(), "GStreamer H.265 硬编开启失败，降级为原生 MP4 软编录制...");
                writer_.open(full_file_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20.0, cv::Size(frame_width, frame_height));
            }

            if (writer_.isOpened()) {
                RCLCPP_INFO(this->get_logger(), "内录已开启，5.4K 原画质 H.265 视频保存至: %s", full_file_path.c_str());
                record_thread_ = std::thread(&CameraOneComponent::recordLoop, this);
            } else {
                RCLCPP_ERROR(this->get_logger(), "所有录制管道均开启失败！关闭本次内录功能。");
                enable_recording_ = false; 
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "内录功能已禁用 (参数 enable_recording=false)。");
        }

        // 4. 初始化发布者与主采集线程
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("cs200_topic", 10);
        capture_thread_ = std::thread(&CameraOneComponent::captureLoop, this);
    }

    ~CameraOneComponent()
    {
        is_running_ = false;
        
        if (enable_recording_) {
            queue_cv_.notify_all(); // 唤醒录制线程把剩下的帧写完
            if (record_thread_.joinable()) record_thread_.join();
        }
        
        if (capture_thread_.joinable()) capture_thread_.join();
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::CameraOneComponent)
