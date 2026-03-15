#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <memory>
//哈基旭师兄开发的海康sdk
#include "rb26SDK/include/CamreaExmple.hpp"

namespace radar_core
{

class CameraOneComponent : public rclcpp::Node
{
    private:
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
        sdk::CameraExmple<sdk::HikCamera> cap_;

        //多线程控制变量
        std::thread capture_thread_;
        std::atomic<bool> is_running_;

        void captureLoop()
        {   
            //只要 ROS 正常运行且标志位为 true，就持续死循环取流
            while(rclcpp::ok() && is_running_)
            {   
                cv::Mat frame = cap_.getFrame(0,0);
                if(frame.empty()) {
                // 如果没取到，短暂休眠让出 CPU 时间片，防止死锁空转
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
                }

                // 【零拷贝构造】
                //  在堆内存中开辟一块干净的 Image 消息空间，由 unique_ptr 独占管理
                auto msg = std::make_unique<sensor_msgs::msg::Image>();

                //填充消息头
                msg->header.stamp = this->now();
                msg->header.frame_id = "cs200_frame";

                //将 cv::Mat 的数据高效拷贝到 msg 中
                // 注意：这里重载的 toImageMsg 传入的是对象的引用 (*msg)，避免了创建额外的 shared_ptr
                cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg(*msg);

                // 使用 std::move 转移所有权给 ROS2 底层
                // 发布完成后，当前的 msg 指针将变为空指针 (nullptr)
                pub_->publish(std::move(msg));
            }
        }

    public:
        //组件化构造函数必须接受 rclcpp::NodeOptions
        explicit CameraOneComponent(const rclcpp::NodeOptions & options)
        :Node("camera_one_node", options),is_running_(false)
        {
            //初始化相机
            sdk::CameraExmple<sdk::HikCamera>::CameraSDKInit();
            const char* sn = "DA7831910";
            if(!cap_.CameraInit(const_cast<char *>(sn), true, 5000, 0.7, 0.3)) {
            RCLCPP_ERROR(this->get_logger(), "相机初始化失败！");
            return;
            }

            //创建发布者
            pub_ = this->create_publisher<sensor_msgs::msg::Image>("cs200_topic",10);

            //启动取流线程
            is_running_ = true;
            capture_thread_ = std::thread(&CameraOneComponent::captureLoop, this);
            RCLCPP_INFO(this->get_logger(), "相机采集子线程已启动，等待图像...");
        }

    ~CameraOneComponent() 
    {
        is_running_ = false;
        if(capture_thread_.joinable()) {
            capture_thread_.join();
        }
        RCLCPP_INFO(this->get_logger(), "相机采集子线程已安全关闭。");
    }
};

}//namespace radar_core

//注册组件宏，这是 Launch 文件能找到它的唯一凭证
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::CameraOneComponent)