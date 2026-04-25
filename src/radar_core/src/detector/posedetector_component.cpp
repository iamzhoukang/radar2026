#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <vector>
#include <cmath>

#include "utils/pose.hpp"

namespace radar_core {

class PoseDetectorComponent : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr angle_pub_; 
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;   

    std::unique_ptr<PoseModel> pose_model_;

    std::vector<cv::Point3f> object_points_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    //  Debug 调试时设为true
    bool show_debug_window_ = true; 

    void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
            return;
        }

        if (pose_model_->Detect(frame)) {
            int text_y_offset = 30; 

            for (const auto& target : pose_model_->detectResults) {
                
                std::vector<cv::Point2f> image_points;
                std::vector<cv::Point3f> valid_obj_points;

                // 1. 提取有效点并绘制
                for (int i = 0; i < 8; ++i) {
                    if (target.keypoints[i].visibility > 0.5f) {
                        image_points.push_back(target.keypoints[i].pt);
                        valid_obj_points.push_back(object_points_[i]); 
                        
                        // 画红点与黄色编号
                        cv::circle(frame, target.keypoints[i].pt, 4, cv::Scalar(0, 0, 255), -1);
                        cv::putText(frame, std::to_string(i), target.keypoints[i].pt, 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                    }
                }

                // 画出绿色的大框
                cv::rectangle(frame, target.box, cv::Scalar(0, 255, 0), 2);

                // 2. PnP 解算
                if (image_points.size() >= 4) {
                    cv::Mat rvec, tvec;
                    bool success = cv::solvePnP(valid_obj_points, image_points, 
                                                camera_matrix_, dist_coeffs_, 
                                                rvec, tvec, false, cv::SOLVEPNP_EPNP);

                    if (success) {
                        double x = tvec.at<double>(0, 0);
                        double y = tvec.at<double>(1, 0);
                        double z = tvec.at<double>(2, 0);

                        double yaw   = atan2(x, z) * 180.0 / M_PI;  
                        double pitch = atan2(-y, z) * 180.0 / M_PI; 
                        double dist  = sqrt(x*x + y*y + z*z);       

                        // 发布角度
                        auto angle_msg = geometry_msgs::msg::Point();
                        angle_msg.x = pitch;
                        angle_msg.y = yaw;
                        angle_msg.z = dist; 
                        angle_pub_->publish(angle_msg);

                        // 3. 在画面左上角打印 Pitch 和 Yaw
                        char text[64];
                        snprintf(text, sizeof(text), "Pitch: %5.1f | Yaw: %5.1f | Dist: %.2fm", pitch, yaw, dist);
                        
                        // 用小号绿色字体打印在左上角 (X=15, Y=动态偏移)
                        cv::putText(frame, text, cv::Point(15, text_y_offset),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                        
                        // 让下一个目标的文字往下移 25 个像素
                        text_y_offset += 25; 
                    }
                }
            }
        }

        // 发布用于 Rviz/rqt_image_view 的调试图像
        auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        debug_pub_->publish(*debug_msg);

        // 4. 本地极速弹出 OpenCV 调试视窗
        if (show_debug_window_) {
            cv::namedWindow("Pose Debug Window", cv::WINDOW_NORMAL);
            
            cv::resizeWindow("Pose Debug Window", 1280, 720);
            
            cv::imshow("Pose Debug Window", frame);
            cv::waitKey(1); 
        }
    }

public:
    explicit PoseDetectorComponent(const rclcpp::NodeOptions & options)
    : Node("pose_detector_node", options)
    {
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
            RCLCPP_INFO(this->get_logger(), "相机内参加载成功!");
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "读取相机 YAML 失败: %s", e.what());
            camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
            dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
        }

        try {
            std::string engine_path = "/home/lzhros/Code/RadarStation/model/engine/module_s_400.engine"; 
            pose_model_ = std::make_unique<PoseModel>(engine_path, 640, 0.5f, 0.45f, 1, 8, true);
            RCLCPP_INFO(this->get_logger(), "TRT Pose 模型加载成功!");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "模型加载失败: %s", e.what());
            return;
        }

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs016_topic", 10, std::bind(&PoseDetectorComponent::image_callback, this, std::placeholders::_1));

        angle_pub_ = this->create_publisher<geometry_msgs::msg::Point>("module_angle_cmd", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("pose_debug_image", 10);
        
        RCLCPP_INFO(this->get_logger(), "姿态检测解算节点已启动，正在等待图像...");
    }
    
    // 析构时记得销毁 OpenCV 窗口
    ~PoseDetectorComponent() {
        if (show_debug_window_) {
            cv::destroyAllWindows();
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::PoseDetectorComponent)