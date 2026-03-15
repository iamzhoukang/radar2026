#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <algorithm>

// 自定义消息接口
#include "radar_interfaces/msg/detect_results.hpp" 
#include "radar_interfaces/msg/detect_result.hpp" 
#include "radar_interfaces/msg/radar_map.hpp" 
#include "std_srvs/srv/trigger.hpp"

namespace radar_core {

class MapComponent : public rclcpp::Node
{
public:
    explicit MapComponent(const rclcpp::NodeOptions & options) : Node("map_component", options)
    {
        // 1. 声明并获取路径参数
        this->declare_parameter<std::string>("camera_yaml", "/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml");
        this->declare_parameter<std::string>("map_yaml", "/home/lzhros/Code/RadarStation/config/map/field_image.yaml");
        this->declare_parameter<std::string>("map_image", "/home/lzhros/Code/RadarStation/config/map/field_image.png");
        
        camera_yaml_path_ = this->get_parameter("camera_yaml").as_string();
        map_yaml_path_ = this->get_parameter("map_yaml").as_string();
        map_image_path_ = this->get_parameter("map_image").as_string();

        // 2. 加载参数文件
        if (!load_all_configs()) {
            RCLCPP_ERROR(this->get_logger(), "启动失败：参数文件或地图图片加载错误！");
        } else {
            RCLCPP_INFO(this->get_logger(), "小地图解算节点启动成功");
        }

        // 3. 订阅检测器坐标
        sub_results_ = this->create_subscription<radar_interfaces::msg::DetectResults>(
            "detector/results", 10,
            std::bind(&MapComponent::results_callback, this, std::placeholders::_1)
        );

        // 4. 双路发布者注册
        pub_map_img_ = this->create_publisher<sensor_msgs::msg::Image>("map/image", 10);
        pub_official_ = this->create_publisher<radar_interfaces::msg::RadarMap>("map/official_data", 10);

        // 5. 热重载服务
        srv_reload_ = this->create_service<std_srvs::srv::Trigger>(
            "map/reload_config",
            std::bind(&MapComponent::handle_reload, this, std::placeholders::_1, std::placeholders::_2)
        );
    }

private:
    std::string camera_yaml_path_, map_yaml_path_, map_image_path_;
    rclcpp::Subscription<radar_interfaces::msg::DetectResults>::SharedPtr sub_results_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_map_img_;
    rclcpp::Publisher<radar_interfaces::msg::RadarMap>::SharedPtr pub_official_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reload_;

    // 矩阵与物理参数
    cv::Mat K_, D_, R_inv_, T_, base_map_; 
    float field_length_ = 28.0f, field_width_ = 15.0f;
    int map_w_ = 772, map_h_ = 388;

    void handle_reload(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        if (load_all_configs()) {
            response->success = true;
            response->message = "重载成功";
            RCLCPP_INFO(this->get_logger(), "收到重载信号，标定参数已热更新");
        } else {
            response->success = false;
            response->message = "重载失败";
        }
    }

    // 数学核心：Y=0 平面求交投影
    cv::Point3f project_pixel_to_ground_y0(cv::Point2f pixel) {
        std::vector<cv::Point2f> src_pts = { pixel }, dst_pts;
        cv::undistortPoints(src_pts, dst_pts, K_, D_);
        
        cv::Mat P_c = (cv::Mat_<double>(3, 1) << dst_pts[0].x, dst_pts[0].y, 1.0);
        cv::Mat Ray_world = R_inv_ * P_c;
        cv::Mat Cam_world = -R_inv_ * T_;

        double dy = Ray_world.at<double>(1);
        double oy = Cam_world.at<double>(1);

        if (std::abs(dy) < 1e-6) return cv::Point3f(0, 0, 0);

        double t = -oy / dy;
        return cv::Point3f(Cam_world.at<double>(0) + t * Ray_world.at<double>(0), 
                           0.0f, 
                           Cam_world.at<double>(2) + t * Ray_world.at<double>(2));
    }

    void results_callback(const radar_interfaces::msg::DetectResults::SharedPtr msg)
    {
        if (K_.empty() || base_map_.empty()) return;

        cv::Mat draw_map = base_map_.clone();
        
        // 实例化 RadarMap，C++ 默认将内部的 float32[6] 初始化为 0.0
        radar_interfaces::msg::RadarMap official_msg;
        official_msg.header = msg->header;

        for (const auto& res : msg->results) {
            // 1. PnP 物理坐标解算 (截断至场地边界内)
            cv::Point3f pnp_pt = project_pixel_to_ground_y0(cv::Point2f(res.x, res.y));
            float official_x = std::max(0.0f, std::min(pnp_pt.z + field_length_ / 2.0f, field_length_));
            float official_y = std::max(0.0f, std::min(pnp_pt.x + field_width_ / 2.0f, field_width_));

            // 2. 字符串解析与靶向定长数组映射
            std::string label = res.number; 
            if (label.length() >= 2) {
                char team = label[0];
                char id_char = label[1];
                int target_idx = -1;

                if (id_char >= '1' && id_char <= '4') {
                    target_idx = id_char - '1'; // 映射至数组 0, 1, 2, 3
                } else if (id_char == '7') {
                    target_idx = 5;             // 映射至数组 5 (跳过 5号位索引4)
                }

                if (target_idx != -1) {
                    if (team == 'B' || team == 'b') {
                        official_msg.blue_x[target_idx] = official_x;
                        official_msg.blue_y[target_idx] = official_y;
                    } else if (team == 'R' || team == 'r') {
                        official_msg.red_x[target_idx] = official_x;
                        official_msg.red_y[target_idx] = official_y;
                    }
                }
            }

            // 3. 像素系映射 (用于 Qt 绘图)
            float norm_x = official_x / field_length_;
            float norm_y = 1.0f - (official_y / field_width_); 
            int px = (int)(norm_x * map_w_);
            int py = (int)(norm_y * map_h_);
            draw_on_map(draw_map, px, py, res.number); 
        }

        // 4. 发布 RadarMap 协议坐标数据
        pub_official_->publish(official_msg);

        // 5. 顺时针旋转 90 度并发布小地图图像
        cv::Mat vertical_map;
        cv::rotate(draw_map, vertical_map, cv::ROTATE_90_CLOCKWISE);
        auto out_img_msg = cv_bridge::CvImage(msg->header, "bgr8", vertical_map).toImageMsg();
        pub_map_img_->publish(*out_img_msg);
    }

    void draw_on_map(cv::Mat& img, int px, int py, const std::string& label)
    {
        px = std::max(0, std::min(px, map_w_ - 1));
        py = std::max(0, std::min(py, map_h_ - 1));
        
        cv::Scalar color(0, 255, 0); 
        if (!label.empty()) {
            if (label[0] == 'R' || label[0] == 'r') color = cv::Scalar(0, 0, 255); 
            else if (label[0] == 'B' || label[0] == 'b') color = cv::Scalar(255, 0, 0); 
        }

        cv::circle(img, cv::Point(px, py), 10, cv::Scalar(255, 255, 255), -1); 
        cv::circle(img, cv::Point(px, py), 8, color, -1);
        cv::putText(img, label, cv::Point(px+12, py+5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
        cv::putText(img, label, cv::Point(px+12, py+5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1);
    }

    bool load_all_configs()
    {
        try {
            // 解析相机与 PnP 参数
            YAML::Node cam_cfg = YAML::LoadFile(camera_yaml_path_);
            auto load_v = [&](std::string k) { return cam_cfg["camera"][k].as<std::vector<double>>(); };
            K_ = cv::Mat(3, 3, CV_64F, load_v("K").data()).clone();
            D_ = cv::Mat(1, 5, CV_64F, load_v("dist").data()).clone();
            cv::Mat rvec = cv::Mat(3, 1, CV_64F, load_v("rvec").data()).clone();
            T_ = cv::Mat(3, 1, CV_64F, load_v("tvec").data()).clone();

            cv::Mat R; cv::Rodrigues(rvec, R); R_inv_ = R.t();

            // 解析场地物理尺寸与分辨率
            YAML::Node map_cfg = YAML::LoadFile(map_yaml_path_);
            auto rs = map_cfg["race_size"].as<std::vector<double>>();
            auto ms = map_cfg["map_size"].as<std::vector<int>>();
            field_length_ = rs[0]; field_width_ = rs[1];
            map_w_ = ms[0]; map_h_ = ms[1];

            // 读取并缩放底图
            base_map_ = cv::imread(map_image_path_);
            if(base_map_.empty()) return false;
            cv::resize(base_map_, base_map_, cv::Size(map_w_, map_h_));

            return true;
        } catch (...) { return false; }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::MapComponent)