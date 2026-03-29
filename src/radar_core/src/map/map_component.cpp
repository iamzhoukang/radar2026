#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>

#include "map/transform.hpp"
#include "map/raycaster.hpp"

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
        this->declare_parameter<std::string>("mesh_path", "/home/lzhros/Code/RadarStation/config/map/field_mesh.ply");
        
        // 决定雷达站视角的红蓝方阵营开关
        this->declare_parameter<bool>("is_blue_team", true);
        
        camera_yaml_path_ = this->get_parameter("camera_yaml").as_string();
        map_yaml_path_ = this->get_parameter("map_yaml").as_string();
        map_image_path_ = this->get_parameter("map_image").as_string();
        mesh_path_ = this->get_parameter("mesh_path").as_string();
        is_blue_team_ = this->get_parameter("is_blue_team").as_bool();

        // 2. 加载配置与物理网格
        if (!load_all_configs()) {
            RCLCPP_ERROR(this->get_logger(), "启动失败：参数文件、底图或 3D 网格加载错误！");
        } else {
            RCLCPP_INFO(this->get_logger(), "地图就绪！当前阵营: %s", is_blue_team_ ? "蓝方" : "红方");
        }

        // 3. 注册 ROS 通信
        sub_results_ = this->create_subscription<radar_interfaces::msg::DetectResults>(
            "detector/results", 10,
            std::bind(&MapComponent::results_callback, this, std::placeholders::_1)
        );
        pub_map_img_ = this->create_publisher<sensor_msgs::msg::Image>("map/image", 10);
        pub_official_ = this->create_publisher<radar_interfaces::msg::RadarMap>("map/official_data", 10);
        srv_reload_ = this->create_service<std_srvs::srv::Trigger>(
            "map/reload_config",
            std::bind(&MapComponent::handle_reload, this, std::placeholders::_1, std::placeholders::_2)
        );
    }

private:
    std::string camera_yaml_path_, map_yaml_path_, map_image_path_, mesh_path_;
    bool is_blue_team_ = true;
    
    rclcpp::Subscription<radar_interfaces::msg::DetectResults>::SharedPtr sub_results_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_map_img_;
    rclcpp::Publisher<radar_interfaces::msg::RadarMap>::SharedPtr pub_official_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reload_;

    cv::Mat K_, D_, R_inv_, T_, base_map_; 
    float field_length_ = 28.0f, field_width_ = 15.0f;
    int map_w_ = 772, map_h_ = 388;

    // 【核心解耦】：物理引擎实例化，完全隐藏了 Open3D 的实现细节
    utils::Raycaster raycaster_;

    void handle_reload(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        // 支持热更新阵营配置
        is_blue_team_ = this->get_parameter("is_blue_team").as_bool();
        if (load_all_configs()) {
            response->success = true;
            response->message = "重载成功";
            RCLCPP_INFO(this->get_logger(), "热重载完成。当前阵营: %s", is_blue_team_ ? "蓝方" : "红方");
        } else {
            response->success = false;
        }
    }

    void results_callback(const radar_interfaces::msg::DetectResults::SharedPtr msg)
    {
        if (K_.empty() || base_map_.empty()) return;

        cv::Mat draw_map = base_map_.clone();
        radar_interfaces::msg::RadarMap official_msg;
        official_msg.header = msg->header;

        for (const auto& res : msg->results) {
            
            // 1. 扔给黑盒：从 2D 像素打出射线，索取物理世界的绝对 3D 坐标
            cv::Point3f mesh_pt = raycaster_.pixelToWorld(cv::Point2f(res.x, res.y), K_, D_, R_inv_, T_);
            
            // 2. 扔给工具箱：解析兵种标签
            char team;
            int target_idx;
            if (utils::parseTargetLabel(res.number, team, target_idx)) {
                
                // 3. 扔给工具箱：应用阵营翻转和极值保护，获取裁判系统安全坐标
                cv::Point2f official_pt = utils::convertToOfficialMap(mesh_pt, field_length_, field_width_, is_blue_team_);
                
                // 4. 封装数据包
                if (team == 'B' || team == 'b') {
                    official_msg.blue_x[target_idx] = official_pt.x;
                    official_msg.blue_y[target_idx] = official_pt.y;
                } else if (team == 'R' || team == 'r') {
                    official_msg.red_x[target_idx] = official_pt.x;
                    official_msg.red_y[target_idx] = official_pt.y;
                }

                // 5. 将实际物理坐标反算回供 UI 渲染的像素图坐标
                float norm_x = official_pt.x / field_length_;
                float norm_y = 1.0f - (official_pt.y / field_width_); 
                int px = (int)(norm_x * map_w_);
                int py = (int)(norm_y * map_h_);
                draw_on_map(draw_map, px, py, res.number); 
            }
        }

        pub_official_->publish(official_msg);
        
        cv::Mat vertical_map;
        cv::rotate(draw_map, vertical_map, cv::ROTATE_90_CLOCKWISE);
        auto out_img_msg = cv_bridge::CvImage(msg->header, "bgr8", vertical_map).toImageMsg();
        pub_map_img_->publish(*out_img_msg);
    }

    void draw_on_map(cv::Mat& img, int px, int py, const std::string& label)
    {
        // 渲染逻辑保持你的原有代码不变
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
            YAML::Node cam_cfg = YAML::LoadFile(camera_yaml_path_);
            auto load_v = [&](std::string k) { return cam_cfg["camera"][k].as<std::vector<double>>(); };
            K_ = cv::Mat(3, 3, CV_64F, load_v("K").data()).clone();
            D_ = cv::Mat(1, 5, CV_64F, load_v("dist").data()).clone();
            cv::Mat rvec = cv::Mat(3, 1, CV_64F, load_v("rvec").data()).clone();
            T_ = cv::Mat(3, 1, CV_64F, load_v("tvec").data()).clone();

            cv::Mat R; cv::Rodrigues(rvec, R); R_inv_ = R.t();

            YAML::Node map_cfg = YAML::LoadFile(map_yaml_path_);
            auto rs = map_cfg["race_size"].as<std::vector<double>>();
            auto ms = map_cfg["map_size"].as<std::vector<int>>();
            field_length_ = rs[0]; field_width_ = rs[1];
            map_w_ = ms[0]; map_h_ = ms[1];

            base_map_ = cv::imread(map_image_path_);
            if(base_map_.empty()) return false;
            cv::resize(base_map_, base_map_, cv::Size(map_w_, map_h_));

            // 调用物理引擎加载网格（不管是 PLY, OBJ 还是 STL 都可以）
            if (!raycaster_.loadMesh(mesh_path_)) {
                RCLCPP_WARN(this->get_logger(), "网格文件 (%s) 加载失败，触发 Y=0 平面降级保护模式", mesh_path_.c_str());
            }

            return true;
        } catch (...) { return false; }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::MapComponent)