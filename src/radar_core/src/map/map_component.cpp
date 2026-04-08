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
#include "radar_interfaces/msg/lidar_cluster_results.hpp" 
#include "std_srvs/srv/trigger.hpp"

namespace radar_core {

class MapComponent : public rclcpp::Node
{
public:
    explicit MapComponent(const rclcpp::NodeOptions & options) : Node("map_component", options)
    {
        this->declare_parameter<std::string>("camera_yaml", "/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml");
        this->declare_parameter<std::string>("map_yaml", "/home/lzhros/Code/RadarStation/config/map/field_image.yaml");
        this->declare_parameter<std::string>("map_image", "/home/lzhros/Code/RadarStation/config/map/field_image.png");
        this->declare_parameter<std::string>("mesh_path", "/home/lzhros/Code/RadarStation/config/map/field_mesh.ply");
        this->declare_parameter<bool>("is_blue_team", true);
        
        camera_yaml_path_ = this->get_parameter("camera_yaml").as_string();
        map_yaml_path_ = this->get_parameter("map_yaml").as_string();
        map_image_path_ = this->get_parameter("map_image").as_string();
        mesh_path_ = this->get_parameter("mesh_path").as_string();
        is_blue_team_ = this->get_parameter("is_blue_team").as_bool();

        if (!load_all_configs()) {
            RCLCPP_ERROR(this->get_logger(), "启动失败：参数文件、底图或 3D 网格加载错误！");
        } else {
            RCLCPP_INFO(this->get_logger(), "\033[1;32m多模态小地图就绪！当前阵营: %s\033[0m", is_blue_team_ ? "蓝方" : "红方");
        }

        sub_results_ = this->create_subscription<radar_interfaces::msg::DetectResults>(
            "detector/results", 10,
            std::bind(&MapComponent::results_callback, this, std::placeholders::_1)
        );

        // 订阅雷达聚类结果
        sub_radar_ = this->create_subscription<radar_interfaces::msg::LidarClusterResults>(
            "/radar/lidar_clusters", 10,
            std::bind(&MapComponent::radar_callback, this, std::placeholders::_1)
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
    rclcpp::Subscription<radar_interfaces::msg::LidarClusterResults>::SharedPtr sub_radar_; // 雷达订阅器
    radar_interfaces::msg::LidarClusterResults::SharedPtr latest_radar_msg_;              // 雷达最新缓存

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_map_img_;
    rclcpp::Publisher<radar_interfaces::msg::RadarMap>::SharedPtr pub_official_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reload_;

    cv::Mat K_, D_, R_inv_, T_, base_map_, rvec_; 
    float field_length_ = 28.0f, field_width_ = 15.0f;
    int map_w_ = 772, map_h_ = 388;

    utils::Raycaster raycaster_;

    void handle_reload(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        is_blue_team_ = this->get_parameter("is_blue_team").as_bool();
        if (load_all_configs()) {
            response->success = true;
            response->message = "重载成功";
            RCLCPP_INFO(this->get_logger(), "热重载完成。当前阵营: %s", is_blue_team_ ? "蓝方" : "红方");
        } else {
            response->success = false;
        }
    }

    // 雷达缓存回调
    void radar_callback(const radar_interfaces::msg::LidarClusterResults::SharedPtr msg)
    {
        latest_radar_msg_ = msg;
    }

    void results_callback(const radar_interfaces::msg::DetectResults::SharedPtr msg)
    {
        if (K_.empty() || base_map_.empty()) return;

        cv::Mat draw_map = base_map_.clone();
        radar_interfaces::msg::RadarMap official_msg;
        official_msg.header = msg->header;

        for (const auto& res : msg->results) {
            
            cv::Point3f mesh_pt;

            // ==========================================
            // 核心：多模态空间解算分流
            // ==========================================
            if (res.number == "Drone") {
                // 【模式 A：空对空】3D 雷达坐标 -> 2D 像素投影匹配
                if (!latest_radar_msg_ || latest_radar_msg_->clusters.empty()) continue; // 没有雷达则无法定位天空

                float min_dist = 1e9;
                bool matched = false;
                cv::Point3f best_pt;

                for (const auto& cluster : latest_radar_msg_->clusters) {
                    // 我们底层雷达发来的 Z 已经是离地绝对高度，滤除地面目标 (比如高度低于 1.0m 的)
                    if (cluster.center.z < 1.0) continue; 

                    std::vector<cv::Point3f> obj_pts = { cv::Point3f(cluster.center.x, cluster.center.y, cluster.center.z) };
                    std::vector<cv::Point2f> img_pts;
                    
                    // 神奇的数学：用相机的内参和外参，将 3D 点“拍扁”到照片像素上！
                    cv::projectPoints(obj_pts, rvec_, T_, K_, D_, img_pts);

                    // 检查拍扁后的像素点，与 YOLO 识别框中心的距离
                    float dx = img_pts[0].x - res.x;
                    float dy = img_pts[0].y - res.y;
                    float dist = std::sqrt(dx*dx + dy*dy);

                    // 如果误差在 200 个像素点以内，且是目前最小的，即视为锁定目标！
                    if (dist < min_dist && dist < 200.0f) {
                        min_dist = dist;
                        best_pt = obj_pts[0];
                        matched = true;
                    }
                }

                if (!matched) continue; // 雷达没扫到或对应不上，放弃发送此目标
                mesh_pt = best_pt;      // 拿走真实 3D 物理坐标！

            } else {
                // 【模式 B：地对地】2D 像素 -> 3D 模型光线追踪求交 (你之前的完美逻辑)
                mesh_pt = raycaster_.pixelToWorld(cv::Point2f(res.x, res.y), K_, D_, R_inv_, T_);
            }

            // ==========================================
            // 封装与发送
            // ==========================================
            char team;
            int target_idx;
            if (utils::parseTargetLabel(res.number, team, target_idx)) {
                
                cv::Point2f official_pt = utils::convertToOfficialMap(mesh_pt, field_length_, field_width_, is_blue_team_);
                
                if (team == 'B' || team == 'b') {
                    official_msg.blue_x[target_idx] = official_pt.x;
                    official_msg.blue_y[target_idx] = official_pt.y;
                } else if (team == 'R' || team == 'r') {
                    official_msg.red_x[target_idx] = official_pt.x;
                    official_msg.red_y[target_idx] = official_pt.y;
                } else if (team == 'A') {
                    // 无人机数据：同时填充双方的 `[4]` 号索引空位
                    official_msg.blue_x[target_idx] = official_pt.x;
                    official_msg.blue_y[target_idx] = official_pt.y;
                    official_msg.red_x[target_idx] = official_pt.x;
                    official_msg.red_y[target_idx] = official_pt.y;
                }

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
        px = std::max(0, std::min(px, map_w_ - 1));
        py = std::max(0, std::min(py, map_h_ - 1));
        cv::Scalar color(0, 255, 0); 
        if (!label.empty()) {
            if (label[0] == 'R' || label[0] == 'r') color = cv::Scalar(0, 0, 255); 
            else if (label[0] == 'B' || label[0] == 'b') color = cv::Scalar(255, 0, 0); 
            else if (label == "Drone") color = cv::Scalar(0, 165, 255); // 无人机画橙色
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

            rvec_ = rvec.clone(); 
            cv::Mat R; cv::Rodrigues(rvec, R); R_inv_ = R.t();

            YAML::Node map_cfg = YAML::LoadFile(map_yaml_path_);
            auto rs = map_cfg["race_size"].as<std::vector<double>>();
            auto ms = map_cfg["map_size"].as<std::vector<int>>();
            field_length_ = rs[0]; field_width_ = rs[1];
            map_w_ = ms[0]; map_h_ = ms[1];

            base_map_ = cv::imread(map_image_path_);
            if(base_map_.empty()) return false;
            cv::resize(base_map_, base_map_, cv::Size(map_w_, map_h_));

            if (!raycaster_.loadMesh(mesh_path_)) {
                RCLCPP_WARN(this->get_logger(), "网格文件加载失败，触发 Y=0 平面降级保护模式");
            }

            return true;
        } catch (...) { return false; }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::MapComponent)