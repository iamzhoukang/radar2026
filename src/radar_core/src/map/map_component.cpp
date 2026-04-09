#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <chrono>

#include "map/transform.hpp"
#include "map/raycaster.hpp"

#include "radar_interfaces/msg/detect_results.hpp" 
#include "radar_interfaces/msg/detect_result.hpp" 
#include "radar_interfaces/msg/radar_map.hpp" 
#include "radar_interfaces/msg/lidar_cluster_results.hpp" 
#include "std_srvs/srv/trigger.hpp"

#include "tracker/cascade_tracker.hpp"

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
            RCLCPP_INFO(this->get_logger(), "\033[1;32m多模态小地图就绪！已挂载 HKUST 级联匹配追踪引擎 | 阵营: %s\033[0m", is_blue_team_ ? "蓝方" : "红方");
        }

        sub_results_ = this->create_subscription<radar_interfaces::msg::DetectResults>(
            "detector/results", 10,
            std::bind(&MapComponent::results_callback, this, std::placeholders::_1)
        );

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
        
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    std::string camera_yaml_path_, map_yaml_path_, map_image_path_, mesh_path_;
    bool is_blue_team_ = true;
    
    rclcpp::Subscription<radar_interfaces::msg::DetectResults>::SharedPtr sub_results_;
    rclcpp::Subscription<radar_interfaces::msg::LidarClusterResults>::SharedPtr sub_radar_; 
    radar_interfaces::msg::LidarClusterResults::SharedPtr latest_radar_msg_;              

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_map_img_;
    rclcpp::Publisher<radar_interfaces::msg::RadarMap>::SharedPtr pub_official_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reload_;

    cv::Mat K_, D_, R_inv_, T_, base_map_, rvec_; 
    float field_length_ = 28.0f, field_width_ = 15.0f;
    int map_w_ = 772, map_h_ = 388;

    utils::Raycaster raycaster_;

    // 【新增】港科大级联追踪引擎实例与时间戳
    tracker::CascadeMatchTracker hkust_tracker_;
    std::chrono::steady_clock::time_point last_time_;

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

    void radar_callback(const radar_interfaces::msg::LidarClusterResults::SharedPtr msg)
    {
        latest_radar_msg_ = msg;
    }

    void results_callback(const radar_interfaces::msg::DetectResults::SharedPtr msg)
    {
        if (K_.empty() || base_map_.empty()) return;

        // 计算时间差 (dt)，用于卡尔曼滤波器的速度推算
        auto current_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time_).count();
        if (dt > 1.0f || dt <= 0.0f) dt = 0.1f; // 保护机制，防止首帧 dt 过大
        last_time_ = current_time;

        cv::Mat draw_map = base_map_.clone();
        radar_interfaces::msg::RadarMap official_msg;
        official_msg.header = msg->header;

        std::vector<tracker::SingleDetectionResult> hku_dets;

        for (const auto& res : msg->results) {
            // ==========================================
            // 【免检通道】无人机专属 3D 雷达投影逻辑
            // ==========================================
            if (res.class_id == -2 || res.number == "Drone") {
                if (!latest_radar_msg_ || latest_radar_msg_->clusters.empty()) continue; 

                float min_dist = 1e9;
                bool matched = false;
                cv::Point3f best_pt;

                for (const auto& cluster : latest_radar_msg_->clusters) {
                    if (cluster.center.z < 1.0) continue; 
                    std::vector<cv::Point3f> obj_pts = { cv::Point3f(cluster.center.x, cluster.center.y, cluster.center.z) };
                    std::vector<cv::Point2f> img_pts;
                    cv::projectPoints(obj_pts, rvec_, T_, K_, D_, img_pts);
                    
                    float dx = img_pts[0].x - res.x;
                    float dy = img_pts[0].y - res.y;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist < min_dist && dist < 200.0f) {
                        min_dist = dist;
                        best_pt = obj_pts[0];
                        matched = true;
                    }
                }

                if (matched) {
                    cv::Point2f off_pt = utils::convertToOfficialMap(best_pt, field_length_, field_width_, is_blue_team_);
                    official_msg.blue_x[4] = official_msg.red_x[4] = off_pt.x;
                    official_msg.blue_y[4] = official_msg.red_y[4] = off_pt.y;
                    
                    int px = (int)((off_pt.x / field_length_) * map_w_);
                    int py = (int)((1.0f - off_pt.y / field_width_) * map_h_);
                    draw_on_map(draw_map, px, py, "Drone");
                }
                continue; // 无人机不进追踪器，直接跳过后续处理！
            }

            // ==========================================
            // 【地面通道】光线追踪解算并装填给 HKUST Tracker
            // ==========================================
            cv::Point3f mesh_pt = raycaster_.pixelToWorld(cv::Point2f(res.x, res.y), K_, D_, R_inv_, T_);
            
            // 将 mesh 坐标立刻转为裁判系统绝对场地坐标 (UWB 坐标)
            cv::Point2f official_pt = utils::convertToOfficialMap(mesh_pt, field_length_, field_width_, is_blue_team_);

            tracker::SingleDetectionResult det;
            det.class_id = res.class_id;
            det.class_conf = res.class_conf;
            det.car_box = cv::Rect2f(res.x - res.width/2.0f, res.y - res.height/2.0f, res.width, res.height);
            det.car_conf = res.car_conf;
            // 将场地绝对坐标装填给 Tracker，便于它算物理运动的 Kalman
            det.pos_3d = cv::Point3f(official_pt.x, official_pt.y, 0.0f); 
            det.bot_id = res.bot_id;

            hku_dets.push_back(det);
        }

        // ==========================================
        // 🌟 一键呼叫港科大级联追踪引擎！
        // ==========================================
        hkust_tracker_.track(hku_dets, dt);

        // ==========================================
        // 提取极度平滑的坐标并发往串口
        // ==========================================
        for (const auto& track : hkust_tracker_.tracks) {
            if (track.is_active) {
                char team;
                int target_idx;
                if (utils::parseTargetLabel(track.name, team, target_idx)) {
                    
                    // 获取 KF 平滑、抗抖动后的 2D 绝对坐标
                    float smooth_x = track.pos_2d_uwb.x;
                    float smooth_y = track.pos_2d_uwb.y;

                    if (team == 'B' || team == 'b') {
                        official_msg.blue_x[target_idx] = smooth_x;
                        official_msg.blue_y[target_idx] = smooth_y;
                    } else if (team == 'R' || team == 'r') {
                        official_msg.red_x[target_idx] = smooth_x;
                        official_msg.red_y[target_idx] = smooth_y;
                    }

                    // 绘制平滑后的小地图像素位置
                    int px = (int)((smooth_x / field_length_) * map_w_);
                    int py = (int)((1.0f - smooth_y / field_width_) * map_h_);
                    draw_on_map(draw_map, px, py, track.name); 
                }
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
            else if (label == "Drone") color = cv::Scalar(0, 165, 255); 
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