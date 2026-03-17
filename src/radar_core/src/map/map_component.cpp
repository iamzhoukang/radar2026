#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory> // 【新增】：用于 std::unique_ptr

// ==========================================
// 【Open3D 核心与显式模块头文件】
// ==========================================
#include <open3d/Open3D.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/core/Tensor.h>

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
        
        // 【新增参数】：3D 场地网格文件路径
        this->declare_parameter<std::string>("mesh_path", "/home/lzhros/Code/RadarStation/config/map/field_mesh.ply");
        
        camera_yaml_path_ = this->get_parameter("camera_yaml").as_string();
        map_yaml_path_ = this->get_parameter("map_yaml").as_string();
        map_image_path_ = this->get_parameter("map_image").as_string();
        mesh_path_ = this->get_parameter("mesh_path").as_string();

        // 2. 加载参数与 3D 模型
        if (!load_all_configs()) {
            RCLCPP_ERROR(this->get_logger(), "启动失败：参数文件、地图图片或 3D 网格加载错误！");
        } else {
            RCLCPP_INFO(this->get_logger(), "小地图解算节点启动成功 (已启用 Open3D 射线物理碰撞)");
        }

        // 3. 订阅与发布
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
    rclcpp::Subscription<radar_interfaces::msg::DetectResults>::SharedPtr sub_results_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_map_img_;
    rclcpp::Publisher<radar_interfaces::msg::RadarMap>::SharedPtr pub_official_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reload_;

    cv::Mat K_, D_, R_inv_, T_, base_map_; 
    float field_length_ = 28.0f, field_width_ = 15.0f;
    int map_w_ = 772, map_h_ = 388;

    // ==========================================
    // 【核心修复】：使用独占智能指针管理场景，绕过拷贝限制
    // ==========================================
    std::unique_ptr<open3d::t::geometry::RaycastingScene> scene_;

    void handle_reload(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        if (load_all_configs()) {
            response->success = true;
            response->message = "重载成功";
            RCLCPP_INFO(this->get_logger(), "收到重载信号，标定参数与 3D 网格已热更新");
        } else {
            response->success = false;
            response->message = "重载失败";
        }
    }

    // ==========================================
    // 核心算法：基于 Open3D 的物理射线碰撞检测
    // ==========================================
    cv::Point3f project_pixel_to_mesh(cv::Point2f pixel) {
        // 1. 去畸变
        std::vector<cv::Point2f> src_pts = { pixel }, dst_pts;
        cv::undistortPoints(src_pts, dst_pts, K_, D_);
        
        // 2. 计算方向与光心
        cv::Mat P_c = (cv::Mat_<double>(3, 1) << dst_pts[0].x, dst_pts[0].y, 1.0);
        cv::Mat Ray_world = R_inv_ * P_c;
        cv::Mat Cam_world = -R_inv_ * T_;

        float ox = Cam_world.at<double>(0);
        float oy = Cam_world.at<double>(1);
        float oz = Cam_world.at<double>(2);
        
        float dx = Ray_world.at<double>(0);
        float dy = Ray_world.at<double>(1);
        float dz = Ray_world.at<double>(2);

        // ==========================================
        // 【关键防御】：如果网格加载失败，强制走保底算法
        // ==========================================
        if (!scene_) {
            RCLCPP_WARN_ONCE(this->get_logger(), "物理场景未初始化，触发 Y=0 降级保护");
            if (std::abs(dy) < 1e-6) return cv::Point3f(0, 0, 0);
            double t_fallback = -oy / dy;
            return cv::Point3f(ox + t_fallback * dx, 0.0f, oz + t_fallback * dz);
        }

        // 3. 构造 Open3D Tensor 射线参数: [ox, oy, oz, dx, dy, dz]
        std::vector<float> ray_data = {ox, oy, oz, dx, dy, dz};
        open3d::core::Tensor ray(ray_data, {1, 6}, open3d::core::Dtype::Float32);

        // 4. 执行物理投射 (使用智能指针 -> 访问)
        auto result = scene_->CastRays(ray);
        float t_hit = result["t_hit"].Item<float>();

        // 5. 结果校验与边界处理
        if (std::isinf(t_hit)) {
            // 【保底机制】：若未击中任何 3D 模型表面（如射向天空），退化为 Y=0 平面求解
            if (std::abs(dy) < 1e-6) return cv::Point3f(0, 0, 0);
            double t_fallback = -oy / dy;
            return cv::Point3f(ox + t_fallback * dx, 0.0f, oz + t_fallback * dz);
        }

        // 6. 返回击中高地或平地的真实 3D 坐标
        return cv::Point3f(ox + t_hit * dx, oy + t_hit * dy, oz + t_hit * dz);
    }

    void results_callback(const radar_interfaces::msg::DetectResults::SharedPtr msg)
    {
        if (K_.empty() || base_map_.empty()) return;

        cv::Mat draw_map = base_map_.clone();
        radar_interfaces::msg::RadarMap official_msg;
        official_msg.header = msg->header;

        for (const auto& res : msg->results) {
            // 【执行物理射线投影】
            cv::Point3f mesh_pt = project_pixel_to_mesh(cv::Point2f(res.x, res.y));
            
            float official_x = std::max(0.0f, std::min(mesh_pt.z + field_length_ / 2.0f, field_length_));
            float official_y = std::max(0.0f, std::min(mesh_pt.x + field_width_ / 2.0f, field_width_));

            std::string label = res.number; 
            if (label.length() >= 2) {
                char team = label[0];
                char id_char = label[1];
                int target_idx = -1;

                if (id_char >= '1' && id_char <= '4') target_idx = id_char - '1'; 
                else if (id_char == '7') target_idx = 5;             

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

            float norm_x = official_x / field_length_;
            float norm_y = 1.0f - (official_y / field_width_); 
            int px = (int)(norm_x * map_w_);
            int py = (int)(norm_y * map_h_);
            draw_on_map(draw_map, px, py, res.number); 
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
        }
        cv::circle(img, cv::Point(px, py), 10, cv::Scalar(255, 255, 255), -1); 
        cv::circle(img, cv::Point(px, py), 8, color, -1);
        cv::putText(img, label, cv::Point(px+12, py+5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 3);
        cv::putText(img, label, cv::Point(px+12, py+5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1);
    }

    bool load_all_configs()
    {
        try {
            // 1. 解析相机与 PnP 参数
            YAML::Node cam_cfg = YAML::LoadFile(camera_yaml_path_);
            auto load_v = [&](std::string k) { return cam_cfg["camera"][k].as<std::vector<double>>(); };
            K_ = cv::Mat(3, 3, CV_64F, load_v("K").data()).clone();
            D_ = cv::Mat(1, 5, CV_64F, load_v("dist").data()).clone();
            cv::Mat rvec = cv::Mat(3, 1, CV_64F, load_v("rvec").data()).clone();
            T_ = cv::Mat(3, 1, CV_64F, load_v("tvec").data()).clone();

            cv::Mat R; cv::Rodrigues(rvec, R); R_inv_ = R.t();

            // 2. 解析场地物理尺寸与分辨率
            YAML::Node map_cfg = YAML::LoadFile(map_yaml_path_);
            auto rs = map_cfg["race_size"].as<std::vector<double>>();
            auto ms = map_cfg["map_size"].as<std::vector<int>>();
            field_length_ = rs[0]; field_width_ = rs[1];
            map_w_ = ms[0]; map_h_ = ms[1];

            // 3. 读取底图
            base_map_ = cv::imread(map_image_path_);
            if(base_map_.empty()) return false;
            cv::resize(base_map_, base_map_, cv::Size(map_w_, map_h_));

            // ==========================================
            // 4. 加载 Legacy 网格并转换为 Tensor 网格
            // ==========================================
            open3d::geometry::TriangleMesh legacy_mesh;
            
            // 使用传统 IO API 读取 .ply 或 .obj
            if (!open3d::io::ReadTriangleMesh(mesh_path_, legacy_mesh)) {
                RCLCPP_WARN(this->get_logger(), "未找到或无法解析 3D 场地网格 (%s)，退化为纯平面模式", mesh_path_.c_str());
                scene_.reset(); // 清空智能指针，触发降级
            } else {
                // 将传统的 Legacy Mesh 转换为用于高速光线追踪的 Tensor Mesh
                open3d::t::geometry::TriangleMesh tensor_mesh = 
                    open3d::t::geometry::TriangleMesh::FromLegacy(legacy_mesh);
                
                // 【核心修复】：使用 std::make_unique 安全重构物理引擎，杜绝双重释放
                scene_ = std::make_unique<open3d::t::geometry::RaycastingScene>(); 
                scene_->AddTriangles(tensor_mesh);
                
                RCLCPP_INFO(this->get_logger(), "3D 物理碰撞网格加载成功！(Legacy -> Tensor 转换完成)");
            }

            return true;
        } catch (...) { return false; }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::MapComponent)