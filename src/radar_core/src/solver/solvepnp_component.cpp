#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <vector>
#include <string>

#include <std_srvs/srv/trigger.hpp>

namespace radar_core {

// 交互上下文 
struct MouseContext {
    cv::Mat raw_img;        // 5K 原始高清图
    cv::Mat img;            // 缩放/裁切后显示的图片
    std::vector<cv::Point2f> points; // 原始分辨率的点击点
    float scale = 1.0f;     
    bool is_zoomed = false; 
    cv::Rect zoom_roi;      
    std::string window_name;
};

// 全局鼠标回调
void on_mouse(int event, int x, int y, int flags, void* userdata) {
    MouseContext* ctx = (MouseContext*)userdata;
    if (ctx->raw_img.empty()) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (!ctx->is_zoomed) {
            if (ctx->points.size() >= 6) {
                RCLCPP_WARN(rclcpp::get_logger("SolvePnP"), "点满了！请按 's' 保存，或 'c' 清空。");
                return;
            }
            float raw_center_x = x / ctx->scale;
            float raw_center_y = y / ctx->scale;

            int roi_w = 600, roi_h = 600;
            int roi_x = std::max(0, (int)raw_center_x - roi_w / 2);
            int roi_y = std::max(0, (int)raw_center_y - roi_h / 2);
            roi_x = std::min(roi_x, ctx->raw_img.cols - roi_w);
            roi_y = std::min(roi_y, ctx->raw_img.rows - roi_h);

            ctx->zoom_roi = cv::Rect(roi_x, roi_y, roi_w, roi_h);
            ctx->is_zoomed = true;
            ctx->img = ctx->raw_img(ctx->zoom_roi).clone();
            cv::putText(ctx->img, "[ZOOM_MODE]", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::imshow(ctx->window_name, ctx->img);
        } else {
            float final_x = ctx->zoom_roi.x + x;
            float final_y = ctx->zoom_roi.y + y;
            ctx->points.push_back(cv::Point2f(final_x, final_y));
            RCLCPP_INFO(rclcpp::get_logger("SolvePnP"), " 选中第 %ld 个点: (%.1f, %.1f)", ctx->points.size(), final_x, final_y);
            ctx->is_zoomed = false;
        }
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        if (ctx->is_zoomed) ctx->is_zoomed = false;
        else if (!ctx->points.empty()) ctx->points.pop_back();
    }
}

class SolvePnPComponent : public rclcpp::Node
{
public:
    explicit SolvePnPComponent(const rclcpp::NodeOptions & options) : Node("solvepnp_component", options)
    {
        // 1. 严格对齐新的路径结构
        this->declare_parameter<std::string>("config_path", "/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml");
        this->declare_parameter<std::string>("keypoint_path", "/home/lzhros/Code/RadarStation/config/solver/keypoint_6.txt");
        config_file_path_ = this->get_parameter("config_path").as_string();
        keypoint_file_path_ = this->get_parameter("keypoint_path").as_string();

        if (!load_camera_intrinsics() || !load_world_points()) {
            RCLCPP_ERROR(this->get_logger(), "初始化失败，请检查 yaml 和 txt 路径！");
            return; 
        }

        mouse_ctx_.window_name = window_name_;

        // 2. 客户端与服务端
        client_map_reload_ = this->create_client<std_srvs::srv::Trigger>("map/reload_config");
        srv_start_calib_ = this->create_service<std_srvs::srv::Trigger>(
            "solvepnp/start",
            std::bind(&SolvePnPComponent::handle_start, this, std::placeholders::_1, std::placeholders::_2)
        );

        // 3. UI 刷新定时器 (30Hz，脱离图像回调独立运行)
        ui_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&SolvePnPComponent::ui_loop, this)
        );

        RCLCPP_INFO(this->get_logger(), "单帧标定组件就绪... 等待 'solvepnp/start' 触发");
    }

private:
    std::string config_file_path_, keypoint_file_path_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_map_reload_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_start_calib_;
    rclcpp::TimerBase::SharedPtr ui_timer_;

    cv::Mat K_, D_;
    std::vector<cv::Point3f> world_points_;
    MouseContext mouse_ctx_;
    bool is_calibrating_ = false;
    const std::string window_name_ = "Calibration Tool";

    // --- 触发标定 ---
    void handle_start(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        if (is_calibrating_) {
            response->success = false;
            response->message = "Already calibrating";
            return;
        }
        
        is_calibrating_ = true;
        mouse_ctx_.points.clear();
        mouse_ctx_.is_zoomed = false; 
        mouse_ctx_.raw_img.release();

        cv::namedWindow(window_name_);
        cv::setMouseCallback(window_name_, on_mouse, &mouse_ctx_);
        
        // 动态开启订阅，抓取唯一一帧 5K 图像
        RCLCPP_INFO(this->get_logger(), "正在向 cs200_topic 请求一张 5K 原图...");
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 1, 
            std::bind(&SolvePnPComponent::image_callback, this, std::placeholders::_1)
        );

        response->success = true;
        response->message = "Start";
    }

    // --- 抓取单帧图像 ---
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!is_calibrating_ || !mouse_ctx_.raw_img.empty()) return;

        try { 
            mouse_ctx_.raw_img = cv_bridge::toCvCopy(msg, "bgr8")->image; 
        } catch (...) { return; }

        RCLCPP_INFO(this->get_logger(), "成功截取原图！尺寸: %dx%d", mouse_ctx_.raw_img.cols, mouse_ctx_.raw_img.rows);
        RCLCPP_INFO(this->get_logger(), "1. 左键点击 -> 进入放大镜 | 2. 放大镜中点击 -> 确认 | 3. 右键 -> 撤销");

        // 【断开订阅】释放零拷贝总线
        sub_.reset(); 
    }

    // --- UI 刷新循环 ---
    void ui_loop()
    {
        if (!is_calibrating_ || mouse_ctx_.raw_img.empty()) return;

        int key = cv::waitKey(1) & 0xFF;

        // 放大模式仅响应按键
        if (mouse_ctx_.is_zoomed) {
            if (key == 'c' || key == 'C') {
                 mouse_ctx_.points.clear();
                 mouse_ctx_.is_zoomed = false;
            }
            return;
        }

        // 全景浏览模式 (降采样预览)
        float target_width = 1280.0f;
        mouse_ctx_.scale = mouse_ctx_.raw_img.cols > target_width ? target_width / (float)mouse_ctx_.raw_img.cols : 1.0f;
        cv::resize(mouse_ctx_.raw_img, mouse_ctx_.img, cv::Size(), mouse_ctx_.scale, mouse_ctx_.scale);

        for (size_t i = 0; i < mouse_ctx_.points.size(); ++i) {
            cv::Point2f display_pt = mouse_ctx_.points[i] * mouse_ctx_.scale;
            cv::circle(mouse_ctx_.img, display_pt, 4, cv::Scalar(0, 0, 255), -1); 
            cv::putText(mouse_ctx_.img, std::to_string(i + 1), display_pt + cv::Point2f(5, -5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow(window_name_, mouse_ctx_.img);
        
        if (key == 's' || key == 'S') perform_solve_pnp();
        if (key == 'c' || key == 'C') mouse_ctx_.points.clear();
    }

    // --- 核心解算 ---
    void perform_solve_pnp()
    {
        if (mouse_ctx_.points.size() != 6) {
            RCLCPP_WARN(this->get_logger(), "点数不足6个!当前点数:%ld", mouse_ctx_.points.size()); return;
        }
        cv::Mat rvec, tvec;
        if (!cv::solvePnP(world_points_, mouse_ctx_.points, K_, D_, rvec, tvec)) {
            RCLCPP_ERROR(this->get_logger(), "解算失败！"); return;
        }

        double avg_error = calculate_reprojection_error(rvec, tvec);
        RCLCPP_INFO(this->get_logger(), "平均重投影误差: %.2f px", avg_error);

        if (avg_error > 30.0) {
            RCLCPP_ERROR(this->get_logger(), "误差过大！标定无效。"); return; 
        }

        show_reprojection_feedback(rvec, tvec);
        save_to_yaml(rvec, tvec);
        trigger_map_reload();
        finish_calibration();
    }

    double calculate_reprojection_error(const cv::Mat& rvec, const cv::Mat& tvec)
    {
        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(world_points_, rvec, tvec, K_, D_, projected_points);
        double total_error = 0.0;
        for (size_t i = 0; i < world_points_.size(); ++i) {
            total_error += cv::norm(mouse_ctx_.points[i] - projected_points[i]);
        }
        return total_error / world_points_.size();
    }

    void show_reprojection_feedback(const cv::Mat& rvec, const cv::Mat& tvec)
    {
        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(world_points_, rvec, tvec, K_, D_, projected_points);
        if (mouse_ctx_.is_zoomed) mouse_ctx_.is_zoomed = false;
        
        cv::resize(mouse_ctx_.raw_img, mouse_ctx_.img, cv::Size(), mouse_ctx_.scale, mouse_ctx_.scale);

        for (const auto& pt : projected_points) {
            cv::circle(mouse_ctx_.img, pt * mouse_ctx_.scale, 8, cv::Scalar(0, 255, 0), 2);
        }
        cv::putText(mouse_ctx_.img, "Saved! Error: " + std::to_string(calculate_reprojection_error(rvec, tvec)).substr(0, 4), 
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::imshow(window_name_, mouse_ctx_.img);
        cv::waitKey(1500); 
    }

    void finish_calibration() {
        cv::destroyWindow(window_name_);
        is_calibrating_ = false;
        for(int i = 0; i < 5; ++i) {
            cv::waitKey(10);
        }
        mouse_ctx_.raw_img.release(); 
        RCLCPP_INFO(this->get_logger(), "标定结束，窗口已销毁，等待下一次触发。");
    }

    void trigger_map_reload() {
        if (!client_map_reload_->wait_for_service(std::chrono::milliseconds(200))) {
            RCLCPP_WARN(this->get_logger(), "Map Node 未响应，仅保存了 yaml 文件。");
            return;
        }
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        client_map_reload_->async_send_request(req);
        RCLCPP_INFO(this->get_logger(), "已发送热重载信号至 MapComponent。");
    }

    void save_to_yaml(const cv::Mat& rvec, const cv::Mat& tvec) {
        try {
            YAML::Node config = YAML::LoadFile(config_file_path_);
            config["camera"]["rvec"] = std::vector<double>{rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
            config["camera"]["tvec"] = std::vector<double>{tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
            std::ofstream fout(config_file_path_);
            fout << config;
            RCLCPP_INFO(this->get_logger(), "更新后的 rvec/tvec 已写入 cs200_calibration.yaml");
        } catch (...) { RCLCPP_ERROR(this->get_logger(), "写入 YAML 失败！"); }
    }

    bool load_world_points() {
        std::ifstream file(keypoint_file_path_);
        if (!file.is_open()) return false;
        world_points_.clear();
        float x, y, z;
        while (file >> x >> y >> z) world_points_.emplace_back(x, y, z);
        return (world_points_.size() == 6);
    }

    bool load_camera_intrinsics() {
        try {
            YAML::Node config = YAML::LoadFile(config_file_path_);
            auto k = config["camera"]["K"].as<std::vector<double>>();
            auto d = config["camera"]["dist"].as<std::vector<double>>();
            K_ = cv::Mat(3, 3, CV_64F, k.data()).clone();
            D_ = cv::Mat(1, 5, CV_64F, d.data()).clone();
            return true;
        } catch (...) { return false; }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::SolvePnPComponent)