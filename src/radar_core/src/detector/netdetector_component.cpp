#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>

#include "radar_interfaces/msg/detect_results.hpp"
#include "radar_interfaces/msg/detect_result.hpp"
#include "radar_interfaces/msg/sentry_tactical.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "utils/model.hpp"
#include "utils/classifier.hpp"
#include "detector/outpost_config_ui.hpp"

namespace radar_core 
{

// 目标类型枚举
enum TargetType { ROBOT = 0, ARMOR = 1, DRONE = 2, OUTPOST = 3 };

struct DetectObject {
    cv::Rect rect;       
    int class_id;        
    std::string label;   
    float class_conf;    
    float car_conf;      
    int bot_id;          
    TargetType type;     
    cv::Point2f center;  
};

class NetDetectorComponent : public rclcpp::Node 
{
public:
    explicit NetDetectorComponent(const rclcpp::NodeOptions & options) 
    : Node("net_detector_component", options),
      system_start_time_(std::chrono::steady_clock::now())
    {
        init_parameters();
        init_models();

        pub_img_ = this->create_publisher<sensor_msgs::msg::Image>("processed_video", 10);
        pub_results_ = this->create_publisher<radar_interfaces::msg::DetectResults>("detector/results", 10);
        pub_tactical_ = this->create_publisher<radar_interfaces::msg::SentryTactical>("detector/tactical_info", 10);
        
        srv_config_outpost_ = this->create_service<std_srvs::srv::Trigger>(
            "detector/config_outpost_roi",
            std::bind(&NetDetectorComponent::handleConfigOutpostROI, this, 
                std::placeholders::_1, std::placeholders::_2)
        );

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 
            5,
            std::bind(&NetDetectorComponent::imageCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "\033[1;32m[Vision] 神经网络引擎已点火！多维度信息输出就绪\033[0m");
        RCLCPP_INFO(this->get_logger(), "[Outpost] 复用 armor 模型检测前哨站，调用服务 'detector/config_outpost_roi' 可配置ROI");
    }

private:
    std::unique_ptr<Model> robot_model_;
    std::unique_ptr<Model> armor_model_;
    std::unique_ptr<Model> plane_model_; 
    std::unique_ptr<Classifier> classifier_model_; 
    
    // 前哨站检测相关（复用 armor 模型）
    cv::Rect outpost_roi_;
    int outpost_miss_count_ = 0;
    bool is_outpost_alive_ = true;
    const int OUTPOST_MISS_THRESHOLD = 20;
    const int OUTPOST_WARMUP_SECONDS = 60; // 恢复为开局前 60 秒免检保护
    std::chrono::steady_clock::time_point system_start_time_;
    
    // 【热配置】ROI配置相关
    std::atomic<bool> outpost_config_mode_{false};
    std::atomic<bool> capture_next_frame_{false}; 
    std::unique_ptr<ui::OutpostConfigUI> outpost_config_ui_;
    cv::Mat latest_frame_;  
    std::mutex frame_mutex_;

    std::vector<std::string> classifier_labels_; 
    std::string config_file_path_;
    
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point last_time_ = std::chrono::steady_clock::now();
    double total_latency_ms_ = 0.0;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
    rclcpp::Publisher<radar_interfaces::msg::DetectResults>::SharedPtr pub_results_;
    rclcpp::Publisher<radar_interfaces::msg::SentryTactical>::SharedPtr pub_tactical_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_config_outpost_;

    void handleConfigOutpostROI(
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        if (outpost_config_mode_.exchange(true)) {
            response->success = false;
            response->message = "已经在配置模式中";
            return;
        }
        
        capture_next_frame_ = true;
        
        std::thread([this]() {
            RCLCPP_INFO(this->get_logger(), "[Outpost] 等待捕获帧用于配置...");
            
            int wait_count = 0;
            while (capture_next_frame_ && wait_count < 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                wait_count++;
            }
            
            cv::Mat config_frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                config_frame = latest_frame_.clone();
            }
            
            if (config_frame.empty()) {
                RCLCPP_ERROR(this->get_logger(), "[Outpost] 无法获取图像，配置失败");
                outpost_config_mode_ = false;
                return;
            }
            
            outpost_config_ui_ = std::make_unique<ui::OutpostConfigUI>();
            outpost_config_ui_->start(config_frame);
            
            if (outpost_config_ui_->run()) {
                cv::Rect new_roi = outpost_config_ui_->getROI();
                outpost_roi_ = new_roi;
                
                try {
                    YAML::Node config = YAML::LoadFile(config_file_path_);
                    config["outpost_roi"] = std::vector<int>{
                        outpost_roi_.x, outpost_roi_.y, 
                        outpost_roi_.width, outpost_roi_.height
                    };
                    std::ofstream fout(config_file_path_);
                    fout << config;
                    fout.close();
                    
                    RCLCPP_INFO(this->get_logger(), "[Outpost] ROI已保存: [%d, %d, %d, %d]", 
                        outpost_roi_.x, outpost_roi_.y, outpost_roi_.width, outpost_roi_.height);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "[Outpost] 保存ROI失败: %s", e.what());
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "[Outpost] 配置已取消");
            }
            
            outpost_config_ui_->stop();
            outpost_config_ui_.reset();
            cv::destroyAllWindows();
            for(int i = 0; i < 10; ++i) cv::waitKey(1);
            
            outpost_config_mode_ = false;
            RCLCPP_INFO(this->get_logger(), "[Outpost] 配置模式已退出");
        }).detach();
        
        response->success = true;
        response->message = "已进入ROI配置模式";
    }

    void imageCallback(sensor_msgs::msg::Image::UniquePtr msg) 
    {
        if (outpost_config_mode_) {
            if (capture_next_frame_) {
                try {
                    cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_frame_ = frame.clone();
                } catch (...) {}
                capture_next_frame_ = false;
            }
            return;
        }
        
        auto start_time = std::chrono::steady_clock::now();

        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());

        // 1. 先跑主干网络获取车辆/装甲板，得到纯净对象队列
        std::vector<DetectObject> objects = process_frame(frame);
        
        // 2. 立即将纯净对象送给级联追踪引擎，避免掺入前哨站干扰追踪
        publish_results(objects, msg->header);

        // 3.  把 frame 传进去，主动抠图并调用 armor 模型推理前哨站
        bool outpost_alive_status = process_outpost(frame, objects);
        
        // 发布战术信息
        radar_interfaces::msg::SentryTactical tactical_msg;
        tactical_msg.header = msg->header;
        tactical_msg.outpost_alive = outpost_alive_status ? 1 : 0;
        tactical_msg.engineer_on_island = 0;
        tactical_msg.enemy_massive_attack = 0;
        tactical_msg.ally_massive_attack = 0;
        pub_tactical_->publish(tactical_msg);

        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_latency_ms_ += latency;
        frame_count_++;

        if (frame_count_ % 30 == 0) {
            auto elapsed_sec = std::chrono::duration<double>(end_time - last_time_).count();
            double avg_fps = 30.0 / elapsed_sec;
            double avg_latency = total_latency_ms_ / 30.0;
            RCLCPP_INFO(this->get_logger(), 
                "[Vision] %.1f FPS | 推理延迟: %.1f ms | 目标: %zu | 前哨站: %s", 
                avg_fps, avg_latency, objects.size(), 
                outpost_alive_status ? "存活" : "摧毁");
            
            total_latency_ms_ = 0.0;
            last_time_ = end_time;
        }

        float scale = 1280.0f / frame.cols; 
        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(), scale, scale, cv::INTER_LINEAR); 

        auto header_copy = msg->header;
        msg.reset(); 

        std::thread([this, small_frame, objects, header_copy, outpost_alive_status]() mutable {
            publish_processed_video(small_frame, objects, header_copy, outpost_alive_status);
        }).detach();
    }

    // 主动截取 ROI 区域，送给 armor 模型极速推理
    bool process_outpost(cv::Mat& frame, std::vector<DetectObject>& objects) {
        if (outpost_roi_.area() <= 0) return true; // 未配置 ROI 则默认存活
        
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - system_start_time_).count();
        if (elapsed < OUTPOST_WARMUP_SECONDS) {
            return true;
        }
        
        cv::Rect dynamic_roi = adapt_roi_to_resolution(frame.cols, frame.rows);
        cv::Rect safe_roi = dynamic_roi & cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (safe_roi.area() <= 0) return is_outpost_alive_;

        // 主动抠出前哨站那一小块区域
        cv::Mat roi_img = frame(safe_roi).clone();

        // 强行让 armor_model_ 扫描这块图，获取纯正的装甲板检测结果
        if (armor_model_ && armor_model_->Detect(roi_img) && !armor_model_->detectResults.empty()) {
            
            for (const auto& res : armor_model_->detectResults) {
                cv::Rect global_rect(res.box.x + safe_roi.x, res.box.y + safe_roi.y, res.box.width, res.box.height);
                DetectObject obj;
                obj.rect = global_rect;
                obj.label = "Outpost"; 
                obj.type = OUTPOST;
                obj.class_id = -3;
                obj.class_conf = res.confidence;
                obj.car_conf = res.confidence;
                obj.bot_id = -1;
                objects.push_back(obj); 
            }

            outpost_miss_count_ = 0;
            is_outpost_alive_ = true;
        } else {
            if (outpost_miss_count_ < OUTPOST_MISS_THRESHOLD) {
                outpost_miss_count_++;
                if (outpost_miss_count_ == OUTPOST_MISS_THRESHOLD) {
                    is_outpost_alive_ = false;
                }
            }
        }
        
        return is_outpost_alive_;
    }

    cv::Rect adapt_roi_to_resolution(int img_width, int img_height) {
        const int ref_width = 5472;
        const int ref_height = 3648;
        
        float scale_x = (float)img_width / ref_width;
        float scale_y = (float)img_height / ref_height;
        
        int x = (int)(outpost_roi_.x * scale_x);
        int y = (int)(outpost_roi_.y * scale_y);
        int w = (int)(outpost_roi_.width * scale_x);
        int h = (int)(outpost_roi_.height * scale_y);
        
        return cv::Rect(x, y, w, h);
    }

    std::vector<DetectObject> process_frame(cv::Mat& frame) 
    {
        std::vector<DetectObject> final_objects;

        // 通道 1：地面机器人与装甲板
        if (robot_model_->Detect(frame)) {
            for (const auto& robot_res : robot_model_->detectResults) {
                cv::Rect robot_roi = robot_res.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (robot_roi.area() <= 0) continue;
                
                cv::Mat robot_img = frame(robot_roi);

                if (armor_model_->Detect(robot_img) && !armor_model_->detectResults.empty()) {
                    auto best_armor_it = std::max_element(
                        armor_model_->detectResults.begin(), armor_model_->detectResults.end(),
                        [](const Result& a, const Result& b) { return a.confidence < b.confidence; }
                    );
                    const auto& armor_res = *best_armor_it;
                    
                    cv::Rect armor_rect_global(armor_res.box.x + robot_roi.x, armor_res.box.y + robot_roi.y, armor_res.box.width, armor_res.box.height);
                    cv::Rect num_roi = armor_rect_global & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (num_roi.area() <= 0) continue;
                    
                    cv::Mat number_img = frame(num_roi);
                    float num_confidence = 0.0f;
                    int class_id = classifier_model_->Classify(number_img, num_confidence);
                    std::string label = (class_id >= 0 && class_id < (int)classifier_labels_.size()) ? classifier_labels_[class_id] : "Unknown";
                    if (label == "Unknown") class_id = -1;
                    
                    DetectObject obj;
                    obj.rect = armor_rect_global;
                    obj.label = label;
                    obj.type = ARMOR; 
                    obj.class_id = class_id;
                    obj.class_conf = num_confidence;
                    obj.car_conf = robot_res.confidence;
                    obj.bot_id = -1;
                    obj.center = cv::Point2f(armor_rect_global.x + armor_rect_global.width / 2.0f, armor_rect_global.y + armor_rect_global.height / 2.0f);
                    final_objects.push_back(obj);
                }   
            }
        }

        // 通道 2：无人机
        if (plane_model_) {
            cv::Rect right_half(frame.cols / 2, 0, frame.cols / 2, frame.rows);
            cv::Mat right_frame = frame(right_half).clone();
            
            if (plane_model_->Detect(right_frame) && !plane_model_->detectResults.empty()) {
                for (const auto& plane_res : plane_model_->detectResults) {
                    cv::Rect plane_roi(
                        plane_res.box.x + frame.cols / 2,
                        plane_res.box.y,
                        plane_res.box.width,
                        plane_res.box.height
                    );
                    plane_roi = plane_roi & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (plane_roi.area() <= 0) continue;

                    DetectObject obj;
                    obj.rect = plane_roi;
                    obj.label = "Drone"; 
                    obj.type = DRONE;    
                    obj.class_id = -2;
                    obj.class_conf = plane_res.confidence;
                    obj.car_conf = plane_res.confidence;  
                    obj.bot_id = -1;
                    obj.center = cv::Point2f(plane_roi.x + plane_roi.width / 2.0f, plane_roi.y + plane_roi.height / 2.0f);
                    final_objects.push_back(obj);
                }
            }
        }

        return final_objects;
    }

    void publish_results(const std::vector<DetectObject>& objects, const std_msgs::msg::Header& header)
    {
        radar_interfaces::msg::DetectResults results_msg;
        results_msg.header = header;
        for(const auto &obj : objects){
            radar_interfaces::msg::DetectResult res;
            res.number = obj.label; 
            res.class_id = obj.class_id;         
            res.class_conf = obj.class_conf;     
            res.x = obj.center.x; 
            res.y = obj.center.y;
            res.width = obj.rect.width;          
            res.height = obj.rect.height;        
            res.car_conf = obj.car_conf;         
            res.bot_id = obj.bot_id;             
            results_msg.results.push_back(res);
        }
        pub_results_->publish(results_msg);
    }

    void publish_processed_video(cv::Mat& small_frame, const std::vector<DetectObject>& objects, const std_msgs::msg::Header& header, bool is_outpost_alive)
    {
        float scale = 1280.0f / 5472.0f; 
        
        // 渲染所有目标 (包含复用出来的 Outpost，完全一样的框和文字逻辑)
        for(const auto &obj : objects){
            cv::Rect scaled_rect(obj.rect.x * scale, obj.rect.y * scale, obj.rect.width * scale, obj.rect.height * scale);
            
            cv::Scalar color;
            if (obj.type == DRONE) color = cv::Scalar(0, 165, 255);
            else if (obj.type == OUTPOST) color = cv::Scalar(255, 255, 0);
            else color = cv::Scalar(0, 255, 0);
            
            cv::rectangle(small_frame, scaled_rect, color, 2);
            cv::putText(small_frame, obj.label, scaled_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 战术可视化 UI
        std::string outpost_str = is_outpost_alive ? "Outpost: ALIVE [======]" : "Outpost: DESTROYED [ ]";
        cv::Scalar outpost_color = is_outpost_alive ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(small_frame, outpost_str, cv::Point(20, 50), cv::FONT_HERSHEY_DUPLEX, 1.2, outpost_color, 2);

        auto out_msg = cv_bridge::CvImage(header, "bgr8", small_frame).toImageMsg();
        pub_img_->publish(*out_msg);
    }

    void init_parameters() {
        this->declare_parameter<std::string>("config_file", "/home/lzhros/Code/RadarStation/config/detector/yolo.yaml");
        this->get_parameter("config_file", config_file_path_);
    }

    void init_models(){
        try {
            YAML::Node config = YAML::LoadFile(config_file_path_);
            
            robot_model_ = std::make_unique<Model>(config["robot_modelpath"].as<std::string>(), config["robot_inputSize"].as<int>(), config["robot_scoreThresh"].as<float>(), config["robot_nmsThresh"].as<float>(), true);
            armor_model_ = std::make_unique<Model>(config["armor_modelpath"].as<std::string>(), config["armor_inputSize"].as<int>(), config["armor_scoreThresh"].as<float>(), config["armor_nmsThresh"].as<float>(), false);
            classifier_model_ = std::make_unique<Classifier>(config["classifier_modelpath"].as<std::string>(), config["classifier_inputSize"].as<int>());
            for (const auto &item : config["classifier_labels"]) classifier_labels_.push_back(item.second.as<std::string>());
            
            try {
                if (config["plane_modelpath"]) {
                    plane_model_ = std::make_unique<Model>(
                        config["plane_modelpath"].as<std::string>(), 
                        config["plane_inputSize"].as<int>(), 
                        config["plane_scoreThresh"].as<float>(), 
                        config["plane_nmsThresh"].as<float>(), 
                        true 
                    );
                }
            } catch (const std::exception &e) {
                RCLCPP_WARN(this->get_logger(), "未找到或无法加载防空网络模型: %s", e.what());
            }

            // 加载前哨站 ROI 配置（不再加载模型）
            try {
                if (config["outpost_roi"]) {
                    auto roi_cfg = config["outpost_roi"].as<std::vector<int>>();
                    outpost_roi_ = cv::Rect(roi_cfg[0], roi_cfg[1], roi_cfg[2], roi_cfg[3]);
                    
                    RCLCPP_INFO(this->get_logger(), "\033[1;34m[Outpost] 前哨站检测已启用! ROI: [%d, %d, %d, %d] (复用 armor 模型)\033[0m",
                        outpost_roi_.x, outpost_roi_.y, outpost_roi_.width, outpost_roi_.height);
                }
            } catch (const std::exception &e) {
                RCLCPP_WARN(this->get_logger(), "前哨站 ROI 配置加载失败: %s", e.what());
            }

        } catch(const std::exception &e) { 
            RCLCPP_ERROR(this->get_logger(), "核心神经网络初始化失败: %s", e.what()); 
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::NetDetectorComponent)