#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "radar_interfaces/msg/detect_results.hpp"
#include "radar_interfaces/msg/detect_result.hpp"
#include "utils/model.hpp"
#include "utils/classifier.hpp"

namespace radar_core 
{

// 目标类型枚举
enum TargetType { ROBOT = 0, ARMOR = 1, DRONE = 2 };

// 扩充的内部检测对象结构体
struct DetectObject {
    cv::Rect rect;       
    int class_id;        
    std::string label;   
    float class_conf;    // 分类置信度
    float car_conf;      // 车辆检测置信度
    int bot_id;          // 底层追踪ID (预留)
    TargetType type;     
    cv::Point2f center;  
};

class NetDetectorComponent : public rclcpp::Node 
{
public:
    explicit NetDetectorComponent(const rclcpp::NodeOptions & options) 
    : Node("net_detector_component", options) 
    {
        init_parameters();
        init_models();

        pub_img_ = this->create_publisher<sensor_msgs::msg::Image>("processed_video", 10);
        pub_results_ = this->create_publisher<radar_interfaces::msg::DetectResults>("detector/results", 10);

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 
            5,
            std::bind(&NetDetectorComponent::imageCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "\033[1;32m[Vision] 神经网络引擎已点火！多维度信息(置信度/宽高)输出就绪\033[0m");
    }

private:
    std::unique_ptr<Model> robot_model_;
    std::unique_ptr<Model> armor_model_;
    std::unique_ptr<Model> plane_model_; 
    std::unique_ptr<Classifier> classifier_model_; 

    std::vector<std::string> classifier_labels_; 
    std::string config_file_path_;
    
    // 性能监控变量
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point last_time_ = std::chrono::steady_clock::now();
    double total_latency_ms_ = 0.0;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
    rclcpp::Publisher<radar_interfaces::msg::DetectResults>::SharedPtr pub_results_;

    void imageCallback(sensor_msgs::msg::Image::UniquePtr msg) 
    {
        auto start_time = std::chrono::steady_clock::now();

        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());

        std::vector<DetectObject> objects = process_frame(frame);

        publish_results(objects, msg->header);

        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_latency_ms_ += latency;
        frame_count_++;

        if (frame_count_ % 30 == 0) {
            auto elapsed_sec = std::chrono::duration<double>(end_time - last_time_).count();
            double avg_fps = 30.0 / elapsed_sec;
            double avg_latency = total_latency_ms_ / 30.0;
            RCLCPP_INFO(this->get_logger(), 
                "[Vision] %.1f FPS | 推理延迟: %.1f ms | 视野目标: %zu", 
                avg_fps, avg_latency, objects.size());
            
            total_latency_ms_ = 0.0;
            last_time_ = end_time;
        }

        float scale = 1280.0f / frame.cols; 
        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(), scale, scale, cv::INTER_LINEAR); 

        auto header_copy = msg->header;
        msg.reset(); 

        std::thread([this, small_frame, objects, header_copy]() mutable {
            publish_processed_video(small_frame, objects, header_copy);
        }).detach();
    }

    std::vector<DetectObject> process_frame(cv::Mat& frame) 
    {
        std::vector<DetectObject> final_objects;

        // ==========================================
        // 通道 1：地面机器人与装甲板数字联级
        // ==========================================
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
                    std::string label = (class_id >= 0 && class_id < classifier_labels_.size()) ? classifier_labels_[class_id] : "Unknown";
                    if (label == "Unknown") class_id = -1; // -1 代表未知装甲板
                    
                    DetectObject obj;
                    obj.rect = armor_rect_global;
                    obj.label = label;
                    obj.type = ARMOR; 
                    obj.class_id = class_id;              // 新增: 兵种 ID
                    obj.class_conf = num_confidence;      // 新增: 分类概率
                    obj.car_conf = robot_res.confidence;  // 新增: 车辆置信度
                    obj.bot_id = -1;                      // 新增: 暂留空位
                    obj.center = cv::Point2f(armor_rect_global.x + armor_rect_global.width / 2.0f, armor_rect_global.y + armor_rect_global.height / 2.0f);
                    final_objects.push_back(obj);
                }   
            }
        }

        // ==========================================
        // 通道 2：空中目标 (无人机)
        // ==========================================
        if (plane_model_ && plane_model_->Detect(frame)) {
            for (const auto& plane_res : plane_model_->detectResults) {
                cv::Rect plane_roi = plane_res.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (plane_roi.area() <= 0) continue;

                DetectObject obj;
                obj.rect = plane_roi;
                obj.label = "Drone"; 
                obj.type = DRONE;    
                obj.class_id = -2;                    // 【关键】赋予 -2 ID，代表不需要 KF 的无人机
                obj.class_conf = plane_res.confidence;// 暂以 YOLO 置信度替代
                obj.car_conf = plane_res.confidence;  
                obj.bot_id = -1;
                obj.center = cv::Point2f(plane_roi.x + plane_roi.width / 2.0f, plane_roi.y + plane_roi.height / 2.0f);
                final_objects.push_back(obj);
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
            res.class_id = obj.class_id;         // 新增填装
            res.class_conf = obj.class_conf;     // 新增填装
            res.x = obj.center.x; 
            res.y = obj.center.y;
            res.width = obj.rect.width;          // 新增填装：框宽
            res.height = obj.rect.height;        // 新增填装：框高
            res.car_conf = obj.car_conf;         // 新增填装：车辆置信度
            res.bot_id = obj.bot_id;             // 新增填装：底层ID
            results_msg.results.push_back(res);
        }
        pub_results_->publish(results_msg);
    }

    void publish_processed_video(cv::Mat& small_frame, const std::vector<DetectObject>& objects, const std_msgs::msg::Header& header)
    {
        float scale = 1280.0f / 5472.0f; 
        for(const auto &obj : objects){
            cv::Rect scaled_rect(obj.rect.x * scale, obj.rect.y * scale, obj.rect.width * scale, obj.rect.height * scale);
            cv::Scalar color = (obj.type == DRONE) ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            
            cv::rectangle(small_frame, scaled_rect, color, 2);
            cv::putText(small_frame, obj.label, scaled_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
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

        } catch(const std::exception &e) { 
            RCLCPP_ERROR(this->get_logger(), "核心神经网络初始化失败: %s", e.what()); 
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::NetDetectorComponent)