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
// #include "utils/tracker.hpp"

namespace radar_core 
{

// 目标类型枚举 
enum TargetType { ROBOT = 0, ARMOR = 1 };

// 检测目标结构体
struct DetectObject {
    cv::Rect rect;       
    int class_id;        
    std::string label;   
    float confidence;    
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

        // 零拷贝订阅核心
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 
            5,
            std::bind(&NetDetectorComponent::imageCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), " 神经网络组件已启动 ，等待零拷贝图像...");
    }

private:
    std::unique_ptr<Model> robot_model_;
    std::unique_ptr<Model> armor_model_;
    std::unique_ptr<Classifier> classifier_model_; 
    // 【已移除】 std::unique_ptr<CascadeTracker> tracker_;

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

        // 1. 回归极致性能：0ms 裸指针内存直接映射
        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());

        // 2. 核心检测管线 (纯 GPU/CPU 推理，只读不写)
        std::vector<DetectObject> objects = process_frame(frame);

        // 3. 发布官方裁判系统所需坐标
        publish_results(objects, msg->header);

        // 4. 结算【纯粹的】神经网络端到端延迟
        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_latency_ms_ += latency;
        frame_count_++;

        // 打印性能
        auto elapsed_sec = std::chrono::duration<double>(end_time - last_time_).count();
        if (elapsed_sec >= 1.0) { 
            double avg_fps = frame_count_ / elapsed_sec;
            double avg_latency = total_latency_ms_ / frame_count_;
            RCLCPP_INFO(this->get_logger(), 
                "[Perf] 吞吐率: %.2f FPS | 神经网络端到端延迟: %.2f ms", 
                avg_fps, avg_latency);
            
            frame_count_ = 0;
            total_latency_ms_ = 0.0;
            last_time_ = end_time;
        }

        // 5. 【终极性能优化】：将 msg 的所有权(Move)移交入后台独立线程
        // 这意味着主线程瞬间结束，而后台线程接管 40MB 内存去慢慢做 1920 的缩放和画框，绝不卡主频！
        std::thread([this, msg_ptr = std::move(msg), objects]() mutable {
            // 在后台线程重新用裸指针挂载图像
            cv::Mat raw_frame(msg_ptr->height, msg_ptr->width, CV_8UC3, msg_ptr->data.data());
            // 执行降采样与发图
            publish_processed_video(raw_frame, objects, msg_ptr->header);
            // 线程结束时，msg_ptr 超出作用域，40MB 内存自动安全释放
        }).detach();
    }

    std::vector<DetectObject> process_frame(cv::Mat& frame) 
    {
        std::vector<DetectObject> final_objects;

        // 1. 全图检测 Robot
        if (robot_model_->Detect(frame)) {
            for (const auto& robot_res : robot_model_->detectResults) {
                
                cv::Rect robot_roi = robot_res.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (robot_roi.area() <= 0) continue;

                cv::Mat robot_img = frame(robot_roi);

                // 2. 局部检测 Armor
                // 如果检测到了装甲板，且数量大于 0
                if (armor_model_->Detect(robot_img) && !armor_model_->detectResults.empty()) {
                    
                    // 利用 C++ 算法库，一键找出置信度 (confidence) 最高的那个结果
                    auto best_armor_it = std::max_element(
                        armor_model_->detectResults.begin(),
                        armor_model_->detectResults.end(),
                        [](const Result& a, const Result& b) {
                            return a.confidence < b.confidence; 
                            // 提示：如果你想改成面积最大，就把这里改成 return a.box.area() < b.box.area();
                        }
                    );

                    // 提取出那个最完美的装甲板
                    const auto& armor_res = *best_armor_it;

                    // 还原全局坐标 (只对这唯一的一个装甲板进行处理)
                    cv::Rect armor_rect_global(
                        armor_res.box.x + robot_roi.x,
                        armor_res.box.y + robot_roi.y,
                        armor_res.box.width,
                        armor_res.box.height
                    );

                    cv::Rect num_roi = armor_rect_global & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (num_roi.area() <= 0) continue;

                    cv::Mat number_img = frame(num_roi);

                    // 3. 分类器识别数字 (现在每辆车只算 1 次，速度更快)
                    float num_confidence = 0.0f;
                    int class_id = classifier_model_->Classify(number_img, num_confidence);
                    std::string label = (class_id >= 0 && class_id < classifier_labels_.size()) 
                                        ? classifier_labels_[class_id] : "Unknown";

                    DetectObject obj;
                    obj.rect = armor_rect_global;
                    obj.class_id = class_id;
                    obj.label = label;
                    obj.confidence = armor_res.confidence;
                    obj.type = TargetType::ARMOR;
                    obj.center = cv::Point2f(armor_rect_global.x + armor_rect_global.width / 2.0f,
                                             armor_rect_global.y + armor_rect_global.height / 2.0f);
                    
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
            res.x = obj.center.x;
            res.y = obj.center.y;
            results_msg.results.push_back(res);
        }
        pub_results_->publish(results_msg);
    }

    void publish_processed_video(cv::Mat& raw_frame, const std::vector<DetectObject>& objects, const std_msgs::msg::Header& header)
    {
        cv::Mat vis_frame;
        float scale = 1980.0f / raw_frame.cols; 
        cv::resize(raw_frame, vis_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);

        for(const auto &obj : objects){
            cv::Rect scaled_rect(obj.rect.x * scale, obj.rect.y * scale, obj.rect.width * scale, obj.rect.height * scale);
            cv::Point scaled_center(obj.center.x * scale, obj.center.y * scale);

            cv::rectangle(vis_frame, scaled_rect, cv::Scalar(0, 255, 0), 1);

            cv::circle(vis_frame, scaled_center, 2, cv::Scalar(0, 0, 255), -1);
            
            cv::Point textOrg(scaled_rect.x, std::max(15, scaled_rect.y - 5));
            cv::putText(vis_frame, obj.label, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        auto out_msg = cv_bridge::CvImage(header, "bgr8", vis_frame).toImageMsg();
        pub_img_->publish(*out_msg);
    }

    void init_parameters() {
        this->declare_parameter<std::string>("config_file", "/home/lzhros/Code/RadarStation/config/detector/yolo.yaml");
        this->get_parameter("config_file", config_file_path_);
    }

    void init_models(){
        try{
            YAML::Node config = YAML::LoadFile(config_file_path_);
            
            // 算力路由分配
            robot_model_ = std::make_unique<Model>(
                config["robot_modelpath"].as<std::string>(), config["robot_inputSize"].as<int>(), 
                config["robot_scoreThresh"].as<float>(), config["robot_nmsThresh"].as<float>(), 
                true); // 大图走 GPU
            
            armor_model_ = std::make_unique<Model>(
                config["armor_modelpath"].as<std::string>(), config["armor_inputSize"].as<int>(), 
                config["armor_scoreThresh"].as<float>(), config["armor_nmsThresh"].as<float>(), 
                false); // 小图走 CPU
            
            classifier_model_ = std::make_unique<Classifier>(
                config["classifier_modelpath"].as<std::string>(), config["classifier_inputSize"].as<int>());
            
            // 【已移除】 tracker_ = std::make_unique<CascadeTracker>();
            
            for (const auto &item : config["classifier_labels"]) {
                classifier_labels_.push_back(item.second.as<std::string>());
            }
        }catch(const std::exception &e){
            RCLCPP_ERROR(this->get_logger(), "模型初始化失败: %s", e.what());
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::NetDetectorComponent)