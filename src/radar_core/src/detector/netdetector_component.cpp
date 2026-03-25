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

enum TargetType { ROBOT = 0, ARMOR = 1 };

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

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 
            5,
            std::bind(&NetDetectorComponent::imageCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "极致丝滑版组件已启动 (包含性能监测)");
    }

private:
    std::unique_ptr<Model> robot_model_;
    std::unique_ptr<Model> armor_model_;
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
        // 开始计时：神经网络端到端延迟
        auto start_time = std::chrono::steady_clock::now();

        // 1. 内存映射
        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data());

        // 2. 神经网络推理
        std::vector<DetectObject> objects = process_frame(frame);

        // 3. 发布目标坐标
        publish_results(objects, msg->header);

        // 4. 计算并累加延迟
        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_latency_ms_ += latency;
        frame_count_++;

        // 5. 吞吐率计算与打印
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

        //主线程极速缩放 + 内存早释放
        float scale = 1280.0f / frame.cols; 
        cv::Mat small_frame;
        // 使用 LINEAR 保证速度
        cv::resize(frame, small_frame, cv::Size(), scale, scale, cv::INTER_LINEAR); 

        // 保存 Header 用于异步发布
        auto header_copy = msg->header;

        // 【关键】手动让 UniquePtr 失效，提前释放海康相机的 60MB 原始内存块
        msg.reset(); 

        // 6. 异步渲染 (此时原始大内存已归还相机，QT 依然可以满帧显示)
        std::thread([this, small_frame, objects, header_copy]() mutable {
            publish_processed_video(small_frame, objects, header_copy);
        }).detach();
    }

    std::vector<DetectObject> process_frame(cv::Mat& frame) 
    {
        std::vector<DetectObject> final_objects;
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
                    DetectObject obj;
                    obj.rect = armor_rect_global;
                    obj.label = label;
                    obj.center = cv::Point2f(armor_rect_global.x + armor_rect_global.width / 2.0f, armor_rect_global.y + armor_rect_global.height / 2.0f);
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
            res.x = obj.center.x; res.y = obj.center.y;
            results_msg.results.push_back(res);
        }
        pub_results_->publish(results_msg);
    }

    void publish_processed_video(cv::Mat& small_frame, const std::vector<DetectObject>& objects, const std_msgs::msg::Header& header)
    {
        // 这里的 small_frame 已经是 1280 宽度了，直接画框
        float scale = 1280.0f / 5472.0f; // 假设原图宽度，请根据实际修改
        for(const auto &obj : objects){
            cv::Rect scaled_rect(obj.rect.x * scale, obj.rect.y * scale, obj.rect.width * scale, obj.rect.height * scale);
            cv::rectangle(small_frame, scaled_rect, cv::Scalar(0, 255, 0), 1);
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
        try{
            YAML::Node config = YAML::LoadFile(config_file_path_);
            robot_model_ = std::make_unique<Model>(config["robot_modelpath"].as<std::string>(), config["robot_inputSize"].as<int>(), config["robot_scoreThresh"].as<float>(), config["robot_nmsThresh"].as<float>(), true);
            armor_model_ = std::make_unique<Model>(config["armor_modelpath"].as<std::string>(), config["armor_inputSize"].as<int>(), config["armor_scoreThresh"].as<float>(), config["armor_nmsThresh"].as<float>(), false);
            classifier_model_ = std::make_unique<Classifier>(config["classifier_modelpath"].as<std::string>(), config["classifier_inputSize"].as<int>());
            for (const auto &item : config["classifier_labels"]) classifier_labels_.push_back(item.second.as<std::string>());
        }catch(const std::exception &e){ RCLCPP_ERROR(this->get_logger(), "模型初始化失败: %s", e.what()); }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::NetDetectorComponent)