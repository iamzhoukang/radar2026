#pragma once
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <opencv2/opencv.hpp>
#include "tracker/kalman.hpp"
#include "tracker/hungarian.hpp"

namespace radar_core {
namespace tracker {

// 前向声明
class PointGuesser;

enum class TrackState { INACTIVE, TENTATIVE, CONFIRMED, LOST };

// 对应 type.py -> SingleDetectionResult
struct SingleDetectionResult {
    int class_id;
    float class_conf;
    cv::Rect2f car_box; // [x, y, w, h]
    float car_conf;
    cv::Point3f pos_3d;
    int bot_id;
};

// 对应 type.py -> BotIdTrack
struct BotIdTrack {
    int bot_id;
    std::deque<int> class_id_queue;
    std::deque<float> class_conf_queue;
    bool updated = false;
    int lost_counter = 0;

    BotIdTrack() {}
    BotIdTrack(int bid, int cid, float conf);
    void update(int cid, float conf);
    std::vector<float> get_class_id_exponent_confidence(int history_length = 10, float tau = 0.5f);
};

// 对应 type.py -> TrackingState
struct TrackingState {
    int class_id;
    std::string name;
    TrackState state = TrackState::INACTIVE;
    float confidence = 0.0f;
    
    bool is_dead = false;
    int miss_count = 0;
    int hit_count = 0;
    int inactive_count = 0;
    int bot_id = -1;
    bool is_active = false;

    cv::Rect2f car_box;
    cv::Point3f pos_3d;
    cv::Point2f pos_2d_uwb;      // 预测值（用于匹配）
    cv::Point2f pos_2d_uwb_det;  // 检测值
    
    // 【新增】猜点相关字段
    cv::Point2f guess_point;      // 猜测点坐标
    bool is_start_guess = false;  // 是否开始猜点

    KalmanFilterBox kalman_box;
    KalmanFilter2d kalman_2d;

    TrackingState() {}
    void init(int cid, const std::string& n);
};

// 对应 tracker.py -> CascadeMatchTracker
class CascadeMatchTracker {
public:
    /**
     * @param faction 阵营 ("red" 或 "blue")
     * @param config_path 猜点配置文件路径（空则使用默认）
     */
    explicit CascadeMatchTracker(
        const std::string& faction = "blue",
        const std::string& guess_config_path = "");

    /**
     * 主追踪函数
     * @param detections 当前帧检测结果
     * @param dt 时间步长（秒）
     */
    void track(const std::vector<SingleDetectionResult>& detections, float dt);

    std::vector<TrackingState> tracks;
    std::map<int, BotIdTrack> bot_id_trajectories;

private:
    // 港科大超级特调超参（与Python版对齐）
    const float W1 = 5.0f;   // 类别置信度权重
    const float W2 = 1.0f;   // IoU权重
    const float W4 = 0.4f;   // 3D距离权重
    const float COST_THRESHOLD = -0.5f;
    const float INACTIVE_COST_THRESHOLD = -1.0f;
    const int HIT_COUNT_THRESHOLD = 2;
    const int MISS_COUNT_THRESHOLD = 5;

    std::string faction_;  // 【新增】阵营
    std::unique_ptr<PointGuesser> point_guesser_;  // 【新增】猜点器

    float compute_score(TrackingState& track, const SingleDetectionResult& det);
    float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b);
};

} // namespace tracker
} // namespace radar_core
