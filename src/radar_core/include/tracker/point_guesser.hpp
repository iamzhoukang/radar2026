#pragma once
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

namespace radar_core {
namespace tracker {

// 前向声明
struct TrackingState;

/**
 * PointGuesser - 目标丢失后的位置猜测器
 * 
 * 根据当前速度向量和场地预设的"猜测点"，
 * 计算目标最可能前往的位置。
 */
class PointGuesser {
public:
    /**
     * @param config_path YAML配置文件路径
     */
    explicit PointGuesser(const std::string& config_path = "");
    
    /**
     * 预测目标丢失后可能去的位置
     * @param track 当前追踪状态（包含位置和速度）
     * @param ref_color 参考阵营 ("red" 或 "blue")
     * @return 最佳猜测点 [x, y]
     */
    std::vector<float> predict_points(const TrackingState& track, const std::string& ref_color);

private:
    float cos_factor_ = 0.003f;   // 余弦相似度权重
    float d_factor_ = 0.1f;       // 距离权重
    
    // 场地中心点（用于蓝方镜像）
    const float field_center_x_ = 14.0f;
    const float field_center_y_ = 7.5f;
    
    // 兵种名称映射
    std::map<int, std::string> name_id_convert_ = {
        {0, "B1"}, {1, "B2"}, {2, "B3"}, {3, "B4"}, {4, "B7"},
        {5, "R1"}, {6, "R2"}, {7, "R3"}, {8, "R4"}, {9, "R7"}
    };
    
    // 各兵种的预设猜测点
    std::map<std::string, std::vector<std::pair<float, float>>> guess_points_;
    
    /**
     * 加载YAML配置文件
     */
    void load_config(const std::string& config_path);
    
    /**
     * 获取指定兵种的猜测点
     * @param class_id 兵种ID (0-9)
     * @param ref_color 阵营颜色 ("red" 或 "blue")
     * @return 猜测点列表
     */
    std::vector<std::pair<float, float>> get_guess_points_for_robot(
        int class_id, const std::string& ref_color);
    
    /**
     * 计算余弦相似度
     */
    float compute_cosine_similarity(
        float vx, float vy, 
        float dx, float dy);
};

} // namespace tracker
} // namespace radar_core
