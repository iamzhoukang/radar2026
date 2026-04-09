#include "tracker/point_guesser.hpp"
#include "tracker/cascade_tracker.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

namespace radar_core {
namespace tracker {

PointGuesser::PointGuesser(const std::string& config_path) {
    std::string path = config_path;
    if (path.empty()) {
        // 默认路径
        path = "/home/lzhros/Code/RadarStation/config/map/guess_pts.yaml";
    }
    load_config(path);
}

void PointGuesser::load_config(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        // 加载参数
        cos_factor_ = config["cos_factor"].as<float>(0.003f);
        d_factor_ = config["d_factor"].as<float>(0.1f);
        
        // 加载猜测点
        YAML::Node points = config["guess_points"];
        for (const auto& robot : points) {
            std::string robot_name = robot.first.as<std::string>();
            std::vector<std::pair<float, float>> pts;
            for (const auto& pt : robot.second) {
                float x = pt[0].as<float>();
                float y = pt[1].as<float>();
                pts.emplace_back(x, y);
            }
            guess_points_[robot_name] = pts;
        }
        
        
    } catch (const std::exception& e) {
        
        // 设置默认猜测点（场地四角+中心）
        for (int i = 0; i < 10; ++i) {
            std::string name = name_id_convert_[i];
            if (i < 5) {  // 蓝方
                guess_points_[name] = {
                    {23.0f, 5.0f}, {20.0f, 7.5f}, {18.0f, 10.0f},
                    {25.0f, 3.0f}, {25.0f, 12.0f}
                };
            } else {  // 红方
                guess_points_[name] = {
                    {5.0f, 5.0f}, {8.0f, 7.5f}, {10.0f, 10.0f},
                    {3.0f, 3.0f}, {3.0f, 12.0f}
                };
            }
        }
    }
}

std::vector<std::pair<float, float>> PointGuesser::get_guess_points_for_robot(
    int class_id, const std::string& ref_color) {
    
    auto it = name_id_convert_.find(class_id);
    if (it == name_id_convert_.end()) {
        return {};
    }
    
    std::string robot_name = it->second;
    auto pts_it = guess_points_.find(robot_name);
    if (pts_it == guess_points_.end()) {
        return {};
    }
    
    std::vector<std::pair<float, float>> points = pts_it->second;
    
    // 蓝方需要镜像坐标
    if (ref_color == "blue" || ref_color == "BLUE") {
        for (auto& pt : points) {
            pt.first = 2 * field_center_x_ - pt.first;
            pt.second = 2 * field_center_y_ - pt.second;
        }
    }
    
    return points;
}

float PointGuesser::compute_cosine_similarity(
    float vx, float vy, float dx, float dy) {
    float dot_product = vx * dx + vy * dy;
    float v_norm = std::sqrt(vx * vx + vy * vy);
    float d_norm = std::sqrt(dx * dx + dy * dy);
    
    if (v_norm < 1e-6 || d_norm < 1e-6) {
        return 0.0f;  // 速度为0或距离为0时，余弦相似度为0
    }
    
    return dot_product / (v_norm * d_norm);
}

std::vector<float> PointGuesser::predict_points(
    const TrackingState& track, const std::string& ref_color) {
    
    // 获取该兵种的预设猜测点
    auto guess_points = get_guess_points_for_robot(track.class_id, ref_color);
    if (guess_points.empty()) {
        // 没有猜测点时，返回当前位置
        return {track.pos_2d_uwb.x, track.pos_2d_uwb.y};
    }
    
    // 获取当前卡尔曼滤波器的状态
    auto pos = track.kalman_2d.get_position();
    auto vel = track.kalman_2d.get_velocity();
    
    float last_x = pos[0];
    float last_y = pos[1];
    float vx = vel[0];
    float vy = vel[1];
    
    // 如果速度为0，直接返回最近的猜测点
    float v_norm = std::sqrt(vx * vx + vy * vy);
    if (v_norm < 0.1f) {
        // 找到最近的猜测点
        float min_dist = 1e9f;
        std::pair<float, float> best_point = guess_points[0];
        for (const auto& pt : guess_points) {
            float dx = pt.first - last_x;
            float dy = pt.second - last_y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
                best_point = pt;
            }
        }
        return {best_point.first, best_point.second};
    }
    
    // 计算每个猜测点的分数
    std::vector<std::pair<std::pair<float, float>, float>> scored_points;  // {point, score}
    
    for (const auto& point : guess_points) {
        float px = point.first;
        float py = point.second;
        
        // 计算到猜测点的向量
        float dx = px - last_x;
        float dy = py - last_y;
        
        // 计算余弦相似度（速度方向 vs 到猜测点方向）
        float cos_sim = compute_cosine_similarity(vx, vy, dx, dy);
        
        // 计算欧式距离
        float distance = std::sqrt(dx * dx + dy * dy);
        float d_score = std::exp(-distance * d_factor_);
        
        // 综合分数（余弦相似度权重低，距离权重高）
        float score = cos_factor_ * cos_sim + (1.0f - cos_factor_) * d_score;
        
        scored_points.push_back({{px, py}, score});
    }
    
    // 按分数排序，选择最高分
    std::sort(scored_points.begin(), scored_points.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    const auto& best = scored_points[0];
    
    
    return {best.first.first, best.first.second};
}

} // namespace tracker
} // namespace radar_core
