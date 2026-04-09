#include "tracker/cascade_tracker.hpp"
#include <cmath>
#include <algorithm>

namespace radar_core {
namespace tracker {

BotIdTrack::BotIdTrack(int bid, int cid, float conf) : bot_id(bid) {
    update(cid, conf);
}

void BotIdTrack::update(int cid, float conf) {
    class_id_queue.push_back(cid);
    class_conf_queue.push_back(conf);
    if(class_id_queue.size() > 30) class_id_queue.pop_front();
    if(class_conf_queue.size() > 30) class_conf_queue.pop_front();
    updated = true;
}

std::vector<float> BotIdTrack::get_class_id_exponent_confidence(int history_length, float tau) {
    std::vector<float> distribution(10, 0.0f);
    int valid_length = std::min({history_length, (int)class_id_queue.size(), (int)class_conf_queue.size()});
    if (valid_length == 0) return distribution;

    std::vector<float> weights(valid_length);
    float weight_sum = 0.0f;
    for (int i = 0; i < valid_length; ++i) {
        weights[i] = std::exp(-tau * (valid_length - 1 - i));
        weight_sum += weights[i];
    }
    for (int i = 0; i < valid_length; ++i) weights[i] /= weight_sum;

    auto it_id = class_id_queue.begin() + (class_id_queue.size() - valid_length);
    auto it_conf = class_conf_queue.begin() + (class_conf_queue.size() - valid_length);
    
    for (int i = 0; i < valid_length; ++i, ++it_id, ++it_conf) {
        int cid = *it_id;
        float conf = *it_conf;
        if (cid >= 0 && cid <= 9) {
            distribution[cid] += weights[i] * conf;
        } else if (cid >= 10 && cid <= 14) { 
            if (cid - 5 >= 0 && cid - 5 <= 9) distribution[cid - 5] += weights[i] * 0.8f * conf;
            if (cid - 10 >= 0 && cid - 10 <= 9) distribution[cid - 10] += weights[i] * 0.8f * conf;
        } else if (cid == -1) {
            for (int j = 0; j < 10; ++j) distribution[j] += weights[i] / 10.0f * conf;
        }
    }
    
    return distribution;
}

void TrackingState::init(int cid, const std::string& n) {
    class_id = cid;
    name = n;
    state = TrackState::INACTIVE;
    confidence = 0.0f;
    hit_count = 0;
    miss_count = 0;
    inactive_count = 0;
    bot_id = -1;
    is_active = false;
    is_dead = false;
    car_box = cv::Rect2f(0, 0, 0, 0);
    pos_3d = cv::Point3f(0, 0, 0);
    pos_2d_uwb = cv::Point2f(0, 0);
    pos_2d_uwb_det = cv::Point2f(0, 0);
    // 使用合理的默认参数初始化卡尔曼滤波器
    kalman_box = KalmanFilterBox(0.1f, 2.0f, 1.0f);
    kalman_2d = KalmanFilter2d(2.0f, 1.0f, 0.1f);
}

CascadeMatchTracker::CascadeMatchTracker() {
    const std::vector<std::string> LABELS = {"B1", "B2", "B3", "B4", "B7", "R1", "R2", "R3", "R4", "R7"};
    tracks.resize(10);
    for (int i = 0; i < 10; ++i) {
        tracks[i].init(i, LABELS[i]);
    }
}

float CascadeMatchTracker::compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    if (inter.area() <= 0) return 0.0f;
    return inter.area() / (a.area() + b.area() - inter.area());
}

float CascadeMatchTracker::compute_score(TrackingState& track, const SingleDetectionResult& det) {
    float score = 0.0f;
    
    // 1. 几何与物理空间打分（只对非INACTIVE状态）
    if (track.state != TrackState::INACTIVE) {
        if (track.car_box.area() > 0) {
            score += compute_iou(det.car_box, track.car_box) * W2;
        }
        float pos3d_diff = std::sqrt(std::pow(track.pos_3d.x - det.pos_3d.x, 2) + std::pow(track.pos_3d.y - det.pos_3d.y, 2));
        score += std::max(-1.0f, 1.0f - pos3d_diff * W4);
        if (track.bot_id != -1 && track.bot_id == det.bot_id) score += 1.0f;
    } else {
        // 【冷启动】：给予基础分，确保能开始匹配
        score += 0.5f;
    }

    // 2. 身份置信度打分（关键修复）
    if (det.bot_id != -1 && bot_id_trajectories.count(det.bot_id)) {
        // 使用bot_id历史轨迹判断类别
        auto dist = bot_id_trajectories[det.bot_id].get_class_id_exponent_confidence();
        score += dist[track.class_id] * W1;
    } else {
        // 直接对比兵种ID（主要修复点）
        if (det.class_id == track.class_id) {
            // 类别完全匹配，给予高分
            score += det.class_conf * W1;
        } else if (det.class_id == -1) {
            // 【修复】未知装甲板给予适中分数，允许匹配但分数较低
            // 这样未知装甲板可以关联到任何track，但已知装甲板有优先权
            score += 0.3f * W1;  // 0.3是未知装甲板的默认置信度
        }
        // class_id不同的情况，不加分（避免错误匹配）
    }

    return score;
}

void CascadeMatchTracker::track(const std::vector<SingleDetectionResult>& detections, float dt) {
    // ==========================================
    // Step 1: 更新 BotID 轨迹池
    // ==========================================
    for (const auto& det : detections) {
        if (det.bot_id == -1) continue;
        if (bot_id_trajectories.find(det.bot_id) == bot_id_trajectories.end()) {
            bot_id_trajectories[det.bot_id] = BotIdTrack(det.bot_id, det.class_id, 1.0f);
        } else {
            bot_id_trajectories[det.bot_id].update(det.class_id, 1.0f);
        }
        bot_id_trajectories[det.bot_id].updated = true;
    }

    for (auto it = bot_id_trajectories.begin(); it != bot_id_trajectories.end();) {
        if (!it->second.updated) {
            it->second.update(-1, 1.0f);
            if (++it->second.lost_counter >= 30) { 
                it = bot_id_trajectories.erase(it); 
                continue; 
            }
        }
        it->second.updated = false; 
        ++it;
    }

    // ==========================================
    // Step 2: Kalman预测 - 所有非INACTIVE轨迹先预测
    // ==========================================
    for (auto& track : tracks) {
        if (track.state != TrackState::INACTIVE) {
            // 像素框预测
            auto xywh = track.kalman_box.predict(dt);
            track.car_box = cv::Rect2f(
                xywh[0] - xywh[2]/2.0f, 
                xywh[1] - xywh[3]/2.0f, 
                xywh[2], 
                xywh[3]
            );
            
            // 物理坐标预测
            auto pos2d = track.kalman_2d.predict(dt);
            track.pos_2d_uwb = cv::Point2f(pos2d[0], pos2d[1]);
        }
    }

    // ==========================================
    // Step 3: 构建代价矩阵进行匈牙利匹配
    // ==========================================
    int M = tracks.size(); 
    int N = detections.size();
    std::vector<std::vector<float>> cost_matrix(M, std::vector<float>(N, 1e5f)); // 默认高代价

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (tracks[i].state == TrackState::INACTIVE) {
                // 【冷启动】：计算匹配分数，但放宽距离限制
                cost_matrix[i][j] = -1.0f * compute_score(tracks[i], detections[j]);
            } else {
                // 【追踪中】：5K分辨率下的空间保护逻辑
                float cx_t = tracks[i].car_box.x + tracks[i].car_box.width / 2.0f;
                float cy_t = tracks[i].car_box.y + tracks[i].car_box.height / 2.0f;
                float cx_d = detections[j].car_box.x + detections[j].car_box.width / 2.0f;
                float cy_d = detections[j].car_box.y + detections[j].car_box.height / 2.0f;
                
                float pixel_dist = std::hypot(cx_t - cx_d, cy_t - cy_d);
                if (pixel_dist > 1500.0f) {
                    cost_matrix[i][j] = 1e5; // 超过 1500 像素强制截断
                } else {
                    cost_matrix[i][j] = -1.0f * compute_score(tracks[i], detections[j]);
                }
            }
        }
    }

    // ==========================================
    // Step 4: 匈牙利匹配
    // ==========================================
    std::vector<int> matched_indices;
    HungarianAlgorithm hungarian;
    hungarian.Solve(cost_matrix, matched_indices);

    std::vector<std::pair<int, int>> matches;
    for (int i = 0; i < M; ++i) {
        int j = matched_indices[i];
        if (j >= 0 && j < N) {
            float thresh = (tracks[i].state == TrackState::INACTIVE) ? INACTIVE_COST_THRESHOLD : COST_THRESHOLD;
            if (cost_matrix[i][j] < thresh) matches.push_back({i, j});
        }
    }

    std::vector<bool> track_matched(M, false);
    std::vector<bool> det_matched(N, false);
    for (auto& m : matches) {
        track_matched[m.first] = true;
        det_matched[m.second] = true;
    }

    // ==========================================
    // Step 5: 根据匹配结果更新状态机
    // ==========================================
    for (auto& m : matches) {
        auto& track = tracks[m.first];
        auto& det = detections[m.second];
        
        // 将检测框转为center-based格式 [cx, cy, w, h]
        std::vector<float> det_xywh = {
            det.car_box.x + det.car_box.width/2.0f, 
            det.car_box.y + det.car_box.height/2.0f, 
            det.car_box.width, 
            det.car_box.height
        };
        std::vector<float> det_pos2d = {det.pos_3d.x, det.pos_3d.y};

        // 死亡标记（class_id >= 10 表示死亡）
        if (det.class_id >= 10) track.is_dead = true;

        switch (track.state) {
            case TrackState::INACTIVE:
                // 冷启动：转为TENTATIVE
                track.state = TrackState::TENTATIVE;
                track.hit_count = 1;  // 第一帧命中
                track.miss_count = 0;
                track.bot_id = det.bot_id;
                track.pos_3d = det.pos_3d;
                track.pos_2d_uwb_det = cv::Point2f(det_pos2d[0], det_pos2d[1]);
                // 初始化卡尔曼滤波器
                track.kalman_box.reset(det_xywh);
                track.kalman_2d.reset(det_pos2d);
                // TENTATIVE期间不设置为active，等待确认
                track.is_active = false;
                break;
                
            case TrackState::TENTATIVE:
                track.hit_count++;
                track.miss_count = 0;
                track.bot_id = det.bot_id;
                track.pos_3d = det.pos_3d;
                track.pos_2d_uwb_det = cv::Point2f(det_pos2d[0], det_pos2d[1]);
                
                // 更新卡尔曼滤波器
                track.kalman_box.update(det_xywh);
                track.kalman_2d.update(det_pos2d);
                
                if (track.hit_count >= HIT_COUNT_THRESHOLD) {
                    // 转为CONFIRMED状态
                    track.state = TrackState::CONFIRMED;
                    track.is_active = true;
                }
                break;
                
            case TrackState::CONFIRMED:
            case TrackState::LOST:
                track.hit_count++;
                track.miss_count = 0;
                track.is_active = true;
                
                // BotID跳变检测（如果bot_id变化了，可能是换人）
                if (track.bot_id != -1 && track.bot_id != det.bot_id) {
                    // BotID变化，重新初始化
                    track.kalman_box.reset(det_xywh);
                    track.kalman_2d.reset(det_pos2d);
                } else {
                    // 正常update
                    track.kalman_box.update(det_xywh);
                    
                    // 物理坐标防跳变（参考Python版OUTLIER_THRESHOLD=1.5m）
                    float d = std::hypot(
                        track.pos_2d_uwb.x - det_pos2d[0], 
                        track.pos_2d_uwb.y - det_pos2d[1]
                    );
                    if (d < 1.5f) {
                        track.kalman_2d.update(det_pos2d);
                    }
                }
                
                track.bot_id = det.bot_id;
                track.pos_3d = det.pos_3d;
                track.pos_2d_uwb_det = cv::Point2f(det_pos2d[0], det_pos2d[1]);
                track.state = TrackState::CONFIRMED;
                break;
        }
        
        // 更新car_box为当前检测值（用于可视化）
        track.car_box = det.car_box;
    }

    // ==========================================
    // Step 6: 处理未匹配的轨迹
    // ==========================================
    for (int i = 0; i < M; ++i) {
        if (track_matched[i]) continue;
        
        auto& track = tracks[i];
        switch (track.state) {
            case TrackState::INACTIVE:
                track.is_active = false;
                track.inactive_count++;
                break;
                
            case TrackState::TENTATIVE:
                track.is_active = false;
                track.hit_count--;
                if (track.hit_count <= 0) {
                    track.state = TrackState::INACTIVE;
                    track.hit_count = 0;
                }
                break;
                
            case TrackState::CONFIRMED:
                track.state = TrackState::LOST;
                track.miss_count = 1;
                // LOST状态仍保持active，使用预测值
                break;
                
            case TrackState::LOST:
                track.miss_count++;
                if (track.miss_count >= MISS_COUNT_THRESHOLD) {
                    // 丢失太久，重置为INACTIVE
                    track.state = TrackState::INACTIVE;
                    track.is_active = false;
                    track.bot_id = -1;
                    track.hit_count = 0;
                    track.miss_count = 0;
                    // 重置卡尔曼滤波器
                    track.kalman_box.reset();
                    track.kalman_2d.reset();
                }
                break;
        }
    }

    // ==========================================
    // Step 7: 处理未匹配的检测结果（可选：作为新轨迹）
    // ==========================================
    // 注意：当前实现中10个兵种是固定的，不需要创建新轨迹
    // 但如果检测到未匹配的已知装甲板，可能需要考虑更换绑定
}

} // namespace tracker
} // namespace radar_core
