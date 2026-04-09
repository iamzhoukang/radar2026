#include "tracker/kalman.hpp"
#include <cmath>

namespace radar_core {
namespace tracker {

// ==========================================
// KalmanFilterBox (对应像素框)
// ==========================================
KalmanFilterBox::KalmanFilterBox(float dt, float q_std, float r_std) 
    : dt_(dt), q_std_(q_std), r_std_(r_std)
{
    x = Eigen::VectorXf::Zero(8);
    P = Eigen::MatrixXf::Identity(8, 8) * 100.0f;
    
    F = Eigen::MatrixXf::Identity(8, 8);
    // 位置 += 速度 * dt
    for(int i=0; i<4; i++) F(i, i+4) = dt;

    // 观测矩阵：只观测位置 [cx, cy, w, h]
    H = Eigen::MatrixXf::Zero(4, 8);
    H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1; H(3, 3) = 1;

    // 测量噪声协方差
    R = Eigen::MatrixXf::Identity(4, 4) * (r_std * r_std);
    
    // 过程噪声协方差（初始值，会在predict中根据dt重新计算）
    Q = Eigen::MatrixXf::Zero(8, 8);
    
    reset();
}

void KalmanFilterBox::updateQ(float dt) {
    float dt2 = dt * dt;
    float dt3 = dt2 * dt / 2.0f;
    float dt4 = dt2 * dt2 / 4.0f;
    
    // 连续时间白噪声模型：Q = q_std^2 * [dt^4/4, dt^3/2; dt^3/2, dt^2]
    // 对每个维度（x, y, w, h）独立计算
    for (int i = 0; i < 4; i++) {
        Q(i, i) = dt4;           // 位置-位置
        Q(i, i+4) = dt3;         // 位置-速度
        Q(i+4, i) = dt3;         // 速度-位置
        Q(i+4, i+4) = dt2;       // 速度-速度
    }
    Q *= (q_std_ * q_std_);
}

std::vector<float> KalmanFilterBox::predict(float dt) {
    float use_dt = (dt > 0) ? dt : dt_;
    
    // 更新状态转移矩阵中的dt
    for(int i=0; i<4; i++) F(i, i+4) = use_dt;
    
    // 根据当前dt更新Q矩阵
    updateQ(use_dt);
    
    x = F * x;
    P = F * P * F.transpose() + Q;
    return {x(0), x(1), x(2), x(3)};
}

std::vector<float> KalmanFilterBox::update(const std::vector<float>& bbox) {
    // 港科大防跳变逻辑: Detect jumping (5K分辨率下使用100像素阈值)
    if (std::abs(x(0) - bbox[0]) > 100.0f || std::abs(x(1) - bbox[1]) > 100.0f) {
        reset(bbox);
        return bbox;
    }
    
    Eigen::VectorXf z(4); 
    z << bbox[0], bbox[1], bbox[2], bbox[3];
    
    Eigen::MatrixXf S = H * P * H.transpose() + R;
    Eigen::MatrixXf K = P * H.transpose() * S.inverse();
    
    x = x + K * (z - H * x);
    P = (Eigen::MatrixXf::Identity(8, 8) - K * H) * P;
    
    return {x(0), x(1), x(2), x(3)};
}

void KalmanFilterBox::reset(const std::vector<float>& initial_bbox) {
    x.setZero();
    if (!initial_bbox.empty() && initial_bbox.size() == 4) {
        x(0) = initial_bbox[0];  // cx
        x(1) = initial_bbox[1];  // cy
        x(2) = initial_bbox[2];  // w
        x(3) = initial_bbox[3];  // h
    } else {
        x(0) = 0.0f; x(1) = 0.0f; x(2) = 1.0f; x(3) = 1.0f;
    }
    // 速度初始化为0
    x(4) = 0.0f; x(5) = 0.0f; x(6) = 0.0f; x(7) = 0.0f;
    
    P = Eigen::MatrixXf::Identity(8, 8) * 100.0f;
}

std::vector<float> KalmanFilterBox::get_state() {
    return {x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7)};
}

// ==========================================
// KalmanFilter2d (对应物理坐标)
// ==========================================
KalmanFilter2d::KalmanFilter2d(float q_std, float r_std, float dt) 
    : dt_(dt), q_std_(q_std), r_std_(r_std) 
{
    x = Eigen::VectorXf::Zero(4);
    P = Eigen::MatrixXf::Identity(4, 4) * 100.0f;
    
    F = Eigen::MatrixXf::Identity(4, 4);
    F(0, 2) = dt; F(1, 3) = dt;

    H = Eigen::MatrixXf::Zero(2, 4);
    H(0, 0) = 1; H(1, 1) = 1;

    R = Eigen::MatrixXf::Identity(2, 2) * (r_std * r_std);
    Q = Eigen::MatrixXf::Zero(4, 4);
    
    reset();
}

std::vector<float> KalmanFilter2d::predict(float dt) {
    float use_dt = (dt > 0) ? dt : dt_;
    float dt2 = use_dt * use_dt;
    float dt3 = dt2 * use_dt / 2.0f;
    float dt4 = dt2 * dt2 / 4.0f;

    F(0, 2) = use_dt; F(1, 3) = use_dt;
    
    // 港科大 Q 矩阵数学推导 (连续时间白噪声模型)
    Q << dt4, 0, dt3, 0,
         0, dt4, 0, dt3,
         dt3, 0, dt2, 0,
         0, dt3, 0, dt2;
    Q *= (q_std_ * q_std_);

    x = F * x;
    P = F * P * F.transpose() + Q;
    return {x(0), x(1)};
}

std::vector<float> KalmanFilter2d::update(const std::vector<float>& pos) {
    Eigen::VectorXf z(2); z << pos[0], pos[1];
    Eigen::MatrixXf S = H * P * H.transpose() + R;
    Eigen::MatrixXf K = P * H.transpose() * S.inverse();
    
    x = x + K * (z - H * x);
    P = (Eigen::MatrixXf::Identity(4, 4) - K * H) * P;
    return {x(0), x(1)};
}

void KalmanFilter2d::reset(const std::vector<float>& initial_pos) {
    x.setZero();
    if (!initial_pos.empty() && initial_pos.size() >= 2) {
        x(0) = initial_pos[0];
        x(1) = initial_pos[1];
    }
    // 速度初始化为0
    x(2) = 0.0f; x(3) = 0.0f;
    
    P = Eigen::MatrixXf::Identity(4, 4) * 100.0f;
}

std::vector<float> KalmanFilter2d::get_position() {
    return {x(0), x(1)};
}

std::vector<float> KalmanFilter2d::get_velocity() {
    return {x(2), x(3)};
}

} // namespace tracker
} // namespace radar_core
