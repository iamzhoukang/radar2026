#pragma once
#include <Eigen/Dense>
#include <vector>

namespace radar_core {
namespace tracker {

// ==========================================
// KalmanFilterBox - 8维状态：像素框 [cx, cy, w, h, vx, vy, vw, vh]
// 使用固定大小矩阵，栈分配，无动态内存分配
// ==========================================
class KalmanFilterBox {
public:
    /**
     * @param dt 默认时间步长
     * @param q_std 过程噪声标准差 (默认2.0)
     * @param r_std 测量噪声标准差 (默认1.0)
     */
    KalmanFilterBox(float dt = 1.0f, float q_std = 2.0f, float r_std = 1.0f);
    
    /**
     * 预测下一状态
     * @param dt 时间步长，<=0时使用默认值
     * @return 预测的位置 [cx, cy, w, h]
     */
    std::vector<float> predict(float dt = -1.0f);
    
    /**
     * 用测量值更新滤波器
     * @param bbox 测量的边界框 [cx, cy, w, h]
     * @return 更新后的位置 [cx, cy, w, h]
     */
    std::vector<float> update(const std::vector<float>& bbox);
    
    /**
     * 重置滤波器
     * @param initial_bbox 初始边界框 [cx, cy, w, h]，空则使用默认值
     */
    void reset(const std::vector<float>& initial_bbox = {});
    
    /**
     * 获取完整状态向量
     * @return [cx, cy, w, h, vx, vy, vw, vh]
     */
    std::vector<float> get_state() const;

    // 状态向量: [cx, cy, w, h, vx, vy, vw, vh]
    Eigen::Matrix<float, 8, 1> x;
    Eigen::Matrix<float, 8, 8> P;  // 误差协方差
    Eigen::Matrix<float, 8, 8> F;  // 状态转移矩阵
    Eigen::Matrix<float, 4, 8> H;  // 观测矩阵 (4维观测)
    Eigen::Matrix<float, 4, 4> R;  // 测量噪声协方差
    Eigen::Matrix<float, 8, 8> Q;  // 过程噪声协方差

private:
    float dt_, q_std_, r_std_;
    void updateQ(float dt);
};

// ==========================================
// KalmanFilter2d - 4维状态：物理坐标 [x, y, vx, vy]
// 使用固定大小矩阵，栈分配，无动态内存分配
// ==========================================
class KalmanFilter2d {
public:
    /**
     * @param q_std 过程噪声标准差 (默认2.0)
     * @param r_std 测量噪声标准差 (默认1.0)
     * @param dt 默认时间步长 (默认0.1)
     */
    KalmanFilter2d(float q_std = 2.0f, float r_std = 1.0f, float dt = 0.1f);
    
    /**
     * 预测下一状态
     * @param dt 时间步长，<=0时使用默认值
     * @return 预测的位置 [x, y]
     */
    std::vector<float> predict(float dt = -1.0f);
    
    /**
     * 用测量值更新滤波器
     * @param pos 测量的位置 [x, y]
     * @return 更新后的位置 [x, y]
     */
    std::vector<float> update(const std::vector<float>& pos);
    
    /**
     * 重置滤波器
     * @param initial_pos 初始位置 [x, y]，空则使用默认值
     */
    void reset(const std::vector<float>& initial_pos = {});
    
    /**
     * 获取当前位置估计
     * @return [x, y]
     */
    std::vector<float> get_position() const;
    
    /**
     * 获取当前速度估计
     * @return [vx, vy]
     */
    std::vector<float> get_velocity() const;

    // 状态向量: [x, y, vx, vy]
    Eigen::Matrix<float, 4, 1> x;
    Eigen::Matrix<float, 4, 4> P;
    Eigen::Matrix<float, 4, 4> F;
    Eigen::Matrix<float, 2, 4> H;  // 观测矩阵 (2维观测)
    Eigen::Matrix<float, 2, 2> R;
    Eigen::Matrix<float, 4, 4> Q;

private:
    float dt_, q_std_, r_std_;
};

} // namespace tracker
} // namespace radar_core
