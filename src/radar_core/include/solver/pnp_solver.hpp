#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace radar_core {
namespace solver {

struct PnPResult {
    bool success = false;
    cv::Mat rvec;
    cv::Mat tvec;
    double reprojection_error = 0.0;
};

class PnPSolver {
public:
    PnPSolver() = default;
    ~PnPSolver() = default;

    // 禁用拷贝构造和赋值，防止 cv::Mat 浅拷贝导致内存污染
    PnPSolver(const PnPSolver&) = delete;
    PnPSolver& operator=(const PnPSolver&) = delete;

    // 初始化与核心解算
    bool loadConfig(const std::string& yaml_path, const std::string& txt_path);
    PnPResult solve(const std::vector<cv::Point2f>& image_points) const;
    bool saveExtrinsics(const std::string& yaml_path, const cv::Mat& rvec, const cv::Mat& tvec) const;

    // 正向物理投影（补全的函数，用于给 UI 提供反馈点）
    std::vector<cv::Point2f> projectPoints(const cv::Mat& rvec, const cv::Mat& tvec) const;

    // 暴露只读数据供外部获取
    const cv::Mat& getK() const { return K_; }
    const cv::Mat& getD() const { return D_; }
    const std::vector<cv::Point3f>& getWorldPoints() const { return world_points_; }

private:
    // 计算重投影误差（内部调用）
    double calculateReprojectionError(const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point2f>& image_points) const;

    cv::Mat K_;                            
    cv::Mat D_;                            
    std::vector<cv::Point3f> world_points_; 
};

} // namespace solver
} // namespace radar_core