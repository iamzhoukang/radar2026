#include "solver/pnp_solver.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>

namespace radar_core {
namespace solver {

bool PnPSolver::loadConfig(const std::string& yaml_path, const std::string& txt_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        auto k = config["camera"]["K"].as<std::vector<double>>();
        auto d = config["camera"]["dist"].as<std::vector<double>>();

        // 使用 clone 深拷贝，避免后续修改影响原数据
        K_ = cv::Mat(3, 3, CV_64F, k.data()).clone();
        D_ = cv::Mat(1, 5, CV_64F, d.data()).clone();
    } catch (...) {
        return false;
    }

    std::ifstream file(txt_path);
    if (!file.is_open()) return false;

    world_points_.clear();
    float x, y, z;
    while (file >> x >> y >> z) {
        world_points_.emplace_back(x, y, z);
    }
    
    // PnP 算法数学上至少需要 4 个点
    return (world_points_.size() >= 4);
}

PnPResult PnPSolver::solve(const std::vector<cv::Point2f>& image_points) const {
    PnPResult result;

    if (image_points.size() != world_points_.size() || world_points_.empty()) {
        result.success = false;
        return result;
    }

    bool is_solved = cv::solvePnP(
        world_points_,
        image_points,
        K_,
        D_,
        result.rvec,
        result.tvec,
        false,                    // 不使用传入的 rvec/tvec 作为初值
        cv::SOLVEPNP_ITERATIVE    // 非线性优化迭代算法
    );

    if (is_solved) {
        result.success = true;
        result.reprojection_error = calculateReprojectionError(result.rvec, result.tvec, image_points);
    }
    
    return result;
}

std::vector<cv::Point2f> PnPSolver::projectPoints(const cv::Mat& rvec, const cv::Mat& tvec) const {
    std::vector<cv::Point2f> projected;
    // 顺向物理投影，将 3D 点投射到 2D 平面
    cv::projectPoints(world_points_, rvec, tvec, K_, D_, projected);
    return projected;
}

double PnPSolver::calculateReprojectionError(const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point2f>& image_points) const {
    // 复用刚刚补全的 projectPoints 函数获取理论点
    std::vector<cv::Point2f> projected_points = projectPoints(rvec, tvec);

    double total_error = 0.0;
    for (size_t i = 0; i < world_points_.size(); ++i) {
        // 使用 L2 范数（欧氏距离）计算物理点击点与理论投影点的距离
        total_error += cv::norm(image_points[i] - projected_points[i]); 
    }
    
    return total_error / world_points_.size();
}

bool PnPSolver::saveExtrinsics(const std::string& yaml_path, const cv::Mat& rvec, const cv::Mat& tvec) const {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        config["camera"]["rvec"] = std::vector<double>{rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
        config["camera"]["tvec"] = std::vector<double>{tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};

        std::ofstream fout(yaml_path);
        fout << config;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace solver
} // namespace radar_core