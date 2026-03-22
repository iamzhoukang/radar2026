#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace radar_core {
namespace ui {

// 定义 UI 会向外部发出的动作指令
enum class UIAction {
    NONE,           // 无事发生
    TRIGGER_SOLVE,  // 用户按下了 's'，请求解算
    TRIGGER_CLEAR   // 用户按下了 'c'，请求清空
};

class CalibrationUI {
public:
    explicit CalibrationUI(const std::string& window_name = "Calibration Tool");
    ~CalibrationUI();

    // 启动 UI 引擎并载入图像
    void start(const cv::Mat& raw_image);
    
    // UI 渲染与事件循环（每次定时器触发时调用一次）
    UIAction spinOnce();
    
    // 展示解算成功后的反馈（绿色圆圈和误差）
    void showFeedback(const std::vector<cv::Point2f>& projected_points, double error);
    
    // 关闭界面
    void stop();

    // 数据获取与操作
    const std::vector<cv::Point2f>& getClickedPoints() const { return clicked_points_; }
    void clearPoints() { clicked_points_.clear(); is_zoomed_ = false; }
    bool isActive() const { return is_active_; }

private:
    // 静态鼠标回调包装器
    static void onMouseWrapper(int event, int x, int y, int flags, void* userdata);
    void handleMouse(int event, int x, int y, int flags);

    std::string window_name_;
    bool is_active_ = false;

    cv::Mat raw_img_;
    cv::Mat display_img_;
    std::vector<cv::Point2f> clicked_points_;
    
    float scale_ = 1.0f;
    bool is_zoomed_ = false;
    cv::Rect zoom_roi_;
};

} // namespace ui
} // namespace radar_core