#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>

namespace radar_core {
namespace ui {

// 前哨站ROI配置UI类（类似CalibrationUI的封装方式）
class OutpostConfigUI {
public:
    explicit OutpostConfigUI(const std::string& window_name = "Outpost ROI Config");
    ~OutpostConfigUI();

    // 启动配置界面，传入当前帧图像（用于显示背景）
    void start(const cv::Mat& background_image);
    
    // 停止配置
    void stop();
    
    // 运行配置循环（阻塞式，在新线程中调用）
    // 返回是否成功保存
    bool run();
    
    // 获取配置的ROI（应在run返回true后调用）
    cv::Rect getROI() const { return final_roi_; }

private:
    std::string window_name_;
    cv::Mat raw_img_;           // 原始图像
    cv::Mat display_img_;       // 显示用图像（已缩放）
    cv::Rect config_roi_;       // 当前配置的ROI
    cv::Point drag_start_;      // 拖动起点
    bool is_dragging_ = false;
    std::atomic<bool> is_active_{false};
    float scale_ = 1.0f;        // 缩放比例
    
    cv::Rect final_roi_;        // 最终保存的ROI
    bool saved_ = false;
    
    // 鼠标回调
    static void onMouseWrapper(int event, int x, int y, int flags, void* userdata);
    void handleMouse(int event, int x, int y);
    
    // 渲染一帧
    void render();
};

} // namespace ui
} // namespace radar_core
