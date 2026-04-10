#include "detector/outpost_config_ui.hpp"
#include <iostream>

namespace radar_core {
namespace ui {

OutpostConfigUI::OutpostConfigUI(const std::string& window_name) 
    : window_name_(window_name) {}

OutpostConfigUI::~OutpostConfigUI() {
    stop();
}

void OutpostConfigUI::start(const cv::Mat& background_image) {
    if (background_image.empty()) return;
    
    // 保存原始图像
    raw_img_ = background_image.clone();
    config_roi_ = cv::Rect(100, 100, 200, 200);  // 默认ROI
    is_dragging_ = false;
    saved_ = false;
    is_active_ = true;

    // 计算缩放比例（参考CalibrationUI，目标宽度1280）
    const float target_width = 1280.0f;
    scale_ = raw_img_.cols > target_width ? target_width / (float)raw_img_.cols : 1.0f;
    
    // 创建窗口
    cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name_, onMouseWrapper, this);
    
    std::cout << "[OutpostConfigUI] 启动配置窗口，图像缩放比例: " << scale_ << std::endl;
}

void OutpostConfigUI::stop() {
    if (is_active_) {
        is_active_ = false;
        cv::destroyWindow(window_name_);
        // 强制处理所有窗口事件
        for(int i = 0; i < 10; ++i) {
            cv::waitKey(1);
        }
        // 释放图像资源
        raw_img_.release();
        display_img_.release();
    }
}

bool OutpostConfigUI::run() {
    if (!is_active_ || raw_img_.empty()) return false;
    
    // 使用更短的延迟，减少性能影响
    while (is_active_) {
        render();
        
        char key = cv::waitKey(5) & 0xFF;  // 从30ms改为5ms
        
        if (key == 's' || key == 'S') {
            // 保存：将显示坐标转换回原始坐标
            final_roi_ = cv::Rect(
                (int)(config_roi_.x / scale_),
                (int)(config_roi_.y / scale_),
                (int)(config_roi_.width / scale_),
                (int)(config_roi_.height / scale_)
            );
            saved_ = true;
            std::cout << "[OutpostConfigUI] ROI已保存: [" 
                      << final_roi_.x << ", " << final_roi_.y << ", "
                      << final_roi_.width << ", " << final_roi_.height << "]" << std::endl;
            break;
        } else if (key == 'q' || key == 'Q' || key == 27) { // ESC
            std::cout << "[OutpostConfigUI] 取消配置" << std::endl;
            saved_ = false;
            break;
        } else if (key == 'r' || key == 'R') {
            // 重置为默认
            config_roi_ = cv::Rect(100, 100, 200, 200);
        }
    }
    
    stop();
    return saved_;
}

void OutpostConfigUI::render() {
    if (!is_active_ || raw_img_.empty()) return;
    
    // 缩放图像以适应屏幕
    cv::resize(raw_img_, display_img_, cv::Size(), scale_, scale_);
    
    // 在缩放后的图像上绘制ROI（绿色框）
    cv::rectangle(display_img_, config_roi_, cv::Scalar(0, 255, 0), 2);
    
    // 绘制中心点
    cv::circle(display_img_, 
        cv::Point(config_roi_.x + config_roi_.width/2, config_roi_.y + config_roi_.height/2),
        5, cv::Scalar(0, 0, 255), -1);
    
    // 绘制文字提示
    cv::putText(display_img_, "Drag to select ROI", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(display_img_, "S:Save  Q:Cancel  R:Reset", cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // 显示当前ROI坐标（原始图像坐标）
    cv::Rect raw_roi(
        (int)(config_roi_.x / scale_),
        (int)(config_roi_.y / scale_),
        (int)(config_roi_.width / scale_),
        (int)(config_roi_.height / scale_)
    );
    std::string roi_text = cv::format("ROI: [%d, %d, %d, %d]", 
        raw_roi.x, raw_roi.y, raw_roi.width, raw_roi.height);
    cv::putText(display_img_, roi_text, cv::Point(10, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    
    cv::imshow(window_name_, display_img_);
}

void OutpostConfigUI::onMouseWrapper(int event, int x, int y, int flags, void* userdata) {
    OutpostConfigUI* ui = static_cast<OutpostConfigUI*>(userdata);
    ui->handleMouse(event, x, y);
}

void OutpostConfigUI::handleMouse(int event, int x, int y) {
    if (!is_active_) return;
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        is_dragging_ = true;
        drag_start_ = cv::Point(x, y);
        config_roi_ = cv::Rect(x, y, 0, 0);
    } else if (event == cv::EVENT_MOUSEMOVE && is_dragging_) {
        config_roi_.x = std::min(drag_start_.x, x);
        config_roi_.y = std::min(drag_start_.y, y);
        config_roi_.width = std::abs(x - drag_start_.x);
        config_roi_.height = std::abs(y - drag_start_.y);
        
        // 限制最小尺寸
        if (config_roi_.width < 5) config_roi_.width = 5;
        if (config_roi_.height < 5) config_roi_.height = 5;
    } else if (event == cv::EVENT_LBUTTONUP) {
        is_dragging_ = false;
    }
}

} // namespace ui
} // namespace radar_core
