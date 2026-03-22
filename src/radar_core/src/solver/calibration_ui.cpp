#include "solver/calibration_ui.hpp"

namespace radar_core {
namespace ui {

CalibrationUI::CalibrationUI(const std::string& window_name) 
    : window_name_(window_name) {}

CalibrationUI::~CalibrationUI() {
    stop();
}

void CalibrationUI::start(const cv::Mat& raw_image) {
    if (raw_image.empty()) return;
    
    raw_img_ = raw_image.clone();
    clicked_points_.clear();
    is_zoomed_ = false;
    is_active_ = true;

    cv::namedWindow(window_name_);
    // 绑定鼠标事件到当前 UI 实例
    cv::setMouseCallback(window_name_, onMouseWrapper, this);
}

void CalibrationUI::stop() {
    if (is_active_) {
        cv::destroyWindow(window_name_);
        is_active_ = false;
        raw_img_.release();
        display_img_.release();
        for(int i = 0; i < 5; ++i) cv::waitKey(10); // 确保窗口彻底销毁
    }
}

UIAction CalibrationUI::spinOnce() {
    if (!is_active_ || raw_img_.empty()) return UIAction::NONE;

    int key = cv::waitKey(1) & 0xFF;

    if (is_zoomed_) {
        if (key == 'c' || key == 'C') return UIAction::TRIGGER_CLEAR;
        return UIAction::NONE;
    }

    // 计算缩放比例并显示全局预览
    float target_width = 1280.0f;
    scale_ = raw_img_.cols > target_width ? target_width / (float)raw_img_.cols : 1.0f;
    cv::resize(raw_img_, display_img_, cv::Size(), scale_, scale_);

    // 绘制已点击的点
    for (size_t i = 0; i < clicked_points_.size(); ++i) {
        cv::Point2f display_pt = clicked_points_[i] * scale_;
        cv::circle(display_img_, display_pt, 4, cv::Scalar(0, 0, 255), -1); 
        cv::putText(display_img_, std::to_string(i + 1), display_pt + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow(window_name_, display_img_);

    // 向上层报告用户的按键意图
    if (key == 's' || key == 'S') return UIAction::TRIGGER_SOLVE;
    if (key == 'c' || key == 'C') return UIAction::TRIGGER_CLEAR;
    
    return UIAction::NONE;
}

void CalibrationUI::showFeedback(const std::vector<cv::Point2f>& projected_points, double error) {
    if (is_zoomed_) is_zoomed_ = false;
    cv::resize(raw_img_, display_img_, cv::Size(), scale_, scale_);

    for (const auto& pt : projected_points) {
        cv::circle(display_img_, pt * scale_, 8, cv::Scalar(0, 255, 0), 2);
    }
    
    std::string err_str = "Saved! Error: " + std::to_string(error).substr(0, 4);
    cv::putText(display_img_, err_str, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow(window_name_, display_img_);
    cv::waitKey(1500); 
}

void CalibrationUI::onMouseWrapper(int event, int x, int y, int flags, void* userdata) {
    CalibrationUI* ui = static_cast<CalibrationUI*>(userdata);
    ui->handleMouse(event, x, y, flags);
}

void CalibrationUI::handleMouse(int event, int x, int y, int flags) {
    if (raw_img_.empty()) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (!is_zoomed_) {
            if (clicked_points_.size() >= 6) {
                return; // 点满了，拒绝进入放大镜
            }
            float raw_center_x = x / scale_;
            float raw_center_y = y / scale_;

            int roi_w = 600, roi_h = 600;
            int roi_x = std::max(0, (int)raw_center_x - roi_w / 2);
            int roi_y = std::max(0, (int)raw_center_y - roi_h / 2);
            roi_x = std::min(roi_x, raw_img_.cols - roi_w);
            roi_y = std::min(roi_y, raw_img_.rows - roi_h);

            zoom_roi_ = cv::Rect(roi_x, roi_y, roi_w, roi_h);
            is_zoomed_ = true;
            display_img_ = raw_img_(zoom_roi_).clone();
            cv::putText(display_img_, "[ZOOM_MODE]", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::imshow(window_name_, display_img_);
        } else {
            float final_x = zoom_roi_.x + x;
            float final_y = zoom_roi_.y + y;
            clicked_points_.push_back(cv::Point2f(final_x, final_y));
            is_zoomed_ = false;
        }
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        if (is_zoomed_) is_zoomed_ = false;
        else if (!clicked_points_.empty()) clicked_points_.pop_back();
    }
}

} // namespace ui
} // namespace radar_core