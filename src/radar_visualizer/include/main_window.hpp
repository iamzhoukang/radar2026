#pragma once

#include <QMainWindow>
#include <QImage>
#include <QPaintEvent>
#include <QWidget>
#include <memory>

// 前向声明，加快编译速度
class QPushButton;
namespace radar_visualizer {
    class RosWorker;
}

namespace radar_visualizer {

// 自定义极速渲染画板 (替代低效的 QLabel)
class VideoWidget : public QWidget {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget* parent = nullptr);
    
public slots:
    // 接收图像的槽函数
    void updateImage(const QImage& img);

protected:
    // 重写底层绘图事件：绕过 QPixmap 的转换，直接把图像拍在显存/屏幕上！
    void paintEvent(QPaintEvent* event) override;

private:
    QImage current_image_;
};

// 真正的 MainWindow 
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(std::shared_ptr<RosWorker> worker, QWidget *parent = nullptr);
    ~MainWindow() override = default;

private slots:
    // 按钮按下时的动画与状态锁定
    void onCalibClicked();
    void onFlipClicked();
    void onOutpostConfigClicked();  // 【新增】前哨站ROI配置

private:
    // 初始化 UI 布局
    void setupUi();

    std::shared_ptr<RosWorker> worker_;
    
    VideoWidget* video_widget_;
    VideoWidget* map_widget_;
    
    QPushButton* calib_btn_;
    QPushButton* flip_btn_;
    QPushButton* outpost_config_btn_;  // 【新增】前哨站配置按钮
};

} // namespace radar_visualizer