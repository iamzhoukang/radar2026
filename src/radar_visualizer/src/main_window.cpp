#include "main_window.hpp"
#include "ros_worker.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QPainter>
#include <QTimer>

namespace radar_visualizer {

// ==========================================
// VideoWidget 极速画板实现
// ==========================================
VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent) {
    // 设置背景为纯黑，防闪烁
    setAttribute(Qt::WA_OpaquePaintEvent);
    setStyleSheet("background-color: black;");
}

void VideoWidget::updateImage(const QImage& img) {
    current_image_ = img;
    update(); // 触发底层 paintEvent 刷新屏幕
}

void VideoWidget::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    
    // 如果还没收到图，画个提示语
    if (current_image_.isNull()) {
        painter.setPen(Qt::white);
        painter.setFont(QFont("Arial", 20, QFont::Bold));
        painter.drawText(rect(), Qt::AlignCenter, "Waiting for Signal...");
        return;
    }

    // 【性能核心】：使用 FastTransformation，放弃极其耗 CPU 的平滑插值，把算力省给 ROS
    QImage scaled_img = current_image_.scaled(size(), Qt::KeepAspectRatio, Qt::FastTransformation);
    
    // 计算居中偏移量
    int x = (width() - scaled_img.width()) / 2;
    int y = (height() - scaled_img.height()) / 2;
    
    // 直接把像素拍到显存/屏幕上
    painter.drawImage(x, y, scaled_img);
}

// ==========================================
// MainWindow 主界面实现
// ==========================================
MainWindow::MainWindow(std::shared_ptr<RosWorker> worker, QWidget *parent)
    : QMainWindow(parent), worker_(worker) 
{
    setupUi();

    // 【架构核心】：QueuedConnection 跨线程通信！
    // 工作线程 (ROS) 发射信号，主线程 (UI) 排队接收，完美解耦，绝对不死锁！
    connect(worker_.get(), &RosWorker::videoReady, video_widget_, &VideoWidget::updateImage, Qt::QueuedConnection);
    connect(worker_.get(), &RosWorker::mapReady, map_widget_, &VideoWidget::updateImage, Qt::QueuedConnection);
}

void MainWindow::setupUi() {
    QWidget *central_widget = new QWidget(this);
    setCentralWidget(central_widget);
    QHBoxLayout *main_layout = new QHBoxLayout(central_widget);

    // 1. 左侧视频区 (使用我们的极速画板)
    video_widget_ = new VideoWidget(this);
    video_widget_->setMinimumSize(800, 600);
    main_layout->addWidget(video_widget_, 7);

    // 2. 右侧控制区
    QVBoxLayout *right_layout = new QVBoxLayout();
    
    map_widget_ = new VideoWidget(this);
    map_widget_->setMinimumSize(350, 600);
    right_layout->addWidget(map_widget_, 8);

    // 按钮组
    QHBoxLayout *btn_layout = new QHBoxLayout();
    calib_btn_ = new QPushButton(" 触发标定", this);
    calib_btn_->setFixedHeight(50);
    calib_btn_->setStyleSheet("font-size: 16px; font-weight: bold; background-color: #e74c3c; color: white; border-radius: 8px;");
    
    flip_btn_ = new QPushButton(" 翻转阵营", this);
    flip_btn_->setFixedHeight(50);
    flip_btn_->setStyleSheet("font-size: 16px; font-weight: bold; background-color: #f39c12; color: white; border-radius: 8px;");

    btn_layout->addWidget(calib_btn_);
    btn_layout->addWidget(flip_btn_);
    right_layout->addLayout(btn_layout, 1);

    main_layout->addLayout(right_layout, 3);

    setWindowTitle("Radar Station - High Performance Client");
    resize(1280, 720);

    // 绑定按钮事件到自己的槽函数
    connect(calib_btn_, &QPushButton::clicked, this, &MainWindow::onCalibClicked);
    connect(flip_btn_, &QPushButton::clicked, this, &MainWindow::onFlipClicked);
}

void MainWindow::onCalibClicked() {
    worker_->triggerCalibration();
    calib_btn_->setText("标定请求中...");
    calib_btn_->setEnabled(false);
    
    // 2秒后恢复按钮状态
    QTimer::singleShot(2000, [this](){
        calib_btn_->setText(" 触发标定");
        calib_btn_->setEnabled(true);
    });
}

void MainWindow::onFlipClicked() {
    worker_->toggleMapFlip();
    // 可以在这里增加状态文字改变等 UI 效果
}

} // namespace radar_visualizer