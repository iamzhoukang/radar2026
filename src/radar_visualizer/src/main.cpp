#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>
#include <QMessageBox>
#include <memory>
#include <thread>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "std_srvs/srv/trigger.hpp"
#include <opencv2/opencv.hpp>

// ==========================================
// ROS 2 后台工作线程 (QObject 必须放在第一继承位)
// ==========================================
class RosWorker : public QObject, public rclcpp::Node
{
    Q_OBJECT
public:
    RosWorker() : QObject(), Node("radar_vis_node")
    {
        // 1. 订阅检测后的画框视频
        sub_video_ = this->create_subscription<sensor_msgs::msg::Image>(
            "processed_video", rclcpp::SensorDataQoS(),
            std::bind(&RosWorker::video_callback, this, std::placeholders::_1)
        );

        // 2. 订阅小地图
        sub_map_ = this->create_subscription<sensor_msgs::msg::Image>(
            "map/image", 10,
            std::bind(&RosWorker::map_callback, this, std::placeholders::_1)
        );

        // 3. 标定服务客户端
        client_calib_ = this->create_client<std_srvs::srv::Trigger>("solvepnp/start");
    }

    // 线程安全的地图翻转开关
    void toggleMapFlip() {
        is_map_flipped_ = !is_map_flipped_;
        RCLCPP_INFO(this->get_logger(), "地图翻转状态: %s", is_map_flipped_ ? "ON (180°)" : "OFF");
    }

    // 呼叫标定服务
    void callCalibration() {
        if (!client_calib_->wait_for_service(std::chrono::milliseconds(500))) {
            RCLCPP_WARN(this->get_logger(), "SolvePnP 服务未响应，请检查后台！");
            return;
        }
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        client_calib_->async_send_request(req);
        RCLCPP_INFO(this->get_logger(), "已发送标定触发请求！");
    }

signals:
    // 定义向 UI 线程发送图像的信号
    void videoReady(const QImage &img);
    void mapReady(const QImage &img);

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_video_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_map_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_calib_;
    
    std::atomic<bool> is_map_flipped_{false}; // 无锁原子变量

    // 视频回调：OpenCV 极速缩放
    void video_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); } catch (...) { return; }

        cv::Mat mat = cv_ptr->image;
        
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        // 必须 copy()，否则 cv::Mat 释放会导致 QImage 内存越界
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        emit videoReady(qimg.copy()); 
    }

    // 地图回调：180度翻转逻辑
    void map_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); } catch (...) { return; }

        cv::Mat mat = cv_ptr->image;
        
        // 核心新增：一键翻转阵营
        if (is_map_flipped_) {
            cv::rotate(mat, mat, cv::ROTATE_180);
        }

        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        emit mapReady(qimg.copy());
    }
};

// ==========================================
// Qt 主窗口 (纯 UI 显示，无密集计算)
// ==========================================
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(std::shared_ptr<RosWorker> worker) : worker_(worker)
    {
        QWidget *centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

        // 1. 左侧视频区
        videoLabel_ = new QLabel("Waiting for Video...", this);
        videoLabel_->setAlignment(Qt::AlignCenter);
        videoLabel_->setMinimumSize(800, 600);
        videoLabel_->setStyleSheet("border: 2px solid #2ecc71; background-color: #1e1e1e; color: white; font-size: 24px; font-weight: bold;");
        mainLayout->addWidget(videoLabel_, 7);

        // 2. 右侧控制区
        QVBoxLayout *rightLayout = new QVBoxLayout();
        
        mapLabel_ = new QLabel("Waiting for Map...", this);
        mapLabel_->setAlignment(Qt::AlignCenter);
        mapLabel_->setMinimumSize(350, 600);
        mapLabel_->setStyleSheet("border: 2px solid #3498db; background-color: #2c3e50; color: white; font-size: 20px;");
        rightLayout->addWidget(mapLabel_, 8);

        // 按钮组
        QHBoxLayout *btnLayout = new QHBoxLayout();
        
        // 标定按钮
        calibBtn_ = new QPushButton(" 🎯 触发标定", this);
        calibBtn_->setFixedHeight(50);
        calibBtn_->setStyleSheet("font-size: 16px; font-weight: bold; background-color: #e74c3c; color: white; border-radius: 8px;");
        
        // 翻转按钮
        flipBtn_ = new QPushButton(" 🔄 翻转阵营", this);
        flipBtn_->setFixedHeight(50);
        flipBtn_->setStyleSheet("font-size: 16px; font-weight: bold; background-color: #f39c12; color: white; border-radius: 8px;");

        btnLayout->addWidget(calibBtn_);
        btnLayout->addWidget(flipBtn_);
        rightLayout->addLayout(btnLayout, 1);

        mainLayout->addLayout(rightLayout, 3);

        setWindowTitle("Radar Station - High Performance Client");
        resize(1280, 720);

        // 绑定按钮事件 -> ROS 动作
        connect(calibBtn_, &QPushButton::clicked, this, &MainWindow::onCalibClicked);
        connect(flipBtn_, &QPushButton::clicked, this, &MainWindow::onFlipClicked);

        // 绑定后台图像信号 -> UI 贴图槽函数 (利用跨线程 QueuedConnection)
        connect(worker_.get(), &RosWorker::videoReady, this, &MainWindow::updateVideo, Qt::QueuedConnection);
        connect(worker_.get(), &RosWorker::mapReady, this, &MainWindow::updateMap, Qt::QueuedConnection);
    }

public slots:
    void updateVideo(const QImage &image) {
        // FastTransformation 避免 UI 卡顿
        videoLabel_->setPixmap(QPixmap::fromImage(image).scaled(videoLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    void updateMap(const QImage &image) {
        mapLabel_->setPixmap(QPixmap::fromImage(image).scaled(mapLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

private slots:
    void onCalibClicked() {
        worker_->callCalibration();
        calibBtn_->setText("⏳ 请求中...");
        calibBtn_->setEnabled(false);
        QTimer::singleShot(2000, [this](){
            calibBtn_->setText(" 🎯 触发标定");
            calibBtn_->setEnabled(true);
        });
    }

    void onFlipClicked() {
        worker_->toggleMapFlip();
    }

private:
    QLabel *videoLabel_;
    QLabel *mapLabel_;
    QPushButton *calibBtn_;
    QPushButton *flipBtn_;
    std::shared_ptr<RosWorker> worker_;
};

// ==========================================
// 主函数：双线程启动
// ==========================================
int main(int argc, char *argv[])
{
    // 初始化 ROS 2 和 Qt
    rclcpp::init(argc, argv);
    QApplication app(argc, argv);

    // 1. 创建 ROS 节点
    auto ros_worker = std::make_shared<RosWorker>();

    // 2. 启动后台线程跑 ROS 2 循环
    std::thread ros_thread([ros_worker]() {
        rclcpp::spin(ros_worker);
    });

    // 3. 创建并显示 Qt 窗口
    MainWindow win(ros_worker);
    win.show();

    // 4. 运行 Qt 主事件循环 (阻塞)
    int ret = app.exec();

    // 5. 安全退出清理
    rclcpp::shutdown();
    if (ros_thread.joinable()) {
        ros_thread.join();
    }

    return ret;
}

// MOC 包含声明 (因为类定义在 cpp 中)

#ifdef __INTELLISENSE__
#else
#include "main.moc"
#endif


