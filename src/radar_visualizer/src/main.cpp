#include <QApplication>
#include <thread>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include "ros_worker.hpp"
#include "main_window.hpp"

int main(int argc, char *argv[])
{
    // 1. 初始化 ROS 2 和 Qt 环境
    rclcpp::init(argc, argv);
    QApplication app(argc, argv);

    // 2. 实例化打工人 (ROS 节点)
    auto ros_worker = std::make_shared<radar_visualizer::RosWorker>();

    // 3. 把打工人丢进地牢 (后台子线程) 一直干活
    std::thread ros_thread([ros_worker]() {
        rclcpp::spin(ros_worker);
    });

    // 4. 实例化并显示高大上的主界面
    radar_visualizer::MainWindow win(ros_worker);
    win.show();

    // 5. 启动 Qt 主事件循环 (UI 线程开始阻塞接客)
    int ret = app.exec();

    // 6. 安全善后：关闭应用时杀死 ROS 节点并回收线程
    rclcpp::shutdown();
    if (ros_thread.joinable()) {
        ros_thread.join();
    }

    return ret;
}