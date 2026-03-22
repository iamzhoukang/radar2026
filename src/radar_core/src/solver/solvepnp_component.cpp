#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_srvs/srv/trigger.hpp>

#include "solver/pnp_solver.hpp" 
#include "solver/calibration_ui.hpp"

namespace radar_core {

class SolvePnPComponent : public rclcpp::Node
{
public:
    explicit SolvePnPComponent(const rclcpp::NodeOptions & options) : Node("solvepnp_component", options)
    {
        // 1. 获取参数
        this->declare_parameter<std::string>("config_path", "/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml");
        this->declare_parameter<std::string>("keypoint_path", "/home/lzhros/Code/RadarStation/config/solver/keypoint_6.txt");
        config_file_path_ = this->get_parameter("config_path").as_string();
        
        // 2. 唤醒模型层 (Model)
        if (!solver_.loadConfig(config_file_path_, this->get_parameter("keypoint_path").as_string())) {
            RCLCPP_ERROR(this->get_logger(), "算法库初始化失败！请检查 yaml 和 txt 路径，以及基准点是否 >= 4。");
            return; 
        }

        // 3. 注册 ROS 通信总线
        client_map_reload_ = this->create_client<std_srvs::srv::Trigger>("map/reload_config");
        srv_start_calib_ = this->create_service<std_srvs::srv::Trigger>(
            "solvepnp/start",
            std::bind(&SolvePnPComponent::handle_start, this, std::placeholders::_1, std::placeholders::_2)
        );

        // 4. 启动视图层的引擎心跳 (30 FPS)
        ui_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&SolvePnPComponent::ui_loop, this)
        );

        RCLCPP_INFO(this->get_logger(), "MVC标定控制器就绪... 等待 'solvepnp/start' 触发");
    }

private:
    std::string config_file_path_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_map_reload_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_start_calib_;
    rclcpp::TimerBase::SharedPtr ui_timer_;


    solver::PnPSolver solver_;          // 模型层：专注算力
    ui::CalibrationUI ui_;              // 视图层：专注交互


    void handle_start(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        // 状态锁：如果 UI 引擎已经在跑了，拒绝重复唤醒
        if (ui_.isActive()) {
            response->success = false;
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "正在请求 5K 原始雷达视野...");
        
        // 临时搭桥：向底层的海康节点索要一帧图像
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "cs200_topic", 1, 
            std::bind(&SolvePnPComponent::image_callback, this, std::placeholders::_1)
        );

        response->success = true;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try { 
            // 将 ROS 消息转化为 OpenCV 矩阵
            cv::Mat raw_img = cv_bridge::toCvCopy(msg, "bgr8")->image; 
            
            // 将图像喂给 UI 引擎并启动
            ui_.start(raw_img); 
            RCLCPP_INFO(this->get_logger(), "UI 引擎已接管图像！等待用户操作...");
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "图像转换失败！");
        }
        
        // 【关键】：过河拆桥！拿到一帧图像后立刻摧毁订阅器，释放零拷贝总线
        sub_.reset(); 
    }

    void ui_loop()
    {
        // 询问 UI 层：用户刚才敲键盘了吗？
        ui::UIAction action = ui_.spinOnce();

        // 根据用户的指令进行战略调度
        if (action == ui::UIAction::TRIGGER_CLEAR) {
            ui_.clearPoints();
            RCLCPP_INFO(this->get_logger(), "坐标系已清空，请重新点选。");
        } 
        else if (action == ui::UIAction::TRIGGER_SOLVE) {
            perform_solve_pnp();
        }
    }

    void perform_solve_pnp()
    {
        // 1. 从视图层拿数据
        const auto& points = ui_.getClickedPoints();
        if (points.size() != 6) {
            RCLCPP_WARN(this->get_logger(), "坐标系重建需要严密的 6 点标定！当前点数: %ld", points.size());
            return;
        }

        // 2. 扔给模型层算数学题
        solver::PnPResult result = solver_.solve(points);

        // 3. 批判性评估解算质量
        if (!result.success || result.reprojection_error > 30.0) {
            RCLCPP_ERROR(this->get_logger(), "空间重构失败或物理形变过大 (误差: %.2f px)！请重试。", result.reprojection_error); 
            return; 
        }

        RCLCPP_INFO(this->get_logger(), "空间重构极度精准！重投影误差: %.2f px", result.reprojection_error);
        
        // 4. 数据持久化与反馈
        if (solver_.saveExtrinsics(config_file_path_, result.rvec, result.tvec)) {
            
            // 指挥 UI 引擎在屏幕上打下绿色的成功烙印
            auto projected_pts = solver_.projectPoints(result.rvec, result.tvec);
            ui_.showFeedback(projected_pts, result.reprojection_error);
            
            // 通知隔壁的地图组件热更新参数
            trigger_map_reload();
            
            // 卸载 UI 引擎，释放 5K 图像的庞大显存
            ui_.stop(); 
            RCLCPP_INFO(this->get_logger(), "标定系统已休眠。");
        } else {
            RCLCPP_ERROR(this->get_logger(), "灾难级错误：无法将物理参数写入 YAML！");
        }
    }

    void trigger_map_reload() {
        if (client_map_reload_->wait_for_service(std::chrono::milliseconds(200))) {
            client_map_reload_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
            RCLCPP_INFO(this->get_logger(), "已向 Map 节点发送物理引擎热重载指令。");
        } else {
            RCLCPP_WARN(this->get_logger(), "Map 节点失联，新标定将在下次重启时生效。");
        }
    }
};

} // namespace radar_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::SolvePnPComponent)