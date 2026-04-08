#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>
#include <cstring>
#include <mutex> // 引入互斥锁保护共享数据

// 引用雷达坐标接口 RadarMap.msg
#include "radar_interfaces/msg/radar_map.hpp"

// 引用驱动与底层协议
#include "serial_driver.hpp"
#include "rm_protocol.hpp"

using namespace std::chrono_literals;

namespace radar_core {

class SerialNode : public rclcpp::Node
{
public:
    SerialNode() : Node("serial_node")
    {
        this->declare_parameter("port_name", "/dev/ttyUSB0");
        this->declare_parameter("is_blue_team", true);

        std::string port = this->get_parameter("port_name").as_string();
        bool is_blue_team = this->get_parameter("is_blue_team").as_bool();
        is_red_side_ = !is_blue_team; 
        my_robot_id_ = is_red_side_ ? 9 : 109;

        RCLCPP_INFO(this->get_logger(), "[串口节点] | 频率: 5Hz | ID: %d", my_robot_id_);

        driver_ = std::make_shared<SerialDriver>(port);
        driver_->setCallback([this](uint16_t cmd_id, uint8_t* data, uint16_t len) {
            this->handlePacket(cmd_id, data, len);
        });

        if (!driver_->openPort()) {
            RCLCPP_ERROR(this->get_logger(), "串口打开失败！");
        } else {
            RCLCPP_INFO(this->get_logger(), "串口连接成功");
        }

        // 1. 订阅者：仅负责实时刷新内存中的最新坐标数据 (Buffer)
        sub_radar_map_ = this->create_subscription<radar_interfaces::msg::RadarMap>(
            "map/official_data", 10,
            [this](const radar_interfaces::msg::RadarMap::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_map_msg_ = *msg;
                has_new_data_ = true;
            });

        // 2. 定时器：严格以 5Hz (200ms) 频率执行发送与全量打印任务
        report_timer_ = this->create_wall_timer(200ms, std::bind(&SerialNode::onTimerReport, this));
    }

private:
    std::shared_ptr<SerialDriver> driver_;
    rclcpp::Subscription<radar_interfaces::msg::RadarMap>::SharedPtr sub_radar_map_;
    rclcpp::TimerBase::SharedPtr report_timer_;
    
    bool is_red_side_;
    uint16_t my_robot_id_;
    
    // 线程安全相关
    std::mutex data_mutex_;
    radar_interfaces::msg::RadarMap latest_map_msg_;
    bool has_new_data_ = false;

    // 触发逻辑相关变量
    uint8_t trigger_seq_ = 0;
    std::chrono::steady_clock::time_point last_trigger_time_ = std::chrono::steady_clock::now();

    // =========================================================
    // 任务：严格 5Hz 发包与【全量调试信息打印】
    // =========================================================
    void onTimerReport()
    {
        // 若没有收到过任何有效数据或串口未开，则跳过本次循环
        if (!has_new_data_ || !driver_->isOpen()) return;

        map_robot_data_t data;
        std::memset(&data, 0, sizeof(data));

        {
            // 锁定数据读取过程
            std::lock_guard<std::mutex> lock(data_mutex_);
            const auto& ally_x = is_red_side_ ? latest_map_msg_.red_x : latest_map_msg_.blue_x;
            const auto& ally_y = is_red_side_ ? latest_map_msg_.red_y : latest_map_msg_.blue_y;
            const auto& oppo_x = is_red_side_ ? latest_map_msg_.blue_x : latest_map_msg_.red_x;
            const auto& oppo_y = is_red_side_ ? latest_map_msg_.blue_y : latest_map_msg_.red_y;

            // --- 己方装填 (米 -> 厘米) ---
            data.ally_hero_position_x       = static_cast<uint16_t>(ally_x[0] * 100.0f);
            data.ally_hero_position_y       = static_cast<uint16_t>(ally_y[0] * 100.0f);
            data.ally_engineer_position_x   = static_cast<uint16_t>(ally_x[1] * 100.0f);
            data.ally_engineer_position_y   = static_cast<uint16_t>(ally_y[1] * 100.0f);
            data.ally_infantry_3_position_x = static_cast<uint16_t>(ally_x[2] * 100.0f);
            data.ally_infantry_3_position_y = static_cast<uint16_t>(ally_y[2] * 100.0f);
            data.ally_infantry_4_position_x = static_cast<uint16_t>(ally_x[3] * 100.0f);
            data.ally_infantry_4_position_y = static_cast<uint16_t>(ally_y[3] * 100.0f);
            
            // 己方无人机默认发送 (0,0)
            data.ally_aerial_position_x     = 0; 
            data.ally_aerial_position_y     = 0;
            
            // 哨兵对应的索引是 5
            data.ally_sentry_position_x     = static_cast<uint16_t>(ally_x[5] * 100.0f);
            data.ally_sentry_position_y     = static_cast<uint16_t>(ally_y[5] * 100.0f);

            // --- 对方装填 (米 -> 厘米) ---
            data.opponent_hero_position_x       = static_cast<uint16_t>(oppo_x[0] * 100.0f);
            data.opponent_hero_position_y       = static_cast<uint16_t>(oppo_y[0] * 100.0f);
            data.opponent_engineer_position_x   = static_cast<uint16_t>(oppo_x[1] * 100.0f);
            data.opponent_engineer_position_y   = static_cast<uint16_t>(oppo_y[1] * 100.0f);
            data.opponent_infantry_3_position_x = static_cast<uint16_t>(oppo_x[2] * 100.0f);
            data.opponent_infantry_3_position_y = static_cast<uint16_t>(oppo_y[2] * 100.0f);
            data.opponent_infantry_4_position_x = static_cast<uint16_t>(oppo_x[3] * 100.0f);
            data.opponent_infantry_4_position_y = static_cast<uint16_t>(oppo_y[3] * 100.0f);
            
            // 获取解算出的对方无人机坐标 (索引为 4)
            data.opponent_aerial_position_x     = static_cast<uint16_t>(oppo_x[4] * 100.0f);
            data.opponent_aerial_position_y     = static_cast<uint16_t>(oppo_y[4] * 100.0f);

            // 哨兵对应的索引是 5
            data.opponent_sentry_position_x     = static_cast<uint16_t>(oppo_x[5] * 100.0f);
            data.opponent_sentry_position_y     = static_cast<uint16_t>(oppo_y[5] * 100.0f);
        }

        // 发送串口包 0x0305
        driver_->sendPacket(CMD_ID_RADAR_MAP, reinterpret_cast<uint8_t*>(&data), sizeof(data));

        // 核心要求：保留并全量输出调试信息，加入对方空军的打印
        RCLCPP_INFO(this->get_logger(),
            "\n=== 发送 0x0305 坐标包 (5Hz) ===\n"
            "[对方] 英:(%d,%d) 工:(%d,%d) 步3:(%d,%d) 步4:(%d,%d) 哨:(%d,%d) 空:(%d,%d)\n"
            "[己方] 英:(%d,%d) 工:(%d,%d) 步3:(%d,%d) 步4:(%d,%d) 哨:(%d,%d) 空:(0,0)",
            data.opponent_hero_position_x, data.opponent_hero_position_y,
            data.opponent_engineer_position_x, data.opponent_engineer_position_y,
            data.opponent_infantry_3_position_x, data.opponent_infantry_3_position_y,
            data.opponent_infantry_4_position_x, data.opponent_infantry_4_position_y,
            data.opponent_sentry_position_x, data.opponent_sentry_position_y,
            data.opponent_aerial_position_x, data.opponent_aerial_position_y, // 新增对方空军展示
            
            data.ally_hero_position_x, data.ally_hero_position_y,
            data.ally_engineer_position_x, data.ally_engineer_position_y,
            data.ally_infantry_3_position_x, data.ally_infantry_3_position_y,
            data.ally_infantry_4_position_x, data.ally_infantry_4_position_y,
            data.ally_sentry_position_x, data.ally_sentry_position_y
        );
    }

    // 处理接收到的 0x020E (双倍易伤机会监控)
    void handlePacket(uint16_t cmd_id, uint8_t* data, uint16_t len)
    {
        if (cmd_id == CMD_ID_RADAR_INFO) {
            if (len < sizeof(radar_info_t)) return;
            radar_info_t info;
            std::memcpy(&info, data, sizeof(info));
            uint8_t chance = info.get_double_damage_chance();

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "[下行监控] 0x020E 收到，当前双倍易伤机会: %d", chance);

            if (chance > 0) {
                auto now = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_trigger_time_).count();
                if (duration >= 2) { 
                    RCLCPP_WARN(this->get_logger(), "⚡️ 执行自动触发！向 0x8080 发送 0x0121 ⚡️");
                    sendTriggerCmd();
                    last_trigger_time_ = now;
                }
            }
        }
    }

    void sendTriggerCmd()
    {
        radar_double_damage_packet_t packet;
        std::memset(&packet, 0, sizeof(packet)); 
        packet.header.data_cmd_id = SUBCMD_ID_RADAR_DECISION;
        packet.header.sender_id = my_robot_id_;
        packet.header.receiver_id = SERVER_ID;
        packet.body.confirm_trigger = ++trigger_seq_; 
        driver_->sendPacket(CMD_ID_INTERACTION, reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
    }
};

} // namespace radar_core

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<radar_core::SerialNode>());
    rclcpp::shutdown();
    return 0;
}