#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>
#include <cstring>
#include <mutex> 

// 引用跨包的消息接口 (这个命名空间一般是全局固定的，无需更改)
#include "radar_interfaces/msg/radar_map.hpp"
#include "radar_interfaces/msg/sentry_tactical.hpp" 

// 引用当前包(radar_serial)的头文件
#include "serial_driver.hpp"
#include "rm_protocol.hpp"

using namespace std::chrono_literals;

namespace radar_serial {

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
        
        // 🌟 确定红蓝方的各个终端 ID
        my_robot_id_      = is_red_side_ ? 9 : 109;       // 雷达 ID
        target_sentry_id_ = is_red_side_ ? 7 : 107;       // 哨兵 ID
        target_dart_id_   = is_red_side_ ? 8 : 108;       // 飞镖 ID

        RCLCPP_INFO(this->get_logger(), "[串口节点] 频率: 坐标 5Hz / 战术 2Hz | 雷达ID: %d | 哨兵ID: %d | 飞镖ID: %d", 
                    my_robot_id_, target_sentry_id_, target_dart_id_);

        driver_ = std::make_shared<SerialDriver>(port);
        driver_->setCallback([this](uint16_t cmd_id, uint8_t* data, uint16_t len) {
            this->handlePacket(cmd_id, data, len);
        });

        if (!driver_->openPort()) {
            RCLCPP_ERROR(this->get_logger(), "串口打开失败！");
        } else {
            RCLCPP_INFO(this->get_logger(), "串口连接成功");
        }

        // 1. 订阅者：实时刷新小地图坐标 (Buffer)
        sub_radar_map_ = this->create_subscription<radar_interfaces::msg::RadarMap>(
            "map/official_data", 10,
            [this](const radar_interfaces::msg::RadarMap::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_map_msg_ = *msg;
                has_new_data_ = true;
            });

        // 2. 订阅者：实时刷新战术情报 (Buffer)
        sub_tactical_ = this->create_subscription<radar_interfaces::msg::SentryTactical>(
            "map/tactical_info", 10,
            [this](const radar_interfaces::msg::SentryTactical::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_tactical_msg_ = *msg;
                has_tactical_data_ = true;
            });

        // 3. 定时器：以 5Hz (200ms) 频率发送坐标包 (0x0305)
        report_timer_ = this->create_wall_timer(200ms, std::bind(&SerialNode::onTimerReport, this));

        // 4. 定时器：以 2Hz (500ms) 频率发送战术包 (0x0301)
        tactical_timer_ = this->create_wall_timer(500ms, std::bind(&SerialNode::onTimerTactical, this));
    }

private:
    std::shared_ptr<SerialDriver> driver_;
    
    rclcpp::Subscription<radar_interfaces::msg::RadarMap>::SharedPtr sub_radar_map_;
    rclcpp::Subscription<radar_interfaces::msg::SentryTactical>::SharedPtr sub_tactical_;
    
    rclcpp::TimerBase::SharedPtr report_timer_;
    rclcpp::TimerBase::SharedPtr tactical_timer_; 
    
    bool is_red_side_;
    uint16_t my_robot_id_;
    uint16_t target_sentry_id_;
    uint16_t target_dart_id_; 
    
    // 线程安全相关
    std::mutex data_mutex_;
    radar_interfaces::msg::RadarMap latest_map_msg_;
    radar_interfaces::msg::SentryTactical latest_tactical_msg_; 
    bool has_new_data_ = false;
    bool has_tactical_data_ = false; 

    // 触发逻辑相关变量
    uint8_t trigger_seq_ = 0;
    std::chrono::steady_clock::time_point last_trigger_time_ = std::chrono::steady_clock::now();

    // =========================================================
    // 任务 A：严格 5Hz 发送 0x0305 小地图坐标
    // =========================================================
    void onTimerReport()
    {
        if (!has_new_data_ || !driver_->isOpen()) return;

        map_robot_data_t data;
        std::memset(&data, 0, sizeof(data));

        {
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
            data.ally_aerial_position_x     = 0; 
            data.ally_aerial_position_y     = 0;
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
            data.opponent_aerial_position_x     = static_cast<uint16_t>(oppo_x[4] * 100.0f);
            data.opponent_aerial_position_y     = static_cast<uint16_t>(oppo_y[4] * 100.0f);
            data.opponent_sentry_position_x     = static_cast<uint16_t>(oppo_x[5] * 100.0f);
            data.opponent_sentry_position_y     = static_cast<uint16_t>(oppo_y[5] * 100.0f);
        }

        driver_->sendPacket(CMD_ID_RADAR_MAP, reinterpret_cast<uint8_t*>(&data), sizeof(data));
    }

    // =========================================================
    // 🌟 任务 B：稳定 2Hz 发送 0x0301 战术情报给【哨兵】与【飞镖】
    // =========================================================
    void onTimerTactical()
    {
        if (!has_tactical_data_ || !driver_->isOpen()) return;

        radar_to_sentry_packet_t packet;
        std::memset(&packet, 0, sizeof(packet));

        // 1. 封装帧头基础信息 (发送者雷达，子命令0x0200)
        packet.header.data_cmd_id = SUBCMD_ID_SENTRY_TACTICAL; // 0x0200
        packet.header.sender_id = my_robot_id_;                // 9 或 109

        // 2. 封装核心数据段 (按顺序装填 4 个 int32)
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            packet.body.outpost_alive        = latest_tactical_msg_.outpost_alive;
            packet.body.engineer_on_island   = latest_tactical_msg_.engineer_on_island;
            packet.body.enemy_massive_attack = latest_tactical_msg_.enemy_massive_attack;
            packet.body.ally_massive_attack  = latest_tactical_msg_.ally_massive_attack;
        }

        // 3. 第一发：发送给【哨兵】
        packet.header.receiver_id = target_sentry_id_;         // 7 或 107
        driver_->sendPacket(CMD_ID_INTERACTION, reinterpret_cast<uint8_t*>(&packet), sizeof(packet));

        // 4. 第二发：修改接收者ID，发送给【飞镖】
        packet.header.receiver_id = target_dart_id_;           // 8 或 108
        driver_->sendPacket(CMD_ID_INTERACTION, reinterpret_cast<uint8_t*>(&packet), sizeof(packet));

        // //打印终端日志方便调试
        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
        //     "[战术下发] -> 哨兵&飞镖 | 前哨:%d 工上岛:%d 敌大攻:%d 我大攻:%d",
        //     packet.body.outpost_alive, packet.body.engineer_on_island, 
        //     packet.body.enemy_massive_attack, packet.body.ally_massive_attack);
    }

    // =========================================================
    // 任务 C：处理下行数据 0x020E 自动触发易伤
    // =========================================================
    void handlePacket(uint16_t cmd_id, uint8_t* data, uint16_t len)
    {
        if (cmd_id == CMD_ID_RADAR_INFO) {
            if (len < sizeof(radar_info_t)) return;
            radar_info_t info;
            std::memcpy(&info, data, sizeof(info));
            uint8_t chance = info.get_double_damage_chance();

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

} // namespace radar_serial

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<radar_serial::SerialNode>());
    rclcpp::shutdown();
    return 0;
}