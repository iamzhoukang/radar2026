#pragma once
#include <vector>

namespace radar_core {
namespace tactical {

// 轻量级目标结构体，隔离 tracker 的复杂底层实现
struct TrackedTarget {
    char team;       // 'B' (蓝方) 或 'R' (红方)
    int target_idx;  // 0:英雄, 1:工程, 2:步兵3, 3:步兵4, 4:无人机, 5:哨兵
    float x;         // 场地绝对 X 坐标 (0 为我方基地)
    float y;         // 场地绝对 Y 坐标
};

class MapTacticalAnalyzer {
public:
    MapTacticalAnalyzer() = default;
    ~MapTacticalAnalyzer() = default;

    // 核心评估函数：每帧传入当前场地上的所有有效目标
    void evaluate(const std::vector<TrackedTarget>& targets, bool is_blue_team);

    // 获取战术状态标志位
    int get_engineer_on_island() const { return engineer_on_island_; }
    int get_enemy_massive_attack() const { return enemy_massive_attack_; }
    int get_ally_massive_attack() const { return ally_massive_attack_; }

private:
    int engineer_on_island_ = 0;
    int enemy_massive_attack_ = 0;
    int ally_massive_attack_ = 0;
};

} // namespace tactical
} // namespace radar_core