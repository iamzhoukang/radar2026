#include "map/map_tactical_analyzer.hpp"

namespace radar_core {
namespace tactical {

void MapTacticalAnalyzer::evaluate(const std::vector<TrackedTarget>& targets, bool is_blue_team) {
    // 1. 确定敌我阵营标识符
    char my_team = is_blue_team ? 'B' : 'R';
    char enemy_team = is_blue_team ? 'R' : 'B';

    // 2. 状态重置
    engineer_on_island_ = 0;
    enemy_massive_attack_ = 0;
    ally_massive_attack_ = 0;

    int enemy_attack_count = 0;
    int ally_attack_count = 0;

    // 3. 遍历所有经过卡尔曼滤波的存活目标
    for (const auto& target : targets) {
        
        // 核心：将绝对坐标转换为“从己方老家出发的推进距离”
        // 红方老家在 0，往前走 X 增大；蓝方老家在 28，往前走 X 减小。
        float dist_from_red_base = target.x;
        float dist_from_blue_base = 28.0f - target.x;

        // 计算这台车距离它自己老家有多远
        float dist_from_enemy_base = (enemy_team == 'R') ? dist_from_red_base : dist_from_blue_base;
        float dist_from_my_base = (my_team == 'R') ? dist_from_red_base : dist_from_blue_base;

        // ==========================================
        // 战术 1：对方工程是否上资源岛
        // 目标：敌方，ID = 1 (工程 2 号)
        // ==========================================
        if (target.team == enemy_team && target.target_idx == 1) {
            // 逻辑：敌方工程距离他们老家的距离在 10~14 米之间，并且 Y 在 6~9 之间
            if (dist_from_enemy_base >= 10.0f && dist_from_enemy_base <= 14.0f &&
                target.y >= 6.0f && target.y <= 9.0f) {
                engineer_on_island_ = 1;
            }
        }

        // ==========================================
        // 战术 2：对方大面积进攻
        // 目标：敌方，步兵 (ID 2, 3) 或 英雄 (ID 0)
        // ==========================================
        if (target.team == enemy_team && 
           (target.target_idx == 0 || target.target_idx == 2 || target.target_idx == 3)) {
            // 逻辑：敌方战斗车辆距离他们老家的推进深度 > 10 米
            if (dist_from_enemy_base > 10.0f) {
                enemy_attack_count++;
            }
        }

        // ==========================================
        // 战术 3：我方大面积进攻
        // 目标：我方，步兵 (ID 2, 3) 或 英雄 (ID 0)
        // ==========================================
        if (target.team == my_team && 
           (target.target_idx == 0 || target.target_idx == 2 || target.target_idx == 3)) {
            // 逻辑：我方战斗车辆距离我方老家的推进深度 > 14 米 (跨过中场)
            if (dist_from_my_base > 14.0f) {
                ally_attack_count++;
            }
        }
    }

    // 4. 判断数量阈值 (满 2 台即触发大面积进攻)
    if (enemy_attack_count >= 2) enemy_massive_attack_ = 1;
    if (ally_attack_count >= 2) ally_massive_attack_ = 1;
}

} // namespace tactical
} // namespace radar_core