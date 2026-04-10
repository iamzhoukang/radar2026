# SentryTactical.msg
std_msgs/Header header

# 战术信息 (1: 是/存活, 0: 否/摧毁)
int32 outpost_alive           # 前哨站是否存活
int32 engineer_on_island      # 工程是否上中心资源岛
int32 enemy_massive_attack    # 对方是否大面积进攻
int32 ally_massive_attack     # 我方是否大面积进攻