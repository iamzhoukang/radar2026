# SentryTactical.msg
std_msgs/Header header

# 战术信息 (1: 是/存活, 0: 否/摧毁)
int8 outpost_alive           # 前哨站是否存活
int8 engineer_on_island      # 工程是否上中心资源岛
int8 enemy_massive_attack    # 对方是否大面积进攻
int8 ally_massive_attack     # 我方是否大面积进攻