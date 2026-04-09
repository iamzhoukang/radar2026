#pragma once

#include <vector>
#include <iostream>
#include <limits>

namespace radar_core {
namespace tracker {

class HungarianAlgorithm {
public:
    HungarianAlgorithm() {}
    ~HungarianAlgorithm() {}

    // 传入代价矩阵 (Cost Matrix)，返回每个检测框匹配到的轨迹索引 (Assignment)
    // 如果没有匹配上，则对应的 Assignment 为 -1
    float Solve(std::vector<std::vector<float>>& DistMatrix, std::vector<int>& Assignment);
};

} // namespace tracker
} // namespace radar_core