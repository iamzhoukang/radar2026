#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace radar_core{
    namespace utils{
        // 将 3D 物理碰撞坐标转换为裁判系统小地图绝对坐标
        cv::Point2f convertToOfficialMap(const cv::Point3f& mesh_pt, float field_length, float field_width, bool is_blue_team);

        // 解析车辆标签 (如 "R1", "B7")，严格遵循 0~4 的数组索引映射
    bool parseTargetLabel(const std::string& label, char& team, int& target_idx);

    }
}