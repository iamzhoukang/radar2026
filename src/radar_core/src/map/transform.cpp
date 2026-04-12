#include "map/transform.hpp"
#include <algorithm> 

namespace radar_core{
    namespace utils{
        cv::Point2f convertToOfficialMap(const cv::Point3f& mesh_pt, float field_length, float field_width, bool is_blue_team)
        {
            float raw_x;
            float raw_y;

            if(is_blue_team){
                //蓝方视角
                raw_x = 28.0 -( -mesh_pt.z + field_length / 2.0f);
                raw_y = 15 - (-mesh_pt.x + field_width / 2.0f);
            }else{
                //红方视角
                raw_x = -mesh_pt.z + field_length / 2.0f;
                raw_y = -mesh_pt.x + field_width / 2.0f;
            }

            //绝对安全越界截断
            float official_x = std::max(0.0f, std::min(raw_x, field_length));
            float official_y = std::max(0.0f, std::min(raw_y, field_width));

            return cv::Point2f(official_x, official_y);
        }

        bool parseTargetLabel(const std::string& label, char& team, int& target_idx)
        {
            // 【新增】无人机专属身份解析
            if (label == "Drone") {
                team = 'A'; // 'A' 代表 All, 双方阵营均可渲染
                target_idx = 4; // 专门留出的空位 (原本 1234对应0123, 7对应5)
                return true;
            }

            if(label.length() < 2) return false;
            team = label[0]; //R or B
            char id_char = label[1];

            if (id_char == '1') target_idx = 0;
            else if (id_char == '2') target_idx = 1;
            else if (id_char == '3') target_idx = 2;
            else if (id_char == '4') target_idx = 3;
            else if (id_char == '7') target_idx = 5;
            else return false;

            return true;
        }
    }
}