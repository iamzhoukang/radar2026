#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory> 

//防止和其他库污染
namespace open3d {
    namespace t{
        namespace geometry{
            class RaycastingScene;
        }
    }
}

namespace radar_core {
    namespace utils{

        class Raycaster {
            public:
                //当 unique_ptr 配合前向声明使用时，析构函数绝对不能在 .hpp 里实现！
                // 必须在这里只声明，在 .cpp 里实现，否则编译器会报错 "delete incomplete type"。
                Raycaster();
                ~Raycaster();

                bool loadMesh(const std::string& mesh_path);

                // 核心算子：将 2D 像素打出一道物理射线，返回击中场地的 3D 绝对坐标
                cv::Point3f pixelToWorld(const cv::Point2f& pixel, 
                             const cv::Mat& K, const cv::Mat& D, 
                             const cv::Mat& R_inv, const cv::Mat& T) const;
            private:
                // 内部实现，使用智能指针持有 Open3D 的场景对象。
                // 因为是前向声明，编译器只知道它是一个指针（固定 8 字节大小），从而完美隔离了实现细节。
                std::unique_ptr<open3d::t::geometry::RaycastingScene> scene_;

        };


    }// namespace utils
}// namespace radar_core