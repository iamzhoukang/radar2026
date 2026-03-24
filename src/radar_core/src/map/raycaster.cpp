#include "map/raycaster.hpp"

// 在 .cpp 中引入 Open3D 
#include <open3d/Open3D.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/core/Tensor.h>

namespace radar_core{
    namespace utils{


        // 析构函数必须在 .cpp 中实现（即便它是空的），让 unique_ptr 能看到完整的 Open3D 类定义来销毁它
        Raycaster::~Raycaster() = default;
        Raycaster::Raycaster() = default;

        bool Raycaster::loadMesh(const std::string& mesh_path){
            open3d::geometry::TriangleMesh legacy_mesh;

            //读取传统的 3D 网格文件
            if (!open3d::io::ReadTriangleMesh(mesh_path, legacy_mesh)) {
                scene_.reset(); // 如果读取失败，清空智能指针
                return false;
            }

            //将传统网格转换为 Tensor 网格 (Open3D 现代接口，支持 GPU/CPU 硬件加速光线追踪)
            open3d::t::geometry::TriangleMesh tensor_mesh = open3d::t::geometry::TriangleMesh::FromLegacy(legacy_mesh);

            // 构建光线追踪场景
            scene_ = std::make_unique<open3d::t::geometry::RaycastingScene>(); 
            scene_->AddTriangles(tensor_mesh);
            return true;
        }

        cv::Point3f Raycaster::pixelToWorld(const cv::Point2f& pixel, 
                             const cv::Mat& K, const cv::Mat& D, 
                             const cv::Mat& R_inv, const cv::Mat& T) const
        {
            //先消除镜头畸变
            std::vector<cv::Point2f> src_pts = {pixel},dst_pts;//pixel是x,y dst_pts是绝对物理偏转角度
            //将弯曲的像素点拉直，转换到归一化相机坐标系
            cv::undistortPoints(src_pts,dst_pts,K,D);

            //构造相机坐标系下的射线终点 P_c = [u_undistorted, v_undistorted, 1]^T
            cv::Mat P_c = (cv::Mat_<double>(3, 1) << dst_pts[0].x, dst_pts[0].y, 1.0);

            //仿射变换 (相机 -> 世界)
            // 射线在世界坐标系中的方向向量
            cv::Mat Ray_world = R_inv * P_c;
            // 相机光心在世界坐标系中的绝对起点位置
            cv::Mat Cam_world = -R_inv * T;

            float ox = Cam_world.at<double>(0), oy = Cam_world.at<double>(1), oz = Cam_world.at<double>(2);
            float dx = Ray_world.at<double>(0), dy = Ray_world.at<double>(1), dz = Ray_world.at<double>(2);

            
            // 保底机制：如果 3D 模型没加载成功，退化为 2D 纯平面的线性代数求解
    
            auto fallback_to_flat_ground = [&]() -> cv::Point3f {
            if (std::abs(dy) < 1e-6) return cv::Point3f(0, 0, 0); // 防止除以零
            double t_fb = -oy / dy; // 令 y = 0，求出射线击中地面的时间 t
            return cv::Point3f(ox + t_fb * dx, 0.0f, oz + t_fb * dz);
            };

            if (!scene_) return fallback_to_flat_ground();// 如果 3D 模型没加载成功，退化为 2D 纯平面的线性代数求解

            // 物理引擎：Open3D 硬件加速光线追踪
            // 构造一根 6D 射线 [起点X, 起点Y, 起点Z, 方向X, 方向Y, 方向Z]
            std::vector<float> ray_data = {ox, oy, oz, dx, dy, dz};
            open3d::core::Tensor ray(ray_data,{1,6},open3d::core::Float32);

            // 调用 Open3D 光线追踪接口，得到射线与网格的交点
            auto result = scene_->CastRays(ray);
            float t_hit = result["t_hit"].Item<float>(); // 击中目标的距离比例系数 t

            // 如果 t_hit 是无限大(无穷远)，说明射到了天上，没打中场地
            if (std::isinf(t_hit)) return fallback_to_flat_ground();

            // 最终物理坐标 = 起点 + t * 方向
            return cv::Point3f(ox + t_hit * dx, oy + t_hit * dy, oz + t_hit * dz);
          

        }
    }// namespace utils
}// namespace radar_core