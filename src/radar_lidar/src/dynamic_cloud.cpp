#include "dynamic_cloud.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h> // 【新增】用于执行空间变换
#include <omp.h> // 多线程加速

namespace radar_lidar {

DynamicCloud::DynamicCloud(const rclcpp::NodeOptions & options)
: Node("dynamic_cloud_node", options) {
    
    // 1. 获取参数
    std::string pcd_path = this->declare_parameter("map_pcd_path", "/home/lzhros/Code/RadarStation/config/lidar/RB2026_rmuc.pcd");
    distance_threshold_ = this->declare_parameter("distance_threshold", 0.2); 

    // 【新增】初始化 TF 监听器
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 2. 加载静态背景地图
    static_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    kdtree_map_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    
    if (!loadStaticMap(pcd_path)) {
        RCLCPP_ERROR(this->get_logger(), "静态背景地图加载失败！");
    }

    sub_raw_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10, std::bind(&DynamicCloud::cloudCallback, this, std::placeholders::_1));
        
    pub_dynamic_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10);

    RCLCPP_INFO(this->get_logger(), "\033[1;32m高性能动态提取就绪 (坐标对齐+Voxel+CropBox+OpenMP)\033[0m");
}

bool DynamicCloud::loadStaticMap(const std::string& pcd_path) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *static_map_) == -1) return false;
    
    // 对背景地图也做降采样，提高搜索效率
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f); 
    voxel_filter.setInputCloud(static_map_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_map(new pcl::PointCloud<pcl::PointXYZ>());
    voxel_filter.filter(*downsampled_map);
    
    static_map_ = downsampled_map;
    if (!static_map_->empty()) {
        kdtree_map_->setInputCloud(static_map_);
    }
    return true;
}

void DynamicCloud::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (static_map_->empty()) return; 

    // ==========================================
    // 核心优化：TF 坐标对齐 (无论实验室模式还是实战模式)
    // ==========================================
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        // 查找从雷达系 (livox_frame) 到世界系 (map) 的变换
        transform_stamped = tf_buffer_->lookupTransform("map", msg->header.frame_id, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "等待定位 TF 广播: %s", ex.what());
        return; 
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *raw_cloud);

    // 构造变换矩阵
    Eigen::Quaternionf q(
        transform_stamped.transform.rotation.w,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z
    );
    Eigen::Vector3f trans(
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z
    );
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    transform_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    transform_matrix.block<3, 1>(0, 3) = trans;

    // 将原始点云变换到 map 坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*raw_cloud, *aligned_cloud, transform_matrix);

    // ==========================================
    // 🔪 性能优化 1：ROI 裁剪 (基于变换后的绝对坐标)
    // ==========================================
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> box_filter;
    // X:-15~15, Y:-10~10, Z:0~4 (标准赛场范围)
    box_filter.setMin(Eigen::Vector4f(-15.0, -10.0, 0.0, 1.0));
    box_filter.setMax(Eigen::Vector4f(15.0, 10.0, 4.0, 1.0));
    box_filter.setInputCloud(aligned_cloud);
    box_filter.filter(*cropped_cloud);

    // ==========================================
    // 📉 性能优化 2：体素降采样
    // ==========================================
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f); 
    voxel_filter.setInputCloud(cropped_cloud);
    voxel_filter.filter(*downsampled_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    float sqr_threshold = distance_threshold_ * distance_threshold_;
    std::vector<int> is_dynamic(downsampled_cloud->points.size(), 0);

    // ==========================================
    // ⚡ 性能优化 3：OpenMP 多核并发背景扣除
    // ==========================================
    #pragma omp parallel for
    for (size_t i = 0; i < downsampled_cloud->points.size(); ++i) {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        
        // 与静态背景地图做差分
        if (kdtree_map_->nearestKSearch(downsampled_cloud->points[i], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            if (pointNKNSquaredDistance[0] > sqr_threshold) {
                is_dynamic[i] = 1; 
            }
        }
    }

    for (size_t i = 0; i < is_dynamic.size(); ++i) {
        if (is_dynamic[i] == 1) {
            dynamic_cloud->points.push_back(downsampled_cloud->points[i]);
        }
    }

    dynamic_cloud->width = dynamic_cloud->points.size();
    dynamic_cloud->height = 1;
    dynamic_cloud->is_dense = true;

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*dynamic_cloud, output_msg);
    output_msg.header.stamp = msg->header.stamp;
    output_msg.header.frame_id = "map"; // 
    
    pub_dynamic_->publish(output_msg);
}

} // namespace radar_lidar

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(radar_lidar::DynamicCloud)