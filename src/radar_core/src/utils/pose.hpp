#ifndef __POSE_HPP__
#define __POSE_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// 关键点结构
struct PoseKeyPoint {
    cv::Point2f pt;    // 在原图上的像素坐标 (x, y)
    float visibility;  // 可见度/置信度 (0.0~1.0)
};

// 姿态检测结果
struct PoseResult {
    int class_id;                        // 类别ID
    float confidence;                    // 目标整体置信度
    cv::Rect box;                        // 边界框
    std::vector<PoseKeyPoint> keypoints; // 关键点集合
};

class PoseModel {
private:
    // 模型基础参数
    int input_w;             
    int input_h;             
    int num_classes;         // 动态类别数
    int num_kpts;            // 动态关键点数
    int num_boxes;           // 输出框的数量 (如 8400)
    float scoreThreshold;    
    float nmsThreshold;
    bool use_cuda_preproc_;  

    // TensorRT 核心组件
    nvinfer1::IRuntime  *runtime = nullptr;             
    nvinfer1::ICudaEngine *engine = nullptr;            
    nvinfer1::IExecutionContext *context = nullptr;     
    cudaStream_t stream = nullptr;

    // 显存指针 (Device)
    void* buffer_idx_0 = nullptr; 
    void* buffer_idx_1 = nullptr; 

    // CUDA 预处理图床
    uint8_t* d_src_img = nullptr; 
    int max_src_size = 0;

    // 绑定名称
    std::string input_name;                              
    std::string output_name;                             

    // 锁页内存 (Host Pinned Memory - DMA 极速通道)
    float* pinned_in_host = nullptr;   
    float* pinned_out_host = nullptr;  
    int output_size_ = 0;              

    // 仿射变换参数 (用于坐标还原)
    float scale;                                       
    int pad_w;                                           
    int pad_h;

    // 内部方法
    void preprocess_cuda(cv::Mat &frame);  
    void preprocess_cpu(cv::Mat &frame);   
    void postprocessing();                 

public:
    std::vector<PoseResult> detectResults; 

    // 构造函数参数注入 (默认 1类, 8点, 开启CUDA加速)
    PoseModel(const std::string modelPath, const int &inputSize,
              const float &scoreThreshold, const float &nmsThreshold, 
              int num_classes = 1, int num_kpts = 8, 
              bool use_cuda = true);
    
    ~PoseModel();

    bool Detect(cv::Mat &frame);
};

#endif