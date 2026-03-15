#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h> // CUDA 运行时 API

struct Result {
    int idx;           // 类别id索引
    float confidence;  // 置信度(0.0~1.0)
    cv::Rect box;      // 边界框(x,y,w,h)
};

class Model {
private:
    // 模型参数
    int input_w;             
    int input_h;             
    int num_classes;         
    int num_boxes;           
    float scoreThreshold;    
    float nmsThreshold;

    bool use_cuda_preproc_;   // 是否使用 CUDA 预处理

    // TensorRT 核心指针
    nvinfer1::IRuntime  *runtime = nullptr;             
    nvinfer1::ICudaEngine *engine = nullptr;            
    nvinfer1::IExecutionContext *context = nullptr;     
    cudaStream_t stream = nullptr;

    // 显存指针 (指向 GPU 内存)
    void* buffer_idx_0 = nullptr;                       
    void* buffer_idx_1 = nullptr;

    //CUDA 专用缓存：用于接收 20MP 原始图像数据
    uint8_t* d_src_img = nullptr; 
    int max_src_size = 0;

    // 绑定名称
    std::string input_name;                              
    std::string output_name;                             

    // 内存缓冲区 (CPU 端)
    std::vector<float> output_buffer_host;

    // Letterbox 关键参数
    float scale;                                       
    int pad_w;                                           
    int pad_h;

    //拆分预处理函数：按需调用 GPU 或 CPU
    void preprocess_cuda(cv::Mat &frame);  // GPU 加速路线
    void preprocess_cpu(cv::Mat &frame);   // CPU 常规路线
    void postprocessing();                 // 图像后处理 (公用)

public:
    std::vector<Result> detectResults;

    //构造函数增加 bool 参数，默认开启 CUDA 加速
    Model(const std::string modelPath, const int &inputSize,
          const float &scoreThreshold, const float &nmsThreshold, 
          bool use_cuda = true);
    
    ~Model();

    bool Detect(cv::Mat &frame);
};

#endif