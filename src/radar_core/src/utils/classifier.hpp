#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Classifier
{
public:
    // 构造函数：传入模型路径(.engine) 和 输入大小(64)
    Classifier(const std::string &modelPath, const int inputSize);
    
    // 析构函数：释放显存和指针，严格执行 RAII
    ~Classifier();

    // 核心函数：输入一张小图(ROI)，返回类别ID，并通过引用返回置信度
    int Classify(cv::Mat &roi, float &confidence);

private:
    int input_w;
    int input_h;
    int num_classes; // 自动从模型获取类别数(10)

    // TensorRT 核心指针
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;  

    // 显存指针 (Device Memory)
    void *buffer_idx_0 = nullptr;
    void *buffer_idx_1 = nullptr;

    // 【主机端核心优化】：锁页内存指针 (Pinned Host Memory)
    // 用于实现 CPU 与 GPU 之间真正的零拷贝 (Zero-Copy) 异步 DMA 传输
    float *pinned_in_host = nullptr;
    float *pinned_out_host = nullptr;

    // 节点名称 (TRT 10 必需)
    std::string input_name;
    std::string output_name;

    // 内部预处理函数
    void preprocessing(cv::Mat &roi);

    // 简单的 Softmax 函数，把原始分数变成概率
    void softmax(std::vector<float> &input);
};

#endif