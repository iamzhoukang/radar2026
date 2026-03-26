#include "classifier.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

// 定义一个简单的 Logger，TensorRT 需要它
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override{
        // 只打印错误信息，避免刷屏
        if (severity <= Severity::kERROR) 
            std::cout << "[Classifier TRT] " << msg << std::endl;
    }
} cLogger;

// ==========================================
// 构造函数
// ==========================================
Classifier::Classifier(const std::string &modelPath, const int inputSize)
{
    this->input_w = inputSize;
    this->input_h = inputSize;

    // 1. 读取二进制 .engine 文件
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "failed to open: " << modelPath << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    // 2. 初始化 TensorRT
    runtime = nvinfer1::createInferRuntime(cLogger);
    engine = runtime->deserializeCudaEngine(data.data(), size);
    context = engine->createExecutionContext();

    // 3. 自动寻找输入输出节点名称 (适配 TensorRT 10)
    int nbIOTensors = engine->getNbIOTensors();
    for(int i = 0; i < nbIOTensors; ++i){
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        if(mode == nvinfer1::TensorIOMode::kINPUT){
            input_name = name;
        }else if(mode == nvinfer1::TensorIOMode::kOUTPUT){
            output_name = name;
        }
    }

    // 4. 获取输出维度，确定类别数量
    auto out_dims = engine->getTensorShape(output_name.c_str());
    num_classes = out_dims.d[1];

    std::cout << "分类模型加载成功: " << modelPath << std::endl;
    std::cout << "  - 输入: " << inputSize << "x" << inputSize << std::endl;
    std::cout << "  - 类别数: " << num_classes << std::endl;

    // 5. 极致内存优化：分配显存 (Device) 与 锁页内存 (Host)
    // 输入端分配：
    cudaMalloc(&buffer_idx_0, 3 * input_w * input_h * sizeof(float));
    cudaMallocHost((void**)&pinned_in_host, 3 * input_w * input_h * sizeof(float));
    
    // 输出端分配：
    cudaMalloc(&buffer_idx_1, num_classes * sizeof(float));
    cudaMallocHost((void**)&pinned_out_host, num_classes * sizeof(float));

    // 6. 创建异步计算流
    cudaStreamCreate(&stream);
}

// ==========================================
// 析构函数 (严谨的资源回收)
// ==========================================
Classifier::~Classifier()
{
    if (stream) cudaStreamDestroy(stream);
    
    // 释放显存
    if (buffer_idx_0) cudaFree(buffer_idx_0);
    if (buffer_idx_1) cudaFree(buffer_idx_1);
    
    // 释放锁页内存
    if (pinned_in_host) cudaFreeHost(pinned_in_host);
    if (pinned_out_host) cudaFreeHost(pinned_out_host);
    
    delete context;
    delete engine;
    delete runtime;
}

// ==========================================
// 预处理：高速手工内存排布 + 异步传输
// ==========================================
void Classifier::preprocessing(cv::Mat &roi)
{
    cv::Mat resized;
    // 保留 CPU 的高速 resize
    cv::resize(roi, resized, cv::Size(input_w, input_h));

    // 【核心优化】：手工将 HWC(BGR) 拆解并归一化为 NCHW(RGB)，直接写入锁页内存
    int area = input_w * input_h;
    uint8_t* data = resized.data;
    
    
    for (int i = 0; i < area; ++i) {
        // data 数组按 [B, G, R, B, G, R...] 排列
        pinned_in_host[i]            = data[i * 3 + 2] / 255.0f; // R 通道写入前半段
        pinned_in_host[area + i]     = data[i * 3 + 1] / 255.0f; // G 通道写入中段
        pinned_in_host[area * 2 + i] = data[i * 3 + 0] / 255.0f; // B 通道写入后半段
    }

    // 此时数据已稳固存在于锁页内存中，触发真正的 DMA 异步传输，绝无悬空指针风险！
    cudaMemcpyAsync(buffer_idx_0, pinned_in_host,
                    3 * input_w * input_h * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

// ==========================================
// 辅助函数：Softmax
// ==========================================
void Classifier::softmax(std::vector<float> &input)
{
    float sum = 0.0f;
    float max_val = *std::max_element(input.begin(), input.end()); // 防止指数爆炸溢出

    for(auto &val : input){
        val = exp(val - max_val);
        sum += val;
    }
    for(auto &val : input){
        val /= sum;
    }
}

// ==========================================
// 核心推理函数
// ==========================================
int Classifier::Classify(cv::Mat &roi, float &confidence)
{
    if(roi.empty()) return -1;

    // 1. 预处理并将数据异步推入 GPU
    preprocessing(roi);

    // 2. 绑定 Tensor 地址 (TRT 10 语法)
    context->setInputTensorAddress(input_name.c_str(), buffer_idx_0);
    context->setOutputTensorAddress(output_name.c_str(), buffer_idx_1);
    
    // 3. 启动异步推理计算
    context->enqueueV3(stream);

    // 4. 将计算结果异步拉取回 CPU 的锁页内存
    cudaMemcpyAsync(pinned_out_host, buffer_idx_1,
                    num_classes * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    // 5. 设立同步墙：CPU 停下等待当前流内的所有任务（传输+计算+传回）完成
    cudaStreamSynchronize(stream);

    // 6. 后处理：转移到 vector 以复用 Softmax 逻辑
    std::vector<float> output_data(pinned_out_host, pinned_out_host + num_classes);
    softmax(output_data);

    // 7. 找最大值的索引和概率
    auto max_it = std::max_element(output_data.begin(), output_data.end());
    int max_idx = std::distance(output_data.begin(), max_it);
    confidence = *max_it; 

    return max_idx;
}
