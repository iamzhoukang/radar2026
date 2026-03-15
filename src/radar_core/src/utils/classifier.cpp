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
            std::cout << "[Classifier] " << msg << std::endl;
    }
}cLogger;

// 构造函数
Classifier::Classifier(const std::string &modelPath, const int inputSize)
{
    this->input_w = inputSize;
    this->input_h = inputSize;

     // 1. 读取二进制 .engine 文件
     std::ifstream file(modelPath, std::ios::binary);
     if (!file.good()) {
        std::cerr << "failed to open:" << modelPath << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> data(size);
    file.read(data.data(),size);
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
    auto out_dims = engine->getTensorShape(output_name.c_str());// 输出维度
    // 输出通常是 [1, num_classes] -> dim[1] 就是类别数
    num_classes = out_dims.d[1];
    output_buffer_host.resize(num_classes);// 预先分配空间，避免后面频繁分配内存

    std::cout << "分类模型加载成功: " << modelPath << std::endl;
    std::cout << "  - 输入: " << inputSize << "x" << inputSize << std::endl;
    std::cout << "  - 类别数: " << num_classes << std::endl;

    //5. 分配显存
    // 输入: 1 * 3 * H * W * float
    cudaMalloc(&buffer_idx_0,3 * input_w * input_h * sizeof(float));
    // 输出: 1 * num_classes * float
    cudaMalloc(&buffer_idx_1, num_classes * sizeof(float));

    cudaStreamCreate(&stream);
}

// 析构函数
Classifier::~Classifier()
{
    if (stream) cudaStreamDestroy(stream);
    if (buffer_idx_0) cudaFree(buffer_idx_0);
    if (buffer_idx_1) cudaFree(buffer_idx_1);
    delete context;
    delete engine;
    delete runtime;
}

// 预处理：OpenCV Mat -> GPU Float
void Classifier::preprocessing(cv::Mat &roi)
{
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(input_w, input_h));

     // 2. 归一化 + 格式转换
    // blobFromImage 参数详解:
    // scale: 1.0/255.0 (把 0-255 变成 0-1)
    // size: 目标大小
    // mean: (0,0,0) 不减均值
    // swapRB: true (因为 OpenCV 是 BGR，模型训练是 RGB)
    // crop: false
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, cv::Size(), cv::Scalar(), true, false);

     // 3. 拷贝到 GPU
     cudaMemcpyAsync(buffer_idx_0, blob.ptr<float>(),
                    3 * input_w * input_h * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

// 辅助函数：Softmax
void Classifier::softmax(std::vector<float> &input){
    float sum = 0.0f;
    float max_val = *std::max_element(input.begin(),input.end());//防止溢出

    for(auto &val : input){
        val = exp(val - max_val);//
        sum += val;
    }
    for(auto &val : input){
        val /= sum;
    }
}

int Classifier::Classify(cv::Mat &roi,float &confidence)
{
    if(roi.empty()) return -1;

    // 1. 预处理
    preprocessing(roi);

    //2. 设置地址 (TRT 10)
    context->setInputTensorAddress(input_name.c_str(), buffer_idx_0);
    context->setOutputTensorAddress(output_name.c_str(), buffer_idx_1);
    
    // 3. 推理
    context->enqueueV3(stream);

    // 4. 取回结果
    cudaMemcpyAsync(output_buffer_host.data(),buffer_idx_1,
                    num_classes * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. 后处理：Softmax + ArgMax
    // 先做 Softmax 得到概率
    softmax(output_buffer_host);

    //找最大值的索引和数值
    auto max_it = std::max_element(output_buffer_host.begin(), output_buffer_host.end());
    int max_idx = std::distance(output_buffer_host.begin(), max_it);
    confidence = *max_it; //现在这里是 0.0~1.0 的概率了

    return max_idx;
}
