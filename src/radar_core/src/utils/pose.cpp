#include "pose.hpp"
#include <fstream>
#include <algorithm> 

// 引入你的 CUDA 预处理核函数声明
#include "cuda_preprocess.cuh" 

class PoseLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if(severity <= Severity::kWARNING)
            std::cout << "[TRT POSE] " << msg << std::endl;
    }     
} gPoseLogger;

// ==========================================
// 构造函数
// ==========================================
PoseModel::PoseModel(const std::string modelPath, const int &inputSize, 
                     const float &scoreThreshold, const float &nmsThreshold, 
                     int num_classes, int num_kpts, 
                     bool use_cuda) 
{
    this->input_w = inputSize;
    this->input_h = inputSize;
    this->scoreThreshold = scoreThreshold;
    this->nmsThreshold = nmsThreshold;
    this->num_classes = num_classes; 
    this->num_kpts = num_kpts;
    this->use_cuda_preproc_ = use_cuda; 
    
    std::ifstream engineFile(modelPath, std::ios::binary);
    if(!engineFile.good()) throw std::runtime_error("Error cannot open engine file: " + modelPath);
    
    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();  
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize); 
    engineFile.close();

    this->runtime = nvinfer1::createInferRuntime(gPoseLogger);
    this->engine = this->runtime->deserializeCudaEngine(engineData.data(), fsize);
    this->context = this->engine->createExecutionContext();

    int nbIOTensors = this->engine->getNbIOTensors();
    for(int i=0; i<nbIOTensors; ++i){
        const char* name = this->engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = this->engine->getTensorIOMode(name);
        if(mode == nvinfer1::TensorIOMode::kINPUT) this->input_name = name;
        else if(mode == nvinfer1::TensorIOMode::kOUTPUT) this->output_name = name; 
    }
    
    nvinfer1::Dims outputDims = this->engine->getTensorShape(this->output_name.c_str());
    int channels = outputDims.d[1];
    this->num_boxes = outputDims.d[2]; 

    // 终极安全锁：验证 TensorRT 模型通道数是否和传进来的参数对得上！
    int expected_channels = 4 + this->num_classes + this->num_kpts * 3;
    if (channels != expected_channels) {
        throw std::runtime_error(
            "Dimension Mismatch! Engine channels: " + std::to_string(channels) +
            ", expected: " + std::to_string(expected_channels) + 
            " (4(box) + " + std::to_string(num_classes) + "(cls) + 3*" + std::to_string(num_kpts) + "(kpts))"
        );
    }

    int outputSize = 1;
    for(int i=0; i<outputDims.nbDims; ++i) outputSize *= outputDims.d[i];
    this->output_size_ = outputSize;

    std::cout << "Pose Model Loaded: " << modelPath 
              << (use_cuda ? " [CUDA Enabled]" : " [CPU Preproc]") << std::endl;

    cudaMalloc(&(this->buffer_idx_0), 3 * this->input_h * this->input_w * sizeof(float));
    cudaMalloc(&(this->buffer_idx_1), this->output_size_ * sizeof(float));
    cudaMallocHost((void**)&(this->pinned_in_host), 3 * this->input_h * this->input_w * sizeof(float));
    cudaMallocHost((void**)&(this->pinned_out_host), this->output_size_ * sizeof(float));
    cudaStreamCreate(&(this->stream));
}

// ==========================================
// CUDA 预处理
// ==========================================
void PoseModel::preprocess_cuda(cv::Mat &frame) {
    int img_size = frame.cols * frame.rows * 3 * sizeof(uint8_t);
    if (img_size > this->max_src_size) {
        if (this->d_src_img) cudaFree(this->d_src_img);
        cudaMalloc((void**)&(this->d_src_img), img_size);
        this->max_src_size = img_size;
    }
    float scale_x = (float)this->input_w / frame.cols;
    float scale_y = (float)this->input_h / frame.rows;
    this->scale = std::min(scale_x, scale_y);
    int new_w = frame.cols * this->scale;
    int new_h = frame.rows * this->scale;
    this->pad_w = (this->input_w - new_w) / 2;
    this->pad_h = (this->input_h - new_h) / 2;

    cudaMemcpyAsync(this->d_src_img, frame.data, img_size, cudaMemcpyHostToDevice, this->stream);
    launch_preprocess_kernel(this->d_src_img, frame.step[0], frame.cols, frame.rows, 
                             (float*)this->buffer_idx_0, this->input_w, this->input_h, this->stream);
}

// ==========================================
// CPU 预处理 (备用)
// ==========================================
void PoseModel::preprocess_cpu(cv::Mat &frame) {
    float scale_x = (float)this->input_w / frame.cols;
    float scale_y = (float)this->input_h / frame.rows;
    this->scale = std::min(scale_x, scale_y);
    int new_w = frame.cols * this->scale;
    int new_h = frame.rows * this->scale;
    this->pad_w = (this->input_w - new_w) / 2;
    this->pad_h = (this->input_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    int area = this->input_w * this->input_h;
    std::fill(this->pinned_in_host, this->pinned_in_host + area * 3, 114.0f / 255.0f);

    uint8_t* data = resized.data;
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_idx = (y * new_w + x) * 3;
            int dst_idx = (y + this->pad_h) * this->input_w + (x + this->pad_w);
            this->pinned_in_host[dst_idx]            = data[src_idx + 2] / 255.0f; 
            this->pinned_in_host[area + dst_idx]     = data[src_idx + 1] / 255.0f; 
            this->pinned_in_host[area * 2 + dst_idx] = data[src_idx + 0] / 255.0f; 
        }
    }
    cudaMemcpyAsync(this->buffer_idx_0, this->pinned_in_host,
                    3 * area * sizeof(float), cudaMemcpyHostToDevice, this->stream);
}

// ==========================================
// 执行推理
// ==========================================
bool PoseModel::Detect(cv::Mat &frame) {
    if(frame.empty()) return false;
    this->detectResults.clear();

    if (this->use_cuda_preproc_) preprocess_cuda(frame);
    else preprocess_cpu(frame);

    this->context->setInputTensorAddress(this->input_name.c_str(), this->buffer_idx_0);
    this->context->setOutputTensorAddress(this->output_name.c_str(), this->buffer_idx_1);

    bool status = this->context->enqueueV3(this->stream);
    if(!status) return false;

    cudaMemcpyAsync(this->pinned_out_host, this->buffer_idx_1,
                    this->output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, this->stream);
    cudaStreamSynchronize(this->stream);
    
    postprocessing();
    return !this->detectResults.empty();
}

// ==========================================
// 后处理：全动态自适应剥离 (极致性能优化版)
// ==========================================
void PoseModel::postprocessing() {
    float* output = this->pinned_out_host;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    
    // 优化 1：不再提前分配二维的关键点数组，而是记录原始输出矩阵中的索引 i
    std::vector<int> valid_raw_indices; 

    // 优化 2：预分配内存，彻底避免 8400 次循环中的 vector 动态扩容开销
    boxes.reserve(100);
    classIds.reserve(100);
    confidences.reserve(100);
    valid_raw_indices.reserve(100);

    float inv_scale = 1.0f / this->scale;
    int stride = this->num_boxes; 
    int kpt_offset = 4 + this->num_classes; 

    for(int i = 0; i < this->num_boxes; ++i) {
        float max_score = 0.0f;
        int max_class_id = -1;
        
        // 优化 3：单类别推理短路优化 (Single-class fast path)
        if (this->num_classes == 1) {
            max_score = output[4 * stride + i];
            max_class_id = 0;
        } else {
            for (int c = 0; c < this->num_classes; ++c) {
                float score = output[(4 + c) * stride + i];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c;
                }
            }
        }

        // 第一轮初筛：只解码 Box
        if(max_score > this->scoreThreshold) {
             float cx = output[0 * stride + i];
             float cy = output[1 * stride + i];
             float w  = output[2 * stride + i];
             float h  = output[3 * stride + i];

             int left   = static_cast<int>((cx - this->pad_w) * inv_scale - w * inv_scale * 0.5f);
             int top    = static_cast<int>((cy - this->pad_h) * inv_scale - h * inv_scale * 0.5f);
             int width  = static_cast<int>(w * inv_scale);
             int height = static_cast<int>(h * inv_scale);

             boxes.emplace_back(cv::Rect(left, top, width, height));
             classIds.emplace_back(max_class_id);
             confidences.emplace_back(max_score);
             
             // 核心：只记录它在 output 里的原始索引，不碰关键点数据！
             valid_raw_indices.emplace_back(i); 
        }
    }

    // 执行 NMS，杀掉重叠的冗余框
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, indices);

    // 优化 4：延迟解码 (Lazy Decoding)。只为 NMS 幸存下来的极少数真目标解算关键点
    for(int idx : indices) {
        int raw_i = valid_raw_indices[idx]; // 拿出它在输出张量里的真实列号
        
        std::vector<PoseKeyPoint> kpts;
        kpts.reserve(this->num_kpts); // 预分配关键点空间

        for (int k = 0; k < this->num_kpts; ++k) {
            float kx = output[(kpt_offset + k * 3 + 0) * stride + raw_i];
            float ky = output[(kpt_offset + k * 3 + 1) * stride + raw_i];
            float kv = output[(kpt_offset + k * 3 + 2) * stride + raw_i];

            int raw_x = static_cast<int>((kx - this->pad_w) * inv_scale);
            int raw_y = static_cast<int>((ky - this->pad_h) * inv_scale);

            kpts.push_back({cv::Point2f(raw_x, raw_y), kv});
        }

        // 组装最终结果
        this->detectResults.emplace_back(PoseResult{
            classIds[idx], 
            confidences[idx], 
            boxes[idx], 
            std::move(kpts) // 使用 std::move 避免拷贝
        });
    }
}

// ==========================================
// 析构函数
// ==========================================
PoseModel::~PoseModel() {
    cudaStreamSynchronize(this->stream);
    cudaStreamDestroy(this->stream);
    if(this->buffer_idx_0) cudaFree(this->buffer_idx_0);
    if(this->buffer_idx_1) cudaFree(this->buffer_idx_1);
    if(this->d_src_img) cudaFree(this->d_src_img); 
    if(this->pinned_in_host) cudaFreeHost(this->pinned_in_host);
    if(this->pinned_out_host) cudaFreeHost(this->pinned_out_host);
    delete this->context;
    delete this->engine;
    delete this->runtime;
}