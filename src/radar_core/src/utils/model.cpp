#include "model.hpp"
#include <fstream>
#include <algorithm> 

#include "cuda_preprocess.cuh" 

class Logger : public nvinfer1::ILogger {
    void log(Severity severity,const char* msg) noexcept override {
        if(severity <= Severity::kWARNING)
            std::cout<<"[TRT]"<<msg<<std::endl;
    }     
} gLogger;

// ==========================================
// 构造函数
// ==========================================
Model::Model(const std::string modelPath, const int &inputSize, 
             const float &scoreThreshold, const float &nmsThreshold, 
             bool use_cuda) 
{
    this->input_w = inputSize;
    this->input_h = inputSize;
    this->scoreThreshold = scoreThreshold;
    this->nmsThreshold = nmsThreshold;
    this->use_cuda_preproc_ = use_cuda; 
    
    std::ifstream engineFile(modelPath, std::ios::binary);
    if(!engineFile.good()){
        throw std::runtime_error("Error cannot open engine file: " + modelPath);
    }
    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();  
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize); 
    engineFile.close();

    this->runtime = nvinfer1::createInferRuntime(gLogger);
    this->engine = this->runtime->deserializeCudaEngine(engineData.data(), fsize);
    this->context = this->engine->createExecutionContext();

    int nbIOTensors = this->engine->getNbIOTensors();
    for(int i=0; i<nbIOTensors; ++i){
        const char* name = this->engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = this->engine->getTensorIOMode(name);
        if(mode == nvinfer1::TensorIOMode::kINPUT){
            this->input_name = name;
        }else if(mode == nvinfer1::TensorIOMode::kOUTPUT){
            this->output_name = name; 
        }
    }
    
    nvinfer1::Dims outputDims = this->engine->getTensorShape(this->output_name.c_str());
    int channels = outputDims.d[1];
    this->num_classes = channels - 4;
    this->num_boxes = outputDims.d[2];

    int outputSize = 1;
    for(int i=0; i<outputDims.nbDims; ++i) outputSize *= outputDims.d[i];
    this->output_size_ = outputSize; // 保存总长度，分配内存时使用

    std::cout << "Model Loaded: " << modelPath 
              << (use_cuda ? " [CUDA Enabled]" : " [CPU Preproc]") << std::endl;

    // 1. 分配 GPU 显存
    cudaMalloc(&(this->buffer_idx_0), 3 * this->input_h * this->input_w * sizeof(float));
    cudaMalloc(&(this->buffer_idx_1), this->output_size_ * sizeof(float));
    
    // 2. 【核心重构：分配主机端锁页内存】
    cudaMallocHost((void**)&(this->pinned_in_host), 3 * this->input_h * this->input_w * sizeof(float));
    cudaMallocHost((void**)&(this->pinned_out_host), this->output_size_ * sizeof(float));

    cudaStreamCreate(&(this->stream));
}

// ==========================================
// 策略 A：极限性能 GPU 预处理 (保持原样，极其优秀)
// ==========================================
void Model::preprocess_cuda(cv::Mat &frame)
{
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

    launch_preprocess_kernel(
        this->d_src_img, frame.step[0], frame.cols, frame.rows, 
        (float*)this->buffer_idx_0, this->input_w, this->input_h, 
        this->stream);
}

// ==========================================
// 策略 B：安全的 CPU 预处理 (重构版)
// ==========================================
void Model::preprocess_cpu(cv::Mat &frame)
{
    float scale_x = (float)this->input_w / frame.cols;
    float scale_y = (float)this->input_h / frame.rows;
    this->scale = std::min(scale_x, scale_y);

    int new_w = frame.cols * this->scale;
    int new_h = frame.rows * this->scale;
    this->pad_w = (this->input_w - new_w) / 2;
    this->pad_h = (this->input_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // 【核心重构：手动 Letterbox 填充 + 格式重排】
    int area = this->input_w * this->input_h;
    
    // 1. 先用 114 填满背景 (等同于 cv::Scalar(114,114,114) 的归一化值)
    std::fill(this->pinned_in_host, this->pinned_in_host + area * 3, 114.0f / 255.0f);

    // 2. 将缩放后的图像数据剥离并塞入锁页内存的中心区域
    uint8_t* data = resized.data;
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_idx = (y * new_w + x) * 3;
            int dst_idx = (y + this->pad_h) * this->input_w + (x + this->pad_w);

            this->pinned_in_host[dst_idx]            = data[src_idx + 2] / 255.0f; // R
            this->pinned_in_host[area + dst_idx]     = data[src_idx + 1] / 255.0f; // G
            this->pinned_in_host[area * 2 + dst_idx] = data[src_idx + 0] / 255.0f; // B
        }
    }

    // 3. 此时内存绝对安全，触发极致的 DMA 异步拷贝！
    cudaMemcpyAsync(this->buffer_idx_0, this->pinned_in_host,
                    3 * area * sizeof(float),
                    cudaMemcpyHostToDevice, this->stream);
}

// ==========================================
// 核心检测调度
// ==========================================
bool Model::Detect(cv::Mat &frame)
{
    if(frame.empty()) return false;
    this->detectResults.clear();

    if (this->use_cuda_preproc_) {
        preprocess_cuda(frame);
    } else {
        preprocess_cpu(frame);
    }

    this->context->setInputTensorAddress(this->input_name.c_str(), this->buffer_idx_0);
    this->context->setOutputTensorAddress(this->output_name.c_str(), this->buffer_idx_1);

    bool status = this->context->enqueueV3(this->stream);
    if(!status) return false;

    // 【核心重构：将结果拷贝回 CPU 的锁页内存中】
    cudaMemcpyAsync(this->pinned_out_host, this->buffer_idx_1,
                    this->output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, this->stream);

    cudaStreamSynchronize(this->stream);
    postprocessing();

    return !this->detectResults.empty();
}

// ==========================================
// 后处理
// ==========================================
void Model::postprocessing()
{
    // 【核心重构：直接从锁页内存读取计算结果】
    float* output = this->pinned_out_host;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    float inv_scale = 1.0f / this->scale;
    int stride = this->num_boxes;

    for(int i=0; i<this->num_boxes; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        for(int c=0; c<this->num_classes; ++c) {
            float score = output[(4+c) * stride + i];
            if(score > max_score){
                max_score = score;
                class_id = c;
            }
        }

        if(max_score > this->scoreThreshold) {
             float cx = output[0 * stride + i];
             float cy = output[1 * stride + i];
             float w = output[2 * stride + i];
             float h = output[3 * stride + i];

             int left   = static_cast<int>((cx - this->pad_w) * inv_scale - w * inv_scale * 0.5f);
             int top    = static_cast<int>((cy - this->pad_h) * inv_scale - h * inv_scale * 0.5f);
             int width  = static_cast<int>(w * inv_scale);
             int height = static_cast<int>(h * inv_scale);

             boxes.push_back(cv::Rect(left, top, width, height));
             classIds.push_back(class_id);
             confidences.push_back(max_score);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, indices);

    for(int idx : indices) {
        this->detectResults.emplace_back(Result{classIds[idx], confidences[idx], boxes[idx]});
    }
}

// ==========================================
// 析构函数
// ==========================================
Model::~Model()
{
    cudaStreamSynchronize(this->stream);
    cudaStreamDestroy(this->stream);

    if(this->buffer_idx_0) cudaFree(this->buffer_idx_0);
    if(this->buffer_idx_1) cudaFree(this->buffer_idx_1);
    if(this->d_src_img) cudaFree(this->d_src_img); 

    // 【核心重构：安全释放锁页内存】
    if(this->pinned_in_host) cudaFreeHost(this->pinned_in_host);
    if(this->pinned_out_host) cudaFreeHost(this->pinned_out_host);

    delete this->context;
    delete this->engine;
    delete this->runtime;
}