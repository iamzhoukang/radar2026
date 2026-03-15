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
// 接收 use_cuda 参数并初始化
Model::Model(const std::string modelPath, const int &inputSize, 
             const float &scoreThreshold, const float &nmsThreshold, 
             bool use_cuda) 
{
    this->input_w = inputSize;
    this->input_h = inputSize;
    this->scoreThreshold = scoreThreshold;
    this->nmsThreshold = nmsThreshold;
    this->use_cuda_preproc_ = use_cuda; // 记录当前模型是否开启 GPU 加速
    
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

    std::cout << "Model Loaded: " << modelPath 
              << (use_cuda ? " [CUDA Enabled]" : " [CPU Preproc]") << std::endl;

    cudaMalloc(&(this->buffer_idx_0), 3 * this->input_h * this->input_w * sizeof(float));
    cudaMalloc(&(this->buffer_idx_1), outputSize * sizeof(float));
    this->output_buffer_host.resize(outputSize);
    cudaStreamCreate(&(this->stream));
}

// ==========================================
//  A：极限性能 GPU 预处理
// ==========================================
void Model::preprocess_cuda(cv::Mat &frame)
{
    // 1. 动态管理显存，确保能装下传进来的任意尺寸图片
    int img_size = frame.cols * frame.rows * 3 * sizeof(uint8_t);
    
    if (img_size > this->max_src_size) {
        if (this->d_src_img) cudaFree(this->d_src_img);
        cudaMalloc((void**)&(this->d_src_img), img_size);
        this->max_src_size = img_size;
    }

    // 2. 计算缩放与填充参数
    float scale_x = (float)this->input_w / frame.cols;
    float scale_y = (float)this->input_h / frame.rows;
    this->scale = std::min(scale_x, scale_y);
    int new_w = frame.cols * this->scale;
    int new_h = frame.rows * this->scale;
    this->pad_w = (this->input_w - new_w) / 2;
    this->pad_h = (this->input_h - new_h) / 2;

    // 3. 将原图(通常为大图) 直接通过 PCIe 异步砸入显存
    cudaMemcpyAsync(this->d_src_img, frame.data, img_size, cudaMemcpyHostToDevice, this->stream);

    // 4. 唤醒 CUDA 核函数干活，结果直接存入 TensorRT 的输入端 buffer_idx_0
    launch_preprocess_kernel(
        this->d_src_img, frame.step[0], frame.cols, frame.rows, 
        (float*)this->buffer_idx_0, this->input_w, this->input_h, 
        this->stream);
}

// ==========================================
// 策略 B：OpenCV 预处理
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

    cv::Mat resized, canvas;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    canvas = cv::Mat::zeros(this->input_h, this->input_w, CV_8UC3);
    canvas.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(this->pad_w, this->pad_h, new_w, new_h)));

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0/255.0, cv::Size(), cv::Scalar(), true);

    cudaMemcpyAsync(this->buffer_idx_0, blob.ptr<float>(),
                    3 * this->input_h * this->input_w * sizeof(float),
                    cudaMemcpyHostToDevice, this->stream);
}

// ==========================================
// 核心检测调度
// ==========================================
bool Model::Detect(cv::Mat &frame)
{
    if(frame.empty()) return false;
    this->detectResults.clear();

    //算力路由：根据开关智能选择预处理路线
    if (this->use_cuda_preproc_) {
        preprocess_cuda(frame);
    } else {
        preprocess_cpu(frame);
    }

    this->context->setInputTensorAddress(this->input_name.c_str(), this->buffer_idx_0);
    this->context->setOutputTensorAddress(this->output_name.c_str(), this->buffer_idx_1);

    bool status = this->context->enqueueV3(this->stream);
    if(!status) return false;

    cudaMemcpyAsync(this->output_buffer_host.data(), this->buffer_idx_1,
                    this->output_buffer_host.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, this->stream);

    cudaStreamSynchronize(this->stream);
    postprocessing();

    return !this->detectResults.empty();
}

// ==========================================
// 后处理与清理 
// ==========================================
void Model::postprocessing()
{
    float* output = this->output_buffer_host.data();
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

Model::~Model()
{
    cudaStreamSynchronize(this->stream);
    cudaStreamDestroy(this->stream);

    if(this->buffer_idx_0) cudaFree(this->buffer_idx_0);
    if(this->buffer_idx_1) cudaFree(this->buffer_idx_1);
    
    // 新增清理：释放承载大图的额外 GPU 缓存
    if(this->d_src_img) cudaFree(this->d_src_img); 

    delete this->context;
    delete this->engine;
    delete this->runtime;
}