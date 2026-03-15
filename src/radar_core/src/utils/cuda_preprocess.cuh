#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// 暴露出给 C++ 调用的接口
// src: 内存中的原图指针 (假设格式为 HWC 连续或带步长的 BGR 格式)
// src_step: 原图每一行的真实字节数 (对应 cv::Mat 的 step)
// src_w, src_h: 原图宽高
// dst: 处理好后要喂给 YOLO 的显存地址 (格式为 NCHW RGB, 范围 0.0~1.0)
void launch_preprocess_kernel(
    uint8_t* src, int src_step, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    cudaStream_t stream
);

#endif