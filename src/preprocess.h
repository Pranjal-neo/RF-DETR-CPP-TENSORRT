#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream);

// NEW: Declaration for the GPU sigmoid function.
void cuda_compute_sigmoid(const float* d_input, float* d_output, int N, cudaStream_t stream);
