#pragma once

#include "macros.h" // Include the macro definition header
#include <opencv2/opencv.hpp>
#include <NvInfer.h> // Include main TensorRT header
#include <vector>
#include <string>
#include <memory> // For smart pointers like unique_ptr

// Forward declare Logger to avoid including logging.h here if possible
// Or include it if the constructor needs the full type immediately.
// #include "logging.h"
class Logger; // Forward declaration

struct Detection {
    cv::Rect bbox;
    float score;     // CORRECTED: Was conf, should match postprocessing output
    int class_idx;   // CORRECTED: Was class_id
};


// Add the API macro here to export the class for the DLL
class API RF_DETR {
public:
    // Constructor takes logger by reference
    // Use nvinfer1::ILogger directly
    RF_DETR(const std::string& model_path, nvinfer1::ILogger& logger);

    // Destructor
    ~RF_DETR();

    // Public methods
    void preprocess(const cv::Mat& image); // Add const& for input image
    bool infer(); // Return bool for success/failure?
    // Add conf_threshold parameter to postprocess
    void postprocess(std::vector<Detection>& output, int originalWidth, int originalHeight, float conf_threshold);
    // Add const& for input detections
    void draw(cv::Mat& image, const std::vector<Detection>& output, int frame_idx, float conf_threshold = 0.5f);

private:
    // Private methods for internal implementation
    void init(const std::string& engine_path, nvinfer1::ILogger& logger);
    void build(const std::string& onnx_path, nvinfer1::ILogger& logger);
    void saveEngine(const std::string& path); // CORRECTED: Return type void

    // Use smart pointers for TensorRT objects for automatic memory management
    // Need to define custom deleters for TensorRT objects
    template <typename T>
    struct TrtDeleter {
        void operator()(T* obj) const {
            if (obj) {
                obj->destroy();
            }
        }
    };
    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

    // Member variables
    TrtUniquePtr<nvinfer1::IRuntime> runtime = nullptr;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine = nullptr;
    TrtUniquePtr<nvinfer1::IExecutionContext> context = nullptr;
    cudaStream_t stream = nullptr;

    // Buffers (consider using managed buffers or vectors)
    void* gpu_buffers[3]{ nullptr }; // Input, OutputBoxes, OutputScores (Adjust size if model differs)
    float* cpu_output_buffer_1 = nullptr; // Boxes
    float* cpu_output_buffer_2 = nullptr; // Scores (Or class indices? Verify model output)

    // Model parameters (initialized after engine creation)
    int input_batch_size = 1; // Assume batch size 1 for now
    int input_c;
    int input_h;
    int input_w;
    int input_idx = -1; // Store binding index

    int output_idx_boxes = -1;
    int output_idx_scores = -1;
    // Add more output indices if needed (e.g., classes)

    // Store output dimensions directly
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims_boxes;
    nvinfer1::Dims output_dims_scores;

    // Preallocation sizes based on model (example)
    size_t input_size_bytes = 0;
    size_t output_size_boxes_bytes = 0;
    size_t output_size_scores_bytes = 0;
    size_t cpu_output_size_boxes_bytes = 0;
    size_t cpu_output_size_scores_bytes = 0;


    // Constants (adjust as needed)
    static const int MAX_IMAGE_SIZE = 1920 * 1080; // Example max size (for preallocation?) - Be careful with fixed constants
};