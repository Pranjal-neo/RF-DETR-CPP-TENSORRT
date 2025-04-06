#include "RF_DETR.h"
#include "logging.h" // Include your logger implementation
#include "preprocess.h" // Include CUDA preprocessing header
#include "cuda_utils.h" // Include CUDA utility header

#include <fstream>
#include <vector>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::max_element

// Include TensorRT headers
#include <NvInfer.h>
#include <NvOnnxParser.h> // Include ONNX parser header

// Constructor
RF_DETR::RF_DETR(const std::string& model_path, nvinfer1::ILogger& logger) {
    // Check if model file exists
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, ("Model file not found: " + model_path).c_str());
        throw std::runtime_error("Model file not found: " + model_path);
    }
    file.close(); // Close file after check

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Determine if it's an ONNX file or TensorRT engine file
    if (model_path.size() > 5 && model_path.substr(model_path.size() - 5) == ".onnx") {
        logger.log(nvinfer1::ILogger::Severity::kINFO, "Building engine from ONNX file...");
        build(model_path, logger);
        // Optionally save the built engine
        // saveEngine(model_path + ".engine"); // Create engine path logic if needed
    } else {
        logger.log(nvinfer1::ILogger::Severity::kINFO, "Loading pre-built TensorRT engine...");
        init(model_path, logger);
    }

     if (!engine || !context) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Model initialization failed.");
         throw std::runtime_error("Failed to initialize TensorRT engine or context.");
     }

    // Allocate CPU output buffers based on engine bindings
    cpu_output_buffer_1 = new float[output_size_boxes_bytes / sizeof(float)];
    cpu_output_buffer_2 = new float[output_size_scores_bytes / sizeof(float)];
    cpu_output_size_boxes_bytes = output_size_boxes_bytes; // Store for later use
    cpu_output_size_scores_bytes = output_size_scores_bytes;

    logger.log(nvinfer1::ILogger::Severity::kINFO, "RF_DETR model initialized successfully.");
}

// Destructor
RF_DETR::~RF_DETR() {
    // Release CUDA stream
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
    }

    // Free GPU buffers
    if (gpu_buffers[input_idx] != nullptr) CUDA_CHECK(cudaFree(gpu_buffers[input_idx]));
    if (gpu_buffers[output_idx_boxes] != nullptr) CUDA_CHECK(cudaFree(gpu_buffers[output_idx_boxes]));
    if (gpu_buffers[output_idx_scores] != nullptr) CUDA_CHECK(cudaFree(gpu_buffers[output_idx_scores])); // Add check for score buffer index if used

    // Free CPU buffers
    delete[] cpu_output_buffer_1;
    delete[] cpu_output_buffer_2;

    // Smart pointers (runtime, engine, context) handle their own destruction
    // logger is passed by reference, not owned
    std::cout << "RF_DETR resources released." << std::endl;
}


// Initialize from saved engine file
void RF_DETR::init(const std::string& engine_path, nvinfer1::ILogger& logger) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to load engine file: " + engine_path).c_str());
        return;
    }

    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size]; // Use smart pointer std::vector<char>
    // std::vector<char> trtModelStream(size); // Recommended
    file.read(trtModelStream, size);
    // file.read(trtModelStream.data(), size); // If using vector
    file.close();

    // ADD nvinfer1:: prefix
    runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create TensorRT Runtime.");
         delete[] trtModelStream; // Manual cleanup if using raw pointer
         return;
    }

    // Pass correct arguments to deserializeCudaEngine
    engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size));
    // engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size)); // If using vector
    delete[] trtModelStream; // Manual cleanup if using raw pointer
    if (!engine) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to deserialize CUDA engine.");
        return;
    }

    context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create execution context.");
        return;
    }

    // --- Get Input/Output Dimensions and Allocate Buffers ---
    assert(engine->getNbBindings() >= 2); // Ensure at least input and one output

    input_idx = engine->getBindingIndex("images"); // Replace "images" with your actual input layer name
    if (input_idx == -1) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Could not find input layer named 'images'. Check ONNX model.");
         return;
    }
    auto input_dims = engine->getBindingDimensions(input_idx);
    input_batch_size = input_dims.d[0]; // Or verify it's dynamic/expected
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    input_size_bytes = input_batch_size * input_c * input_h * input_w * sizeof(float); // Assuming FP32 input
    CUDA_CHECK(cudaMalloc(&gpu_buffers[input_idx], input_size_bytes));

    output_idx_boxes = engine->getBindingIndex("output_boxes"); // Replace with your actual output layer name for boxes
     if (output_idx_boxes == -1) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Could not find output layer named 'output_boxes'. Check ONNX model.");
         return;
     }
    output_dims_boxes = engine->getBindingDimensions(output_idx_boxes);
    // Example: Dims(batch_size, num_detections, 4) -> num_elements = batch * num_det * 4
    size_t num_elements_boxes = 1;
    for(int i=0; i<output_dims_boxes.nbDims; ++i) num_elements_boxes *= output_dims_boxes.d[i];
    output_size_boxes_bytes = num_elements_boxes * sizeof(float); // Assuming FP32 output
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_boxes], output_size_boxes_bytes));


    output_idx_scores = engine->getBindingIndex("output_scores"); // Replace with actual score output name
     if (output_idx_scores == -1) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Could not find output layer named 'output_scores'. Check ONNX model.");
         return; // Or handle if scores are combined with boxes
     }
    output_dims_scores = engine->getBindingDimensions(output_idx_scores);
    // Example: Dims(batch_size, num_detections) -> num_elements = batch * num_det
    size_t num_elements_scores = 1;
    for(int i=0; i<output_dims_scores.nbDims; ++i) num_elements_scores *= output_dims_scores.d[i];
    output_size_scores_bytes = num_elements_scores * sizeof(float); // Assuming FP32 output
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_scores], output_size_scores_bytes));


    // Add similar logic for class indices if it's a separate output
    logger.log(nvinfer1::ILogger::Severity::kINFO, ("Input dimensions: " + std::to_string(input_batch_size) + "x" + std::to_string(input_c) + "x" + std::to_string(input_h) + "x" + std::to_string(input_w)).c_str());
}

// Build engine from ONNX file
void RF_DETR::build(const std::string& onnx_path, nvinfer1::ILogger& logger) {
    // ADD nvinfer1:: prefix
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "createInferBuilder failed");
        return;
    }

    // ADD nvinfer1:: prefixes and kEXPLICIT_BATCH
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "createNetworkV2 failed");
        return;
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "createBuilderConfig failed");
        return;
    }

    // --- ONNX Parser ---
    // ADD nvinfer1:: prefix
    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "createParser failed");
        return;
    }

    // Parse ONNX file
    logger.log(nvinfer1::ILogger::Severity::kINFO, ("Parsing ONNX file: " + onnx_path).c_str());
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to parse ONNX file");
        return;
    }
    logger.log(nvinfer1::ILogger::Severity::kINFO, "ONNX parsing completed successfully.");

    // --- Configure Builder ---
    // Set max workspace size (adjust based on your model and GPU memory)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB example

    // Set precision flags (e.g., FP16)
    if (builder->platformHasFastFp16()) {
        logger.log(nvinfer1::ILogger::Severity::kINFO, "Platform supports FP16. Enabling FP16 mode.");
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // ADD nvinfer1:: prefix
    } else {
        logger.log(nvinfer1::ILogger::Severity::kINFO, "Platform does not support FP16. Using FP32 mode.");
    }

    // --- Build Engine ---
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Building TensorRT engine...");
    // ADD nvinfer1:: prefix for IHostMemory
    TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "buildSerializedNetwork failed");
        return;
    }
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Engine building completed successfully.");


    // --- Deserialize Engine ---
    // Need runtime to deserialize (create it here if init isn't called)
    runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
     if (!runtime) {
         logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create TensorRT Runtime during build.");
         return;
     }

    engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to deserialize engine during build.");
        return;
    }

    // Create context after building engine
    context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create execution context after build.");
        return;
    }

     // --- Get Input/Output Dimensions and Allocate Buffers (same as init) ---
    assert(engine->getNbBindings() >= 2);

    input_idx = engine->getBindingIndex("images");
    if (input_idx == -1) { logger.log(nvinfer1::ILogger::Severity::kERROR, "Input 'images' not found post-build."); return; }
    auto input_dims = engine->getBindingDimensions(input_idx);
    input_batch_size = input_dims.d[0]; input_c = input_dims.d[1]; input_h = input_dims.d[2]; input_w = input_dims.d[3];
    input_size_bytes = input_batch_size * input_c * input_h * input_w * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[input_idx], input_size_bytes));

    output_idx_boxes = engine->getBindingIndex("output_boxes");
    if (output_idx_boxes == -1) { logger.log(nvinfer1::ILogger::Severity::kERROR, "Output 'output_boxes' not found post-build."); return; }
    output_dims_boxes = engine->getBindingDimensions(output_idx_boxes);
    size_t num_elements_boxes = 1; for(int i=0; i<output_dims_boxes.nbDims; ++i) num_elements_boxes *= output_dims_boxes.d[i];
    output_size_boxes_bytes = num_elements_boxes * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_boxes], output_size_boxes_bytes));

    output_idx_scores = engine->getBindingIndex("output_scores");
    if (output_idx_scores == -1) { logger.log(nvinfer1::ILogger::Severity::kERROR, "Output 'output_scores' not found post-build."); return; }
    output_dims_scores = engine->getBindingDimensions(output_idx_scores);
    size_t num_elements_scores = 1; for(int i=0; i<output_dims_scores.nbDims; ++i) num_elements_scores *= output_dims_scores.d[i];
    output_size_scores_bytes = num_elements_scores * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_scores], output_size_scores_bytes));

    // No need to delete network, config, plan, builder - smart pointers handle it.
}


// Preprocess image (CPU -> GPU)
void RF_DETR::preprocess(const cv::Mat& image) {
    if (!context) {
        std::cerr << "ERROR: Execution context is null in preprocess." << std::endl;
        return;
    }
     if (gpu_buffers[input_idx] == nullptr) {
         std::cerr << "ERROR: Input GPU buffer is null in preprocess." << std::endl;
         return;
     }

    // Call CUDA preprocessing function from preprocess.h/.cu
    // Pass the required arguments, casting gpu_buffers[input_idx]
    cuda_preprocess(
        image.data,         // Input image data (uint8_t*)
        image.cols,         // Image width
        image.rows,         // Image height
        static_cast<float*>(gpu_buffers[input_idx]), // Dest GPU buffer (CASTED)
        input_w,            // Network input width
        input_h,            // Network input height
        stream              // CUDA stream
    );
    // Note: cuda_preprocess should handle resizing, normalization, channel conversion (BGR->RGB), and HWC->CHW transposition
}

// Run inference
bool RF_DETR::infer() {
    if (!context) {
        std::cerr << "ERROR: Execution context is null in infer." << std::endl;
        return false;
    }
    // Assuming gpu_buffers[0] is input, [1] is boxes output, [2] is scores output
    // This mapping depends on the binding indices obtained earlier
    // Ensure gpu_buffers indices match input_idx, output_idx_boxes, output_idx_scores
    bool status = context->enqueueV2(gpu_buffers, stream, nullptr); // Use enqueueV2
    if (!status) {
        std::cerr << "ERROR: TensorRT inference enqueue failed." << std::endl;
        return false;
    }

    // Synchronize the stream to wait for inference completion
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy outputs from GPU to CPU
    // Make sure indices and sizes match
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_1, gpu_buffers[output_idx_boxes], output_size_boxes_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_2, gpu_buffers[output_idx_scores], output_size_scores_bytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize the stream again to wait for memory copies
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return true;
}

// Postprocess (GPU -> CPU, parse results)
void RF_DETR::postprocess(std::vector<Detection>& output, int originalWidth, int originalHeight, float conf_threshold) {
    output.clear(); // Clear previous detections

    // Assuming cpu_output_buffer_1 contains boxes [num_detections, 4]
    // Assuming cpu_output_buffer_2 contains scores [num_detections, num_classes] or just scores [num_detections]
    // Check output_dims_boxes and output_dims_scores to confirm layout

    // Example logic assuming boxes [N, 4] and scores [N] (single score per box)
    // Adjust based on your model's actual output structure
    if (output_dims_boxes.nbDims != 2 || output_dims_scores.nbDims != 1 || output_dims_boxes.d[0] != output_dims_scores.d[0]) {
         std::cerr << "ERROR: Unexpected output dimensions for boxes/scores in postprocess." << std::endl;
         // Log dims for debugging
         std::cerr << "Box dims: "; for(int i=0; i<output_dims_boxes.nbDims; ++i) std::cerr << output_dims_boxes.d[i] << " "; std::cerr << std::endl;
         std::cerr << "Score dims: "; for(int i=0; i<output_dims_scores.nbDims; ++i) std::cerr << output_dims_scores.d[i] << " "; std::cerr << std::endl;
         return;
    }

    int num_detections = output_dims_boxes.d[0]; // Number of potential detections

    for (int i = 0; i < num_detections; ++i) {
        float score = cpu_output_buffer_2[i]; // Get score for detection i

        // CORRECTED: Use conf_threshold passed as argument
        if (score >= conf_threshold) {
            float* bbox_ptr = cpu_output_buffer_1 + i * 4; // Pointer to box i [x1, y1, x2, y2]

            // Scale coordinates back to original image size
            // IMPORTANT: Verify the coordinate format (x1y1x2y2 or xywh) and scaling logic
            float x1 = bbox_ptr[0] * originalWidth / input_w;
            float y1 = bbox_ptr[1] * originalHeight / input_h;
            float x2 = bbox_ptr[2] * originalWidth / input_w;
            float y2 = bbox_ptr[3] * originalHeight / input_h;

            Detection det;
            det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
            det.score = score;           // CORRECTED: Use score
            det.class_idx = 0; // Assign class index - Requires class output from model
                               // If model outputs class indices, read from appropriate buffer

            // Placeholder: Assign class based on max score if scores are [N, num_classes]
            // Example: Requires output_dims_scores.d[1] == num_classes
            // float* scores_for_det_i = cpu_output_buffer_2 + i * num_classes;
            // det.class_idx = std::max_element(scores_for_det_i, scores_for_det_i + num_classes) - scores_for_det_i;
            // det.score = scores_for_det_i[det.class_idx]; // Update score to max class score

            // Filter again if score was updated
            // if (det.score < conf_threshold) continue;

            output.push_back(det);
        }
    }
}

// Save engine to file
// CORRECTED: Return type void to match header
void RF_DETR::saveEngine(const std::string& path) {
     if (!engine) {
         std::cerr << "ERROR: Cannot save null engine." << std::endl;
         return; // Return void
     }
    // ADD nvinfer1:: prefix
    TrtUniquePtr<nvinfer1::IHostMemory> serializedEngine(engine->serialize());
    if (!serializedEngine) {
        std::cerr << "ERROR: Failed to serialize engine." << std::endl;
        return; // Return void
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Could not open file to save engine: " << path << std::endl;
        return; // Return void
    }
    file.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    file.close();
    std::cout << "Engine saved successfully to: " << path << std::endl;
    // Return void
}

// Draw bounding boxes
// CORRECTED: Add conf_threshold parameter to definition
void RF_DETR::draw(cv::Mat& image, const std::vector<Detection>& output, float conf_threshold) {
    for (const auto& d : output) {
        // Filter again here based on threshold passed to draw (optional, could rely on postprocess filtering)
        if (d.score < conf_threshold) {
            continue;
        }

        // CORRECTED: Use d.class_idx
        int classId = d.class_idx;
        // Add check for classId bounds if using CLASS_NAMES from common.h
        // if (classId < 0 || classId >= CLASS_NAMES.size()) continue;

        cv::rectangle(image, d.bbox, cv::Scalar(0, 255, 0), 2); // Green box

        // CORRECTED: Use d.score
        std::string label = "Class" + std::to_string(classId) + ": " + cv::format("%.2f", d.score);
        // Optional: Use class names from common.h
        // std::string label = CLASS_NAMES[classId] + ": " + cv::format("%.2f", d.score);

        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int top = std::max(d.bbox.y, labelSize.height);
        cv::rectangle(image, cv::Point(d.bbox.x, top - labelSize.height),
                      cv::Point(d.bbox.x + labelSize.width, top + baseline),
                      cv::Scalar(255, 255, 255), cv::FILLED); // White background for label
        cv::putText(image, label, cv::Point(d.bbox.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1); // Black text
    }
}