#include "RF_DETR.h"
#include "logging.h"
#include "preprocess.h"
#include "cuda_utils.h"

#include <fstream>
#include <iostream>
#include <NvOnnxParser.h>
#include <cmath>  // for std::exp

//----------------------------------------------------------------------------//
// Constructor / Destructor
//----------------------------------------------------------------------------//

RF_DETR::RF_DETR(const std::string& model_path,
                 nvinfer1::ILogger& logger)
{
    std::cout << "[RF_DETR] Constructor: loading model '" << model_path << "'\n";
    std::ifstream f(model_path, std::ios::binary);
    if (!f.good()) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                   ("Model file not found: " + model_path).c_str());
        throw std::runtime_error("Model file not found");
    }
    f.close();

    CUDA_CHECK(cudaStreamCreate(&stream));

    if (model_path.size() > 5 &&
        model_path.substr(model_path.size() - 5) == ".onnx")
    {
        std::cout << "[RF_DETR] Building engine from ONNX\n";
        build(model_path, logger);
    }
    else
    {
        std::cout << "[RF_DETR] Loading serialized engine\n";
        init(model_path, logger);
    }

    if (!engine || !context)
        throw std::runtime_error("Failed to initialize TRT engine");

    cpu_output_buffer_1 = new float[output_size_boxes_bytes / sizeof(float)];
    cpu_output_buffer_2 = new float[output_size_scores_bytes / sizeof(float)];

    std::cout << "[RF_DETR] Initialized successfully\n";
}

RF_DETR::~RF_DETR()
{
    std::cout << "[RF_DETR] Destructor: freeing resources\n";
    if (stream)
        cudaStreamDestroy(stream);

    for (int idx : {input_idx, output_idx_boxes, output_idx_scores})
        if (gpu_buffers[idx])
            cudaFree(gpu_buffers[idx]);

    delete[] cpu_output_buffer_1;
    delete[] cpu_output_buffer_2;
}

//----------------------------------------------------------------------------//
// init(): deserialize engine + allocate buffers
//----------------------------------------------------------------------------//

void RF_DETR::init(const std::string& engine_path,
                   nvinfer1::ILogger& logger)
{
    if (!engine_path.empty()) {
        std::ifstream file(engine_path, std::ios::binary);
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> buf(size);
        file.read(buf.data(), size);
        file.close();

        runtime = TrtUniquePtr<nvinfer1::IRuntime>(
                      nvinfer1::createInferRuntime(logger));
        engine  = TrtUniquePtr<nvinfer1::ICudaEngine>(
                      runtime->deserializeCudaEngine(buf.data(), buf.size()));
        context = TrtUniquePtr<nvinfer1::IExecutionContext>(
                      engine->createExecutionContext());
    }

    int nb = engine->getNbIOTensors();
    std::cout << "[RF_DETR:init] Found " << nb << " I/O tensors\n";
    for (int i = 0; i < nb; ++i) {
        const char* name  = engine->getIOTensorName(i);
        auto mode          = engine->getTensorIOMode(name);
        auto shape         = engine->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT &&
            std::string(name) == "input")
        {
            input_idx        = i;
            input_dims       = shape;
            input_batch_size = shape.d[0];
            input_c          = shape.d[1];
            input_h          = shape.d[2];
            input_w          = shape.d[3];
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT &&
                 std::string(name) == "dets")
        {
            output_idx_boxes  = i;
            output_dims_boxes = shape;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT &&
                 std::string(name) == "labels")
        {
            output_idx_scores  = i;
            output_dims_scores = shape;
        }
    }

    // Allocate GPU buffers
    input_size_bytes =
        input_batch_size * input_c * input_h * input_w * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[input_idx],
                          input_size_bytes));

    size_t nbBoxes = 1;
    for (int d = 0; d < output_dims_boxes.nbDims; ++d)
        nbBoxes *= output_dims_boxes.d[d];
    output_size_boxes_bytes = nbBoxes * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_boxes],
                          output_size_boxes_bytes));

    size_t nbScores = 1;
    for (int d = 0; d < output_dims_scores.nbDims; ++d)
        nbScores *= output_dims_scores.d[d];
    output_size_scores_bytes = nbScores * sizeof(float);
    CUDA_CHECK(cudaMalloc(&gpu_buffers[output_idx_scores],
                          output_size_scores_bytes));
}

//----------------------------------------------------------------------------//
// build(): parse ONNX → build engine → call init("") to bind & alloc
//----------------------------------------------------------------------------//

void RF_DETR::build(const std::string& onnx_path,
                    nvinfer1::ILogger& logger)
{
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
                       nvinfer1::createInferBuilder(logger));
    const auto explicitBatch =
      1U << static_cast<uint32_t>(
             nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
                       builder->createNetworkV2(explicitBatch));

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(
                      nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parse failed");

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
                      builder->createBuilderConfig());
    config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, size_t(1) << 30);
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(
                    builder->buildSerializedNetwork(*network, *config));

    runtime = TrtUniquePtr<nvinfer1::IRuntime>(
                  nvinfer1::createInferRuntime(logger));
    engine  = TrtUniquePtr<nvinfer1::ICudaEngine>(
                  runtime->deserializeCudaEngine(plan->data(), plan->size()));
    context = TrtUniquePtr<nvinfer1::IExecutionContext>(
                  engine->createExecutionContext());

    // reuse init logic for binding & allocation
    init("", logger);
}

//----------------------------------------------------------------------------//
// preprocess(): upload & resize / normalize on GPU
//----------------------------------------------------------------------------//

void RF_DETR::preprocess(const cv::Mat& image) {
    // one-time init flag, local static
    static bool preprocess_inited = false;

    // std::cout << "[RF_DETR::preprocess] ENTER\n";
    // std::cout << "  image.cols=" << image.cols
    //           << " image.rows=" << image.rows << "\n";
    // std::cout << "  target (C×H×W)="
    //           << input_c << "×" << input_h << "×" << input_w << "\n";

    if (!preprocess_inited) {
        size_t max_img_sz = size_t(image.cols) * image.rows;
        // std::cout << "[RF_DETR::preprocess] Initializing cuda_preprocess buffers for "
        //           << image.cols << "×" << image.rows << " => " << max_img_sz << " pixels\n";
        cuda_preprocess_init(max_img_sz);
        preprocess_inited = true;
    }

    if (gpu_buffers[input_idx] == nullptr)
        throw std::runtime_error("Null GPU input buffer");
    if (image.empty())
        throw std::runtime_error("Empty input image");

    cuda_preprocess(
      image.data,
      image.cols,   // src_width
      image.rows,   // src_height
      static_cast<float*>(gpu_buffers[input_idx]),
      input_w,      // dst_width
      input_h,      // dst_height
      stream);

    cudaStreamSynchronize(stream);
    // std::cout << "[RF_DETR::preprocess] KERNEL done\n";

    // probe a few floats
    size_t probe_count = std::min<size_t>(10, size_t(input_c)*input_h*input_w);
    std::vector<float> probe(probe_count);
    cudaMemcpy(probe.data(), gpu_buffers[input_idx],
               probe.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "[RF_DETR::preprocess] first " << probe.size()
    //           << " floats = ";
    // for (auto f : probe) std::cout << f << " ";
    // std::cout << "\n";
}

//----------------------------------------------------------------------------//
// infer(): bind, enqueue, download
//----------------------------------------------------------------------------//

bool RF_DETR::infer() {
    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        context->setTensorAddress(name, gpu_buffers[i]);
    }

    if (!context->enqueueV3(stream))
        return false;
    cudaStreamSynchronize(stream);

    CUDA_CHECK(cudaMemcpyAsync(
      cpu_output_buffer_1, gpu_buffers[output_idx_boxes],
      output_size_boxes_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
      cpu_output_buffer_2, gpu_buffers[output_idx_scores],
      output_size_scores_bytes, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    return true;
}

//----------------------------------------------------------------------------//
// postprocess(): threshold & scale boxes, sigmoid on scores
//----------------------------------------------------------------------------//

void RF_DETR::postprocess(std::vector<Detection>& output,
                          int originalW, int originalH,
                          float conf_thresh)
{
    // std::cout << "[DEBUG][postprocess] originalW=" << originalW
    //           << " originalH=" << originalH
    //           << " conf_thresh=" << conf_thresh << "\n";

    output.clear();
    int N = output_dims_boxes.d[1];
    // std::cout << "[DEBUG][postprocess] raw box count N=" << N << "\n";

    for (int i = 0; i < N; ++i) {
        float raw_logit = cpu_output_buffer_2[i];
        float score = 1.0f / (1.0f + std::exp(-raw_logit));
        // std::cout << "[DEBUG][postprocess] i=" << i
        //           << " raw_logit=" << raw_logit
        //           << " score=" << score;
        if (score < conf_thresh) {
            // std::cout << " → below threshold, skipping\n";
            continue;
        }
        float* b = cpu_output_buffer_1 + 4*i;
        float cx = b[0] * originalW, cy = b[1] * originalH;
        float bw = b[2] * originalW, bh = b[3] * originalH;
        int x1 = std::max(0, int(cx - bw/2));
        int y1 = std::max(0, int(cy - bh/2));
        int x2 = std::min(originalW - 1, int(cx + bw/2));
        int y2 = std::min(originalH - 1, int(cy + bh/2));
        // std::cout << "  → decoded box = [" 
        //           << x1 << "," << y1 << "," << x2 << "," << y2 << "]\n";
        output.push_back({ cv::Rect(x1,y1,x2-x1,y2-y1), score, 0 });
    }

    // std::cout << "[DEBUG][postprocess] kept " << output.size()
    //           << " boxes after thresholding\n";
}

//----------------------------------------------------------------------------//
// draw(): render boxes + scores, and overlay frame index
//----------------------------------------------------------------------------//

void RF_DETR::draw(cv::Mat& img,
                   const std::vector<Detection>& output,
                   int frame_idx,
                   float conf_thresh)
{
    // std::cout << "[DEBUG][draw] frame_idx=" << frame_idx
    //           << " dets=" << output.size()
    //           << " conf_thresh=" << conf_thresh << "\n";

    // draw each detection
    for (size_t i = 0; i < output.size(); ++i) {
        const auto& d = output[i];
        // std::cout << "[DEBUG][draw] det["<<i<<"] score="<<d.score
        //           << " bbox="<<d.bbox<<"\n";
        if (d.score < conf_thresh) {
            // std::cout << "  -> below threshold, skip draw\n";
            continue;
        }
        cv::rectangle(img, d.bbox, cv::Scalar(0,255,0), 2);
        std::string label = cv::format("%.2f", d.score);
        int baseline = 0;
        auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                  0.5, 1, &baseline);
        cv::rectangle(img,
            cv::Point(d.bbox.x, d.bbox.y - sz.height - baseline),
            cv::Point(d.bbox.x + sz.width, d.bbox.y),
            cv::Scalar(255,255,255), cv::FILLED);
        cv::putText(img, label,
            cv::Point(d.bbox.x, d.bbox.y - baseline),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }

    // overlay frame number
    std::string frame_label = "Frame: " + std::to_string(frame_idx);
    int font      = cv::FONT_HERSHEY_SIMPLEX;
    double scale  = 1.0;
    int thickness = 2;
    int baseline  = 0;
    auto ts = cv::getTextSize(frame_label, font, scale, thickness, &baseline);
    cv::Point org(img.cols - ts.width - 10, ts.height + 10);
    // std::cout << "[DEBUG][draw] frame label '"<<frame_label
    //           <<"' at "<< org << "\n";

    cv::rectangle(img,
        org + cv::Point(0, baseline),
        org + cv::Point(ts.width, -ts.height),
        cv::Scalar(0,0,0), cv::FILLED);
    cv::putText(img, frame_label, org, font, scale,
                cv::Scalar(255,255,255), thickness);

    // std::cout << "[DEBUG][draw] done drawing frame "<<frame_idx<<"\n";
}
