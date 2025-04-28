#include "RF_DETR.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <cstdlib>  // for std::getenv

namespace fs = std::filesystem;

std::vector<std::string> list_image_files(const std::string& path) {
    std::cout << "[DEBUG] Listing images in: " << path << "\n";
    std::vector<std::string> files;
    try {
        for (const auto& e : fs::directory_iterator(path)) {
            if (e.is_regular_file()) {
                auto ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".png" || ext == ".bmp") {
                    files.push_back(e.path().string());
                    std::cout << "[DEBUG]  Found image: " << e.path().string() << "\n";
                }
            }
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "[ERROR] Directory iteration failed: " << ex.what() << "\n";
    }
    std::cout << "[DEBUG] Total images found: " << files.size() << "\n";
    return files;
}

int main(int argc, char** argv) {
    std::cout << "[DEBUG] Starting application, argc=" << argc << "\n";
    if (argc < 3) {
        std::cerr << "[ERROR] Usage: " << argv[0]
                  << " <model.onnx|.engine> <image_dir|video.mp4> [out.mp4]\n";
        return 1;
    }
    std::string modelPath  = argv[1];
    std::string inputPath  = argv[2];
    std::string outputPath = (argc > 3 ? argv[3] : "");
    std::cout << "[DEBUG] modelPath  = " << modelPath  << "\n";
    std::cout << "[DEBUG] inputPath  = " << inputPath
              << ", outputPath = " << (outputPath.empty() ? "[none]" : outputPath)
              << "\n";

    Logger gLogger(nvinfer1::ILogger::Severity::kINFO);
    std::cout << "[DEBUG] Creating RF_DETR instance\n";
    RF_DETR model(modelPath, gLogger);
    std::cout << "[DEBUG] Model initialized successfully\n";

    bool isVideo   = fs::path(inputPath).extension() == ".mp4";
    bool saveVideo = !outputPath.empty();
    bool headless  = (std::getenv("DISPLAY") == nullptr);

    cv::VideoWriter writer;
    cv::VideoCapture cap;
    std::vector<std::string> images;
    int W=0, H=0; double fps=0;

    if (isVideo) {
        cap.open(inputPath);
        if (!cap.isOpened()) return 1;
        W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        fps = cap.get(cv::CAP_PROP_FPS);
        if (saveVideo) {
            writer.open(outputPath,
                        cv::VideoWriter::fourcc('m','p','4','v'),
                        fps, cv::Size(W,H));
            if (!writer.isOpened()) saveVideo = false;
        }
    } else {
        images = list_image_files(inputPath);
        if (images.empty()) return 1;
    }

    auto global_t0 = std::chrono::high_resolution_clock::now();
    cv::Mat frame;
    std::vector<Detection> dets;
    int idx = 0;

    while (true) {
        if (isVideo) {
            if (!cap.read(frame)) break;
            ++idx;
        } else {
            if (idx >= (int)images.size()) break;
            frame = cv::imread(images[idx]); ++idx;
        }
        if (frame.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();
        model.preprocess(frame);
        if (!model.infer()) break;
        model.postprocess(dets, frame.cols, frame.rows, 0.5f);
        model.draw(frame, dets, idx, 0.5f);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // std::cout << (isVideo?"Frame ":"Image ")
        //           << idx << ": " << dets.size()
        //           << " objects, inference time: " << ms << " ms\n";

        if (isVideo && saveVideo) writer.write(frame);
        if (!headless) {
            cv::imshow("RF-DETR Detection", frame);
            if (cv::waitKey(isVideo?1:0)==27) break;
        }
    }

    auto global_t1   = std::chrono::high_resolution_clock::now();
    double total_ms  = std::chrono::duration<double, std::milli>(global_t1 - global_t0).count();
    int    processed = idx;
    std::cout << "\n[SUMMARY] Processed " << processed
              << " frames in " << (total_ms/1000.0)
              << "s (" << (processed/(total_ms/1000.0))
              << " FPS)\n";

    if (isVideo) cap.release();
    if (saveVideo) writer.release();
    if (!headless) cv::destroyAllWindows();
    return 0;
}
