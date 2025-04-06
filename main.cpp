#include "RF_DETR.h"
#include "logging.h" // Include logging header [cite: 300]
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem> // For directory iteration (C++17)

// Helper function to list image files in a directory
std::vector<std::string> list_image_files(const std::string& path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }
    return files;
}


int main(int argc, char** argv) {
    // --- Argument Parsing ---
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path.engine/.onnx> <image_folder | video_path> [output_video_path]" << std::endl;
        return -1;
    }
    std::string modelPath = argv[1];
    std::string inputPath = argv[2];
    std::string outputPath = (argc > 3) ? argv[3] : "";

    // --- Initialization ---
    Logger gLogger(nvinfer1::ILogger::Severity::kINFO); // Or kWARNING, kERROR etc. [cite: 337]
    RF_DETR model(modelPath, gLogger.getTRTLogger()); // Pass the logger instance [cite: 341]

    std::filesystem::path path_obj(inputPath);
    bool is_video = false;
    bool is_directory = false;

    if (std::filesystem::is_directory(path_obj)) {
         is_directory = true;
         std::cout << "Processing images from directory: " << inputPath << std::endl;
    } else if (std::filesystem::is_regular_file(path_obj)) {
        std::string ext = path_obj.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") { // Add other video extensions if needed
             is_video = true;
             std::cout << "Processing video file: " << inputPath << std::endl;
        } else {
             std::cerr << "Input path is a file, but not a recognized video format. Treating as single image." << std::endl;
             // Treat as single image path list
             is_directory = true; // Use directory logic but with one file
             inputPath = path_obj.parent_path().string(); // Use parent dir for consistency if needed later
        }
    } else {
        std::cerr << "Error: Input path is neither a valid directory nor a recognized file." << std::endl;
        return -1;
    }


    // --- Processing Logic ---
    if (is_video) {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file: " << inputPath << std::endl;
            return -1;
        }

        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        cv::VideoWriter videoWriter;
        if (!outputPath.empty()) {
            videoWriter.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));
            if (!videoWriter.isOpened()) {
                std::cerr << "Error opening output video file: " << outputPath << std::endl;
                // Continue without writing if output fails to open
            } else {
                 std::cout << "Writing output video to: " << outputPath << std::endl;
            }
        }


        cv::Mat frame;
        std::vector<Detection> objects;
        int frameCount = 0;

        while (cap.read(frame)) {
            frameCount++;
            if (frame.empty()) {
                std::cerr << "Warning: Skipped empty frame " << frameCount << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // --- Inference ---
            model.preprocess(frame);
            model.infer();
            model.postprocess(objects, frameWidth, frameHeight); // Pass original dimensions
            // ---------------

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Frame " << frameCount << ": Found " << objects.size() << " objects in " << duration.count() << " ms" << std::endl;

            // --- Drawing ---
            model.draw(frame, objects); // Draw on the current frame
            // -------------

            // --- Display/Write ---
            cv::imshow("RF-DETR Detection", frame);
             if (videoWriter.isOpened()) {
                 videoWriter.write(frame);
             }

            // Exit on ESC key
            if (cv::waitKey(1) == 27) {
                std::cout << "ESC key pressed. Exiting." << std::endl;
                break;
            }
        }

        // --- Cleanup Video ---
        cap.release();
         if (videoWriter.isOpened()) {
             videoWriter.release();
         }
        cv::destroyAllWindows();

    } else if (is_directory) {
        std::vector<std::string> imagePathList;
        // If it was treated as a single image file initially
        if (!is_video && !std::filesystem::is_directory(path_obj)) {
             imagePathList.push_back(path_obj.string());
        } else {
            imagePathList = list_image_files(inputPath);
        }

        if (imagePathList.empty()) {
            std::cerr << "No image files found in directory: " << inputPath << std::endl;
            return -1;
        }


        for (const auto& imagePath : imagePathList) {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Error reading image: " << imagePath << std::endl;
                continue;
            }

            int imageWidth = image.cols;
            int imageHeight = image.rows;
            std::vector<Detection> objects;

            auto start = std::chrono::high_resolution_clock::now();

            // --- Inference ---
            model.preprocess(image);
            model.infer();
            model.postprocess(objects, imageWidth, imageHeight); // Pass original dimensions
            // ---------------

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

             std::cout << "Image '" << imagePath << "': Found " << objects.size() << " objects in " << duration.count() << " ms" << std::endl;

            // --- Drawing ---
            model.draw(image, objects); // Draw on the image
            // -------------

            // --- Display/Save ---
             cv::imshow("RF-DETR Detection", image);

             if (!outputPath.empty()) {
                 // Construct output path if saving is desired (e.g., save to a subfolder)
                 std::filesystem::path inputFilePath(imagePath);
                 std::filesystem::path outputDirPath(outputPath);
                 // Ensure output directory exists (optional, create if needed)
                 // std::filesystem::create_directories(outputDirPath);
                 std::string outputImagePath = (outputDirPath / inputFilePath.filename()).string();
                  if (cv::imwrite(outputImagePath, image)) {
                       std::cout << "Saved result to: " << outputImagePath << std::endl;
                  } else {
                      std::cerr << "Error saving result image: " << outputImagePath << std::endl;
                  }
             }


             // Wait for a key press (or remove for batch processing)
             int key = cv::waitKey(0);
             if (key == 27) { // Exit on ESC
                 std::cout << "ESC key pressed. Exiting." << std::endl;
                 break;
             }
        }
         cv::destroyAllWindows();
    }


    std::cout << "Processing finished." << std::endl;
    return 0;
}