#pragma once

#include <NvInferRuntimeBase.h> // Use NvInferRuntimeBase.h for ILogger definition
#include <iostream>
#include <mutex> // Include mutex for thread safety

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
private:
    Severity reportableSeverity;
    std::mutex log_mutex; // Add a mutex for thread-safe logging

public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity) {}

    // The virtual destructor is important for classes inheriting from interfaces
    virtual ~Logger() = default;

    // The logging function override
    // ADD noexcept HERE
    void log(Severity severity, const char* msg) noexcept override
    {
        // Suppress messages with severity lower than the reportable level.
        if (severity > reportableSeverity)
            return;

        // Use a lock guard for thread safety
        std::lock_guard<std::mutex> lock(log_mutex);

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR:   std::cerr << "ERROR: ";   break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO:    std::cerr << "INFO: ";    break;
        case Severity::kVERBOSE: std::cerr << "VERBOSE: "; break;
        default:                 std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    // Getter for the internal TRT logger (optional, if needed elsewhere)
    // nvinfer1::ILogger& getTRTLogger() { return *this; } // No longer needed with direct passing
};

// Example global logger instance (consider dependency injection instead for larger projects)
// static Logger gLogger(nvinfer1::ILogger::Severity::kINFO); // Move instance creation to where it's used