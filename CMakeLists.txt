cmake_minimum_required(VERSION 3.18)
project(RF_DETR_DLL LANGUAGES CXX CUDA)

# --- EDIT THESE PATHS ---
# Set the exact paths to your OpenCV and TensorRT installations here.
# Use forward slashes / even on Windows.

set(MY_OpenCV_INCLUDE_DIRS "F:/Program Files (x86)/opencv-4.10.0-windows/opencv/build/include") # EDIT THIS PATH
set(MY_OpenCV_LIB_DIR "F:/Program Files (x86)/opencv-4.10.0-windows/opencv/build/x64/vc16/lib") # EDIT THIS PATH (Corrected Format)

set(MY_TensorRT_INCLUDE_DIRS "F:/Program Files/TensorRT-8.6.1.6/include")  # EDIT THIS PATH (Corrected Format)
set(MY_TensorRT_LIB_DIR      "F:/Program Files/TensorRT-8.6.1.6/lib")      # EDIT THIS PATH (e.g., "C:/TensorRT-8.x.x.x/lib")

# --- Verify Library Names ---
# Check these library names against the actual files in your LIB_DIRs.
# Remove the 'lib' prefix and file extensions (.so, .lib, .dll). Adjust if needed.
set(MY_OpenCV_LIBS opencv_world4100d)
set(MY_TensorRT_LIBS nvinfer nvparsers nvonnxparser)                           # EDIT THESE NAMES if needed

# --- End Editable Paths ---


# --- Find CUDA ---
# Keep finding CUDA automatically as it's usually well-integrated.
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Using CUDA includes: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Using CUDA libraries: ${CUDA_LIBRARIES}")


# --- Verify Provided Paths ---
if(NOT EXISTS "${MY_OpenCV_INCLUDE_DIRS}")
    message(FATAL_ERROR "OpenCV Include directory not found: ${MY_OpenCV_INCLUDE_DIRS} - Please edit CMakeLists.txt")
endif()
if(NOT EXISTS "${MY_OpenCV_LIB_DIR}")
    message(FATAL_ERROR "OpenCV Library directory not found: ${MY_OpenCV_LIB_DIR} - Please edit CMakeLists.txt")
endif()
if(NOT EXISTS "${MY_TensorRT_INCLUDE_DIRS}")
    message(FATAL_ERROR "TensorRT Include directory not found: ${MY_TensorRT_INCLUDE_DIRS} - Please edit CMakeLists.txt")
endif()
if(NOT EXISTS "${MY_TensorRT_LIB_DIR}")
    message(FATAL_ERROR "TensorRT Library directory not found: ${MY_TensorRT_LIB_DIR} - Please edit CMakeLists.txt")
endif()
message(STATUS "Using OpenCV include: ${MY_OpenCV_INCLUDE_DIRS}")
message(STATUS "Using OpenCV lib dir: ${MY_OpenCV_LIB_DIR}")
message(STATUS "Using TensorRT include: ${MY_TensorRT_INCLUDE_DIRS}")
message(STATUS "Using TensorRT lib dir: ${MY_TensorRT_LIB_DIR}")


# --- Library Definition ---
add_library(RF_DETR SHARED
    src/RF_DETR.cpp
    src/preprocess.cu
)
target_compile_definitions(RF_DETR PRIVATE API_EXPORTS)


# --- Include Directories ---
# Add the manually set include directories
target_include_directories(RF_DETR PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MY_OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}        # From find_package(CUDA)
    ${MY_TensorRT_INCLUDE_DIRS}
)


# --- Link Dependencies ---
# Specify the directories where the linker should look for libraries
target_link_directories(RF_DETR PUBLIC
    ${MY_OpenCV_LIB_DIR}
    # CUDA libraries are usually handled by ${CUDA_LIBRARIES} which includes paths
    ${MY_TensorRT_LIB_DIR}
)

# Link the specific libraries by name (without paths)
target_link_libraries(RF_DETR PUBLIC
    ${MY_OpenCV_LIBS}           # Manually listed OpenCV lib names
    ${CUDA_LIBRARIES}           # From find_package(CUDA)
    ${CUDA_CUBLAS_LIBRARIES}    # Add other CUDA libs as needed
    ${MY_TensorRT_LIBS}         # Manually listed TensorRT lib names
)


# --- CUDA Specific Settings ---
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86) # Example architectures


# --- Output Directory (Optional) ---
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


message(STATUS "Configuration complete for RF_DETR DLL using hardcoded paths.")