cmake_minimum_required(VERSION 3.18)
project(RF_DETR_DLL LANGUAGES CXX CUDA)


# require at least C++17 for std::filesystem
set(CMAKE_CXX_STANDARD       17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)

# --- EDIT THESE PATHS ---
set(MY_OpenCV_INCLUDE_DIRS   "/usr/local/include/opencv4")
set(MY_OpenCV_LIB_DIR        "/usr/local/lib")

set(MY_TensorRT_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu")
set(MY_TensorRT_LIB_DIR      "/usr/lib/x86_64-linux-gnu")

set(MY_OpenCV_LIBS opencv_world)
set(MY_TensorRT_LIBS nvinfer nvparsers nvonnxparser)
# --- End Editable Paths ---

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Using CUDA includes: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Using CUDA libraries: ${CUDA_LIBRARIES}")

# sanity checks
if(NOT EXISTS "${MY_OpenCV_INCLUDE_DIRS}")
  message(FATAL_ERROR "OpenCV include dir not found: ${MY_OpenCV_INCLUDE_DIRS}")
endif()
if(NOT EXISTS "${MY_OpenCV_LIB_DIR}")
  message(FATAL_ERROR "OpenCV lib dir not found:     ${MY_OpenCV_LIB_DIR}")
endif()
if(NOT EXISTS "${MY_TensorRT_INCLUDE_DIRS}")
  message(FATAL_ERROR "TensorRT include dir not found: ${MY_TensorRT_INCLUDE_DIRS}")
endif()
if(NOT EXISTS "${MY_TensorRT_LIB_DIR}")
  message(FATAL_ERROR "TensorRT lib dir not found:     ${MY_TensorRT_LIB_DIR}")
endif()

message(STATUS "OpenCV include: ${MY_OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV  lib dir: ${MY_OpenCV_LIB_DIR}")
message(STATUS "TensorRT include: ${MY_TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT  lib dir: ${MY_TensorRT_LIB_DIR}")

# --- RF_DETR shared library ---
add_library(RF_DETR SHARED
  src/RF_DETR.cpp
  src/preprocess.cu
)
target_compile_definitions(RF_DETR PRIVATE API_EXPORTS)
target_include_directories(RF_DETR PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MY_OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${MY_TensorRT_INCLUDE_DIRS}
)
target_link_directories(RF_DETR PUBLIC
  ${MY_OpenCV_LIB_DIR}
  ${MY_TensorRT_LIB_DIR}
)
target_link_libraries(RF_DETR PUBLIC
  ${MY_OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${MY_TensorRT_LIBS}
)
target_compile_options(RF_DETR PRIVATE
  -Wno-deprecated-declarations
)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- rf_detr_app executable ---
add_executable(rf_detr_app
  main.cpp
)
# so that app can #include "RF_DETR.h"
target_include_directories(rf_detr_app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# link against your shared lib (and transitively everything RF_DETR needs)
target_link_libraries(rf_detr_app PRIVATE
  RF_DETR
  Threads::Threads         # if you need pthread
  stdc++fs                 # for std::filesystem
)
# suppress the same deprecation warnings in the app
target_compile_options(rf_detr_app PRIVATE
  -Wno-deprecated-declarations
)

# (Optional) place binaries in bin/, libraries in lib/
set_target_properties(RF_DETR PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
set_target_properties(rf_detr_app PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Config complete — RF_DETR library and rf_detr_app executable ready.")
