cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

SET (CMAKE_BUILD_TYPE Debug CACHE INTERNAL “FORCE”)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(/home/lostye/TensorRT-8.2.3.0/include)
link_directories(/home/lostye/TensorRT-8.2.3.0/lib)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/opt/DahuaTech/MVviewer/include)


cuda_add_executable(yolov5 calibrator.cpp yolov5.cpp preprocess.cu yolo/yolo_go.cpp DAHUA/dhua.cpp DAHUA/streamRetrieve.cpp serial/serial.cpp)

target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} "/opt/DahuaTech/MVviewer/lib/libMVSDK.so")


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


