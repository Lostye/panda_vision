#ifndef YOLO_GO_H
#define YOLO_GO_H
#define YOLO_SHOW

#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "../cuda_utils.h"
#include "../logging.h"
#include "../common.hpp"
#include "../utils.h"
#include "../calibrator.h"
#include "../object.h"
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 8
using namespace cv;
using namespace std;



class yolo_Go
{
    private:

    string engine_name = "panda.engine";
    IExecutionContext* context;
    cudaStream_t stream;
    void* buffers[2];

    ICudaEngine* engine;
    IRuntime* runtime;
    int input;
    int output;

    const char *my_classes[3]={ "none", "green", "red"};
    
    public:
    bool flag;
    int fps;
    yolo_Go();
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    bool parse_args(int argc, char** argv, string& engine);
    bool ready();
    vector<object> go(Mat frame,int &fcount);
    void release();
};

#endif // YOLO_GO_H
