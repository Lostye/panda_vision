#include "yolo_go.h"

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
static float prob[BATCH_SIZE * OUTPUT_SIZE];



void yolo_Go:: doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool yolo_Go:: parse_args(int argc, char** argv, std::string& engine) {
    if (argc < 3) return false;
    if (std::string(argv[1]) == "-v" && argc == 3) {
        engine = std::string(argv[2]);
    } else {
        return false;
    }
    return true;
}

bool yolo_Go::ready(){
    if(!flag)
        return -1;
    return true;
}

yolo_Go::yolo_Go()
{
    cudaSetDevice(DEVICE);

    //std::string wts_name = "";
    //float gd = 0.0f, gw = 0.0f;
    //std::string img_dir;

//	if(!parse_args(argc,argv,engine_name)){
//		std::cerr << "arguments not right!" << std::endl;
//        	std::cerr << "./yolov5 -v [.engine] // run inference with camera" << std::endl;
//		return -1;
//	}

    ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
        cerr<<" read "<<engine_name<<" error! "<<std::endl;
        flag=false;
    }
    char *trtModelStream{ nullptr };
    size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();


    // prepare input data ---------------------------
//    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
//    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
//    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    input=inputIndex;
    output=outputIndex;
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
//    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
}

vector<object> yolo_Go::go(Mat frame ,int &fcount){
    vector<object> obj;

    fcount++;

    //if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
    for (int b = 0; b < fcount; b++) {
        //cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
        Mat img = frame;
        if (img.empty()) continue;
        Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    // Run inference
    auto start = chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);

    auto end = chrono::system_clock::now();
    cout << "inference time: "<<chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    fps = 1000.0/chrono::duration_cast<chrono::milliseconds>(end - start).count();
    vector<vector<Yolo::Detection>> batch_res(fcount);

    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    }
    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        //std::cout << res.size() << std::endl;
        //cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
        for (size_t j = 0; j < res.size(); j++) {
            object cache;
            cache.rect = get_rect(frame, res[j].bbox);;
            cache.label = my_classes[(int)res[j].class_id];
            obj.push_back(cache);
#ifdef YOLO_SHOW
            rectangle(frame, cache.rect, Scalar(0x27, 0xC1, 0x36), 2);
            putText(frame, cache.label, Point2f(cache.rect.x, cache.rect.y-1), FONT_HERSHEY_PLAIN, 1.2, Scalar(0xFF, 0xFF, 0xFF), 2);
//            cout<<cache.label<<endl;
            //add FPS in the windows
            string yolo_fps = "FPS: " + std::to_string(fps);
            putText(frame, yolo_fps, Point(11,80), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 2, LINE_AA);
#endif
        }
        //imwrite("_" + file_names[f - fcount + 1 + b], img);
    }
#ifdef YOLO_SHOW
    imshow("yolov5",frame);
#endif
    return obj;
}

void yolo_Go::release(){
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[input]));
    CUDA_CHECK(cudaFree(buffers[output]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}
