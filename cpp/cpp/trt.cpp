#include <iostream>
#include <string>
#include <time.h>
#include <fstream>
#include "trt.h"


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;



ModelTrt::ModelTrt() {
    input_size = 0;
    output_size = 0;
    builder = NULL;
    flag = NULL;
    network = NULL;
    parser = NULL;
    config = NULL;
    serializedModel = NULL;
    runtime = NULL;
    engine = NULL;
    context = NULL;
    stream = NULL;
    cuRet = cudaSuccess(0);
    insize = 1;
    outsize = 1;
    datatype = sizeof(float);
    indim = nvinfer1::Dims32();
    outdim = nvinfer1::Dims32();
    devInPtr = NULL;
    devOutPtr = NULL;
    isSuccess = false;
    modelLoaded = false;
    canInference = false;
}

ModelTrt::~ModelTrt() {
    cudaFree(devOutPtr);
    cudaFree(devInPtr);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    delete serializedModel;
    delete config;
    delete parser;
    delete network;
    delete builder;
}

void ModelTrt::load_trt_model(const char* load_path) {
    std::ifstream file(load_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << load_path << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    file.read(buffer, size);
    file.close();

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(buffer, size);
    delete[] buffer;
    context = engine->createExecutionContext();
    modelLoaded = true;
}

void ModelTrt::save_trt_model(const char* save_path) {
    std::ofstream file(save_path, std::ios::binary);
    file.write((char*)serializedModel->data(), serializedModel->size());
    file.close();

    std::cout << "save trt model." << std::endl;

}

long ModelTrt::get_model_sizeX() {
    if (modelLoaded) {
        return indim.d[3];
    }
    else {
        fprintf(stderr, "Model not loaded!\n");
    }
}
long ModelTrt::get_model_sizeY() {
    if (modelLoaded) {
        return indim.d[2];
    }
    else {
        fprintf(stderr, "Model not loaded!\n");
    }
}
long ModelTrt::get_model_channel() {
    if (modelLoaded) {
        return indim.d[1];
    }
    else {
        fprintf(stderr, "Model not loaded!\n");
    }
}

void ModelTrt::load_onnx_model(const char* model_path, bool fp16mode) {
    builder = createInferBuilder(logger);
    flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = builder->createNetworkV2(flag);
    parser = createParser(*network, logger);

    parser->parseFromFile(model_path, static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1U << 30);
    if (fp16mode){
        if (builder->platformHasFastFp16())
        {
            std::cout << "fp16 mode" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }
    serializedModel = builder->buildSerializedNetwork(*network, *config);
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    context = engine->createExecutionContext();
    modelLoaded = true;
}

void ModelTrt::mallocData() {
    cudaStreamCreate(&stream);
    indim = engine->getBindingDimensions(engine->getBindingIndex("input_1"));
    for (int i = 0; i < indim.nbDims; i++) {
        insize = insize * indim.d[i];
    }
    insize = insize * datatype;

    outdim = engine->getBindingDimensions(engine->getBindingIndex("output_1"));
    for (int i = 0; i < outdim.nbDims; i++) {
        outsize = outsize * outdim.d[i];
    }
    outsize = outsize * datatype;

    cuRet = cudaMalloc(&devInPtr, insize);
    if (cuRet) {
        fprintf(stderr, "Cuda malloc input failed\n");
        return;
    }
    cuRet = cudaMalloc(&devOutPtr, outsize);
    if (cuRet) {
        fprintf(stderr, "Cuda malloc output failed\n");
        return;
    }

    canInference = true;
}

bool ModelTrt::inference(std::vector<float> input, std::vector<float>* output) {
    if (!modelLoaded) {
        std::cout << "model is not loaded." << std::endl;
        return false;
    }
    if (!canInference) { mallocData(); }

    (*output).resize(outsize / sizeof(float));
    cuRet = cudaMemcpy(devInPtr, input.data(), insize, cudaMemcpyHostToDevice);
    if (cuRet) {
        fprintf(stderr, "CudaMemcpyHostToDevice memory input copy failed\n");
        return false;
    }
    cuRet = cudaMemcpy(devOutPtr, (*output).data(), outsize, cudaMemcpyHostToDevice);
    if (cuRet) {
        fprintf(stderr, "CudaMemcpyHostToDevice memory output copy failed\n");
        return false;
    }


    context->enqueueV2(&devInPtr, stream, nullptr);

    //(context)->enqueueV3(stream);

    cuRet = cudaMemcpy((*output).data(), devOutPtr, outsize, cudaMemcpyDeviceToHost);
    if (cuRet) {
        fprintf(stderr, "cudaMemcpyDeviceToHost memory result copy failed\n");
        return false;
    }
    return true;
}

void ModelTrt::load_model(const char* load_path, bool save_trt, bool fp16mode) {
    std::string name = load_path;

    if (name.substr(name.find_last_of(".") + 1) == "onnx") {
        load_onnx_model(load_path, fp16mode);
        if (save_trt) {
            name.replace(name.length() - 4, 4, "trt");
            save_trt_model(name.data());
        }
    }
    else if (name.substr(name.find_last_of(".") + 1) == "trt") {
        load_trt_model(load_path);
    }
    else {
        std::cout << "File extension is not trt or onnx." << std::endl;
    }
}
