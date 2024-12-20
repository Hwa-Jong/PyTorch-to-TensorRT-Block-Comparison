#pragma once
#ifndef __trt_H__
#define __trt_H__

#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"


using namespace nvinfer1;
using namespace nvonnxparser;


class ModelTrt {
private:
    size_t input_size;
    size_t output_size;
    IBuilder* builder;
    uint32_t flag;
    INetworkDefinition* network;
    IParser* parser;
    IBuilderConfig* config;
    IHostMemory* serializedModel;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    cudaError_t cuRet;
    size_t insize, outsize;
    size_t datatype;
    Dims indim, outdim;
    void* devInPtr;
    void* devOutPtr;
    bool isSuccess, modelLoaded, canInference;

public:
    ModelTrt();
    ~ModelTrt();

    long get_model_sizeX();
    long get_model_sizeY();
    long get_model_channel();

    void load_onnx_model(const char* model_path, bool fp16mode);
    void mallocData();
    bool inference(std::vector<float> input, std::vector<float>* output);
    void save_trt_model(const char* save_path);
    void load_trt_model(const char* load_path);
    void load_model(const char* load_path, bool save_trt = true, bool fp16mode = true);
};

#endif