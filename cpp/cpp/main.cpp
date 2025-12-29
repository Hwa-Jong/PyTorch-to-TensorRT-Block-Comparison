#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "trt.h"

using namespace std;

bool run_inference(const char* model_path, int height, int width, int channel, int repeat, bool fp16mode) {

	std::vector<float> output;
	std::vector<float> img;

	bool success;

	float  inference_time = 0.0;
	cudaEvent_t inference_start, inference_finish;
	cudaEventCreate(&inference_start);
	cudaEventCreate(&inference_finish);

	ModelTrt model = ModelTrt();
	model.load_model(model_path, true, fp16mode);
	model.mallocData();

	img.resize(height*width*channel);

	cudaEventRecord(inference_start);

	for (int i = 0; i < repeat; i++) {
		success = model.inference(img, &output);
		if (!success) {
			std::cerr << "Inference failed at iteration " << i << std::endl;
			break;
		}
	}

	cudaEventRecord(inference_finish);
	cudaEventSynchronize(inference_finish);

	cudaEventElapsedTime(&inference_time, inference_start, inference_finish);

	std::cout << repeat << " inference : " << inference_time / repeat << " ms" << std::endl;
	cout << "Test Success!" << endl;


	// Save inference results to file
	std::string model_path_str = model_path;
	std::filesystem::path full_path(model_path_str);

	std::string precision_suffix = fp16mode ? "_fp16" : "_fp32";
	std::filesystem::path parent_dir = full_path.parent_path();
	std::string file_name = full_path.stem().string() + "_inference" + precision_suffix + ".txt";
	std::filesystem::path output_file = parent_dir / file_name;

	std::ofstream result_file(output_file.string());
	if (result_file.is_open()) {
		result_file << "Model Path: " << model_path << "\n";
		result_file << "Total Inference Time: " << inference_time << " ms\n";
		result_file << "Average Inference Time: " << inference_time / repeat << " ms\n";
		result_file.close();
		std::cout << "Inference time saved to '" << output_file.string() << "'" << std::endl;
	}
	else {
		std::cerr << "Failed to open file for writing." << std::endl;
	}

	cudaEventDestroy(inference_start);
	cudaEventDestroy(inference_finish);

	return true;
}

int main() {

	const int height = 64;
	const int width = 64;
	const int channel = 3;
	const int repeat = 1000;
	const bool fp16mode = false;

	// Set your ONNX models directory path
	std::filesystem::path onnx_dir = "SET_YOUR_PATH\\PyTorch-to-TensorRT-Block-Comparison\\onnx";

	// Model file names
	std::vector<std::string> model_names = {
		"exp1_vgg.onnx",
		"exp1_resnet.onnx",
		"exp1_mobilenet.onnx",
		"exp1_convnext.onnx",
		"exp2_vgg.onnx",
		"exp2_resnet.onnx",
		"exp2_mobilenet.onnx",
		"exp2_convnext.onnx",
		"exp3_vgg.onnx",
		"exp3_resnet.onnx",
		"exp3_mobilenet.onnx",
		"exp3_convnext.onnx",
		"exp4_vgg.onnx",
		"exp4_resnet.onnx",
		"exp4_mobilenet.onnx",
		"exp4_convnext.onnx",
	};

	for (const auto& model_name : model_names) {
		std::filesystem::path model_path = onnx_dir / model_name;
		std::string model_path_str = model_path.string();

		std::cout << "Running inference for model: " << model_path_str
			<< " with precision: " << (fp16mode ? "FP16" : "FP32") << "\n";

		if (!run_inference(model_path_str.c_str(), height, width, channel, repeat, fp16mode)) {
			std::cerr << "Error occurred during inference for model: " << model_path_str
				<< " with precision: " << (fp16mode ? "FP16" : "FP32") << "\n";
		}
		else {
			std::cout << "Inference completed for model: " << model_path_str
				<< " with precision: " << (fp16mode ? "FP16" : "FP32") << "\n";
		}
	}

	std::cout << "Test Success!" << std::endl;

	return 0;
}