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
			break; // �ʿ�� ������ �ߴ�
		}
	}

	cudaEventRecord(inference_finish);
	cudaEventSynchronize(inference_finish);

	cudaEventElapsedTime(&inference_time, inference_start, inference_finish);

	std::cout << repeat << " inference : " << inference_time / repeat << " ms" << std::endl;
	cout << "Test Success!" << endl;


	std::string model_path_str = model_path; // C ���ڿ��� std::string���� ��ȯ
	std::filesystem::path full_path(model_path_str); // ��θ� std::filesystem::path�� ��ȯ

	// ��θ� ������ ä ���� �̸� ����
	std::string precision_suffix = fp16mode ? "_fp16" : "_fp32"; // fp16 or fp32 suffix ����
	std::filesystem::path parent_dir = full_path.parent_path(); // �θ� ���丮 ���
	std::string file_name = full_path.stem().string() + "_inference" + precision_suffix + ".txt"; // ���� ���� �̸� ����
	std::filesystem::path output_file = parent_dir / file_name; // ���ο� ���� ��� ����

	// ��� ����
	std::ofstream result_file(output_file.string()); // ������ ��η� ���� ����
	if (result_file.is_open()) {
		result_file << "Model Path: " << model_path << "\n";
		result_file << "Total Inference Time: " << inference_time << " ms\n"; // ���� ������
		result_file << "Average Inference Time: " << inference_time / repeat << " ms\n"; // ��հ�
		result_file.close(); // ���� �ݱ�
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

	const char* model_path = "C:\\Users\\IntekPlus\\Desktop\\vscode\\PyTorch-to-TensorRT-Block-Comparison\\onnx\\exp1.onnx";
	const int height = 256;
	const int width = 256;
	const int channel = 3;
	const int repeat = 1000;
	bool fp16mode = true;

	if (!run_inference(model_path, height, width, channel, repeat, fp16mode)) {
		std::cerr << "Error occurred during inference.\n";
		return -1;
	}

	std::cout << "Test Success!" << std::endl;

	return 0;
}