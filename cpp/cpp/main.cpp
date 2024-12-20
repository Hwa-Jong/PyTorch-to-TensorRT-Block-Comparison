#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "trt.h"

using namespace std;

int main() {

	std::vector<float> output;
	std::vector<float> img;

	int height = 256;
	int width = 256;
	int channel = 3;

	bool success;
	int repeat = 1000;

	float  inference_time = 0.0;
	cudaEvent_t inference_start, inference_finish;
	cudaEventCreate(&inference_start);
	cudaEventCreate(&inference_finish);

	const char* model_path = "C:\\Users\\IntekPlus\\Desktop\\vscode\\PyTorch-to-TensorRT-Block-Comparison\\onnx\\exp1.json.onnx";

	ModelTrt model = ModelTrt();
	model.load_model(model_path, true);
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
	std::filesystem::path parent_dir = full_path.parent_path(); // �θ� ���丮 ���
	std::string file_name = full_path.stem().string() + "_inference.txt"; // ���ο� ���� �̸�
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

	return 0;
}