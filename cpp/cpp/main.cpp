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
			break; // 필요시 루프를 중단
		}
	}

	cudaEventRecord(inference_finish);
	cudaEventSynchronize(inference_finish);

	cudaEventElapsedTime(&inference_time, inference_start, inference_finish);

	std::cout << repeat << " inference : " << inference_time / repeat << " ms" << std::endl;
	cout << "Test Success!" << endl;


	std::string model_path_str = model_path; // C 문자열을 std::string으로 변환
	std::filesystem::path full_path(model_path_str); // 경로를 std::filesystem::path로 변환

	// 경로를 유지한 채 파일 이름 변경
	std::string precision_suffix = fp16mode ? "_fp16" : "_fp32"; // fp16 or fp32 suffix 결정
	std::filesystem::path parent_dir = full_path.parent_path(); // 부모 디렉토리 경로
	std::string file_name = full_path.stem().string() + "_inference" + precision_suffix + ".txt"; // 최종 파일 이름 생성
	std::filesystem::path output_file = parent_dir / file_name; // 새로운 파일 경로 생성

	// 결과 저장
	std::ofstream result_file(output_file.string()); // 생성된 경로로 파일 열기
	if (result_file.is_open()) {
		result_file << "Model Path: " << model_path << "\n";
		result_file << "Total Inference Time: " << inference_time << " ms\n"; // 실제 측정값
		result_file << "Average Inference Time: " << inference_time / repeat << " ms\n"; // 평균값
		result_file.close(); // 파일 닫기
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