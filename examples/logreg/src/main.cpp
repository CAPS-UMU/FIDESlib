#include "data.hpp"
#include "fhe.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

std::vector<int> devices		  = {};
bool prescale					  = true;
bool sparse_encaps				  = true;
bool boot_every_iter			  = false;
std::vector<uint32_t> bStep		  = { 16, 16 };
std::vector<uint32_t> levelBudget = { 2, 2 };
uint32_t numSlots				  = 1 << 15;
uint32_t ringDim				  = 1 << 16;

// Read ring dim from env var if set
void read_ring_dim() {
	char* env = getenv("FIDESLIB_RING_DIM");
	if (env && env[0] != '\0') {
		ringDim = 1 << std::atoi(env);
	}
}

static void print_usage(const char* name) {
	std::cerr << "Usage:\n"
			  << "  " << name << " train <dataset> <iterations> <sparse> <boot_every_iter>\n"
			  << "  " << name << " inference <dataset> <sparse> <boot_every_iter>\n"
			  << "  " << name << " perf <dataset> <iterations> <sparse> <boot_every_iter>\n"
			  << "\nDatasets: random, mnist\n"
			  << "Sparse: 0 = UNIFORM_TERNARY, 1 = SPARSE_ENCAPSULATED\n"
			  << "Boot every iter: 0 = false, 1 = true\n"
			  << "Set FIDESLIB_USE_NUM_GPUS env var for number of GPUs (default: 0)\n";
	exit(EXIT_FAILURE);
}

static dataset_t parse_dataset(const std::string& s) {
	if (s == "random")
		return RANDOM;
	if (s == "mnist")
		return MNIST;
	std::cerr << "Unknown dataset: " << s << std::endl;
	exit(EXIT_FAILURE);
}

static void setup_devices() {
	devices.clear();
	int count = 0;
	char* env = getenv("FIDESLIB_USE_NUM_GPUS");
	if (env && env[0] != '\0') {
		count = std::atoi(env);
	}
	for (int i = 0; i < count; ++i)
		devices.push_back(i);
}

int main(int argc, char* argv[]) {
	if (argc < 4)
		print_usage(argv[0]);

	std::string mode  = argv[1];
	dataset_t dataset = parse_dataset(argv[2]);
	setup_devices();

	read_ring_dim();
	std::cout << "Using ring dimension: " << ringDim << std::endl;

	if (mode == "train") {
		if (argc != 6)
			print_usage(argv[0]);
		size_t iterations = std::stoul(argv[3]);
		sparse_encaps = std::stoi(argv[4]) != 0;
		boot_every_iter = std::stoi(argv[5]) != 0;

		std::vector<std::vector<double>> data;
		std::vector<double> results, weights;
		size_t features = prepare_data_csv(dataset, TRAIN, data, results);
		generate_weights(features, weights);

		auto times = fideslib_training(data, results, weights, iterations);
		print_times(times, "TRAIN", !devices.empty(), data.size());

	} else if (mode == "inference") {
		if (argc != 5)
			print_usage(argv[0]);
		sparse_encaps = std::stoi(argv[3]) != 0;
		boot_every_iter = std::stoi(argv[4]) != 0;

		std::vector<std::vector<double>> data;
		std::vector<double> results, weights;
		size_t features = prepare_data_csv(dataset, VALIDATION, data, results);
		load_weights("../weights/weights.csv", features, weights);

		auto [times, accuracy] = fideslib_inference(data, results, weights);
		print_times(times, "INFERENCE", !devices.empty(), data.size());
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	} else if (mode == "perf") {
		if (argc != 6)
			print_usage(argv[0]);
		size_t iterations = std::stoul(argv[3]);
		sparse_encaps = std::stoi(argv[4]) != 0;
		boot_every_iter = std::stoi(argv[5]) != 0;

		std::vector<std::vector<double>> train_data, val_data;
		std::vector<double> train_results, val_results;
		prepare_data_csv(dataset, TRAIN, train_data, train_results);
		size_t features = prepare_data_csv(dataset, VALIDATION, val_data, val_results);

		std::vector<double> weights;
		generate_weights(features, weights);

		auto train_times = fideslib_training(train_data, train_results, weights, iterations);
		print_times(train_times, "TRAIN", !devices.empty(), train_data.size());

		auto [val_times, accuracy] = fideslib_inference(val_data, val_results, weights);
		print_times(val_times, "INFERENCE", !devices.empty(), val_data.size());
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	} else {
		print_usage(argv[0]);
	}

	return EXIT_SUCCESS;
}