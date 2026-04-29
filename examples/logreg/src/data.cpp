#include "data.hpp"

#include <bit>
#include <fstream>
#include <iostream>
#include <sstream>

static std::string dataset_path(dataset_t dataset, exec_t exec) {
	std::string base = (dataset == RANDOM) ? "../data/random_data" : "../data/mnist_data";
	return base + ((exec == TRAIN) ? "_train.csv" : "_validation.csv");
}

static size_t dataset_label_index(dataset_t dataset) {
	return (dataset == RANDOM) ? 25 : 196;
}

static bool load_csv(const std::string& filename, std::vector<std::vector<std::string>>& data) {
	data.clear();
	std::ifstream file(filename);
	if (!file)
		return false;

	std::string line;
	while (std::getline(file, line)) {
		std::vector<std::string> row;
		std::stringstream ss(line);
		std::string cell;
		while (std::getline(ss, cell, ','))
			row.push_back(cell);
		data.push_back(row);
	}
	return true;
}

static std::pair<size_t, size_t>
parse_data(const std::vector<std::vector<std::string>>& raw_data, std::vector<std::vector<double>>& data, std::vector<double>& results, size_t result_idx) {
	data.clear();
	results.clear();
	for (const auto& raw_row : raw_data) {
		std::vector<double> row;
		for (size_t j = 0; j < raw_row.size(); ++j) {
			if (j == result_idx)
				results.push_back(std::stod(raw_row[j]));
			else
				row.push_back(std::stod(raw_row[j]));
		}
		data.push_back(row);
	}
	return { data[0].size(), data.size() };
}

size_t prepare_data_csv(dataset_t dataset, exec_t exec, std::vector<std::vector<double>>& data, std::vector<double>& results) {
	std::vector<std::vector<std::string>> raw;
	if (!load_csv(dataset_path(dataset, exec), raw)) {
		std::cerr << "Failed to load data" << std::endl;
		exit(EXIT_FAILURE);
	}
	auto [features, samples] = parse_data(raw, data, results, dataset_label_index(dataset));
	std::cout << "Loaded " << samples << " samples, " << features << " features" << std::endl;
	return features;
}

static std::tuple<size_t, size_t, size_t> pack_data(const std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& data_fhe, size_t num_slots) {

	const size_t num_samples	 = data.size();
	const size_t num_features	 = data[0].size();
	const size_t cols			 = std::bit_ceil(num_features);
	const size_t rows			 = num_slots / cols;
	const size_t num_ciphertexts = (num_samples + rows - 1) / rows;

	data_fhe.clear();
	data_fhe.resize(num_ciphertexts);

	size_t last_rows = 0;
	for (size_t ct = 0; ct < num_ciphertexts; ++ct) {
		data_fhe[ct].reserve(num_slots);
		last_rows = 0;
		for (size_t d = 0; d < rows; ++d) {
			size_t idx = ct * rows + d;
			if (idx >= num_samples) {
				data_fhe[ct].resize(num_slots, 0.0);
				break;
			}
			auto datum = data[idx];
			datum.resize(cols, 0.0);
			data_fhe[ct].insert(data_fhe[ct].end(), datum.begin(), datum.end());
			++last_rows;
		}
	}
	return { cols, rows, last_rows };
}

static std::tuple<size_t, size_t, size_t> pack_results(const std::vector<double>& results, std::vector<std::vector<double>>& results_fhe, size_t cols, size_t num_slots) {

	const size_t rows			 = num_slots / cols;
	const size_t num_ciphertexts = (results.size() + rows - 1) / rows;

	results_fhe.clear();
	results_fhe.resize(num_ciphertexts);

	size_t last_rows = 0;
	for (size_t ct = 0; ct < num_ciphertexts; ++ct) {
		results_fhe[ct].resize(num_slots, 0.0);
		last_rows = 0;
		for (size_t d = 0; d < rows; ++d) {
			size_t idx = ct * rows + d;
			if (idx >= results.size())
				break;
			results_fhe[ct][d * cols] = results[idx];
			++last_rows;
		}
	}
	return { cols, rows, last_rows };
}

static bool pack_weights(const std::vector<double>& weights, std::vector<double>& weights_fhe, size_t cols, size_t rows) {
	if (weights.size() > cols)
		return false;
	weights_fhe.assign(rows * cols, 0.0);
	for (size_t i = 0; i < weights.size(); ++i)
		for (size_t r = 0; r < rows; ++r)
			weights_fhe[r * cols + i] = weights[i];
	return true;
}

std::tuple<size_t, size_t, size_t> pack_data_fhe(const std::vector<std::vector<double>>& data,
  const std::vector<double>& results,
  const std::vector<double>& weights,
  std::vector<std::vector<double>>& data_fhe,
  std::vector<std::vector<double>>& results_fhe,
  std::vector<double>& weights_fhe,
  size_t num_slots) {

	auto [cols, rows, last] = pack_data(data, data_fhe, num_slots);
	pack_results(results, results_fhe, cols, num_slots);
	pack_weights(weights, weights_fhe, cols, rows);

	std::cout << "Packed: " << data_fhe.size() << " ciphertexts, " << cols << "x" << rows << " matrix" << std::endl;
	return { cols, rows, last };
}

void unpack_data(const std::vector<std::vector<double>>& data_fhe, std::vector<std::vector<double>>& data, size_t rows, size_t cols, size_t last_rows, size_t num_features) {
	data.clear();
	for (size_t m = 0; m < data_fhe.size(); ++m) {
		for (size_t r = 0; r < rows; ++r) {
			if (m == data_fhe.size() - 1 && r >= last_rows)
				break;
			std::vector<double> row(num_features);
			for (size_t c = 0; c < num_features; ++c)
				row[c] = data_fhe[m][r * cols + c];
			data.push_back(row);
		}
	}
}

bool unpack_weights(const std::vector<double>& weights_fhe, std::vector<double>& weights, size_t n) {
	weights.assign(weights_fhe.begin(), weights_fhe.begin() + n);
	return true;
}

void generate_weights(size_t n, std::vector<double>& weights) {
	weights.assign(n, 0.0);
}

void save_weights(const std::string& filename, const std::vector<double>& weights) {
	std::ofstream file(filename);
	for (size_t i = 0; i < weights.size(); ++i) {
		file << weights[i];
		if (i < weights.size() - 1)
			file << ",";
	}
}

void load_weights(const std::string& filename, size_t n, std::vector<double>& weights) {
	weights.clear();
	std::ifstream file(filename);
	std::string line, cell;
	std::getline(file, line);
	std::stringstream ss(line);
	while (std::getline(ss, cell, ','))
		weights.push_back(std::stod(cell));
}

void print_times(const std::vector<iteration_time_t>& times, const std::string& mode, bool gpu, size_t samples) {
	time_unit_t total{ 0 }, boot{ 0 };
	for (const auto& [t, b] : times) {
		total += t;
		boot += b;
	}
	std::cout << "[" << mode << "] " << (gpu ? "GPU" : "CPU") << " | Samples: " << samples << " | Total: " << total.count() / 1000.0 << "ms"
			  << " Boot_every=" << boot_every_iter << " Sparse_encaps=" << sparse_encaps << std::endl;
}