#ifndef DATA_HPP
#define DATA_HPP

#include <chrono>
#include <string>
#include <tuple>
#include <vector>

enum dataset_t { RANDOM, MNIST };

extern bool sparse_encaps;
extern bool boot_every_iter;

enum exec_t { TRAIN, VALIDATION, PERFORMANCE };

using time_unit_t	   = std::chrono::microseconds;
using iteration_time_t = std::pair<time_unit_t, time_unit_t>;

size_t prepare_data_csv(dataset_t dataset, exec_t exec, std::vector<std::vector<double>>& data, std::vector<double>& results);

std::tuple<size_t, size_t, size_t> pack_data_fhe(const std::vector<std::vector<double>>& data,
  const std::vector<double>& results,
  const std::vector<double>& weights,
  std::vector<std::vector<double>>& data_fhe,
  std::vector<std::vector<double>>& results_fhe,
  std::vector<double>& weights_fhe,
  size_t num_slots);

void unpack_data(const std::vector<std::vector<double>>& data_fhe, std::vector<std::vector<double>>& data, size_t rows, size_t cols, size_t last_rows, size_t num_features);

bool unpack_weights(const std::vector<double>& weights_fhe, std::vector<double>& weights, size_t num_features);

void generate_weights(size_t num_features, std::vector<double>& weights);
void save_weights(const std::string& filename, const std::vector<double>& weights);
void load_weights(const std::string& filename, size_t num_features, std::vector<double>& weights);

void print_times(const std::vector<iteration_time_t>& times, const std::string& mode, bool gpu, size_t samples);

#endif // DATA_HPP