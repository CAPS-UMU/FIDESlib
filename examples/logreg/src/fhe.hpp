#ifndef FHE_HPP
#define FHE_HPP

#include "data.hpp"
#include <cstdint>
#include <fideslib.hpp>

extern bool prescale;
extern bool sparse_encaps;
extern bool boot_every_iter;
extern std::vector<int> devices;
extern std::vector<uint32_t> levelBudget;
extern std::vector<uint32_t> bStep;
extern uint32_t ringDim;
extern uint32_t numSlots;

extern fideslib::CryptoContext<fideslib::DCRTPoly> cc;
extern fideslib::KeyPair<fideslib::DCRTPoly> keys;
extern uint32_t depth;

uint32_t create_context(bool inference);
void prepare_context(const fideslib::KeyPair<fideslib::DCRTPoly>& keys, size_t cols, size_t rows);

fideslib::Ciphertext<fideslib::DCRTPoly> encrypt_data(const std::vector<double>& data, const fideslib::PublicKey<fideslib::DCRTPoly>& pk, int scale_deg, int level);

std::vector<double> decrypt_data(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, const fideslib::PrivateKey<fideslib::DCRTPoly>& sk, size_t num_slots);

std::vector<iteration_time_t> logistic_regression_train(const std::vector<std::vector<double>>& data,
  const std::vector<std::vector<double>>& results,
  fideslib::Ciphertext<fideslib::DCRTPoly>& weights,
  size_t rows,
  size_t cols,
  size_t last_rows,
  size_t iterations,
  const fideslib::PublicKey<fideslib::DCRTPoly>& pk);

std::vector<iteration_time_t> logistic_regression_inference(std::vector<std::vector<double>>& data,
  const fideslib::Ciphertext<fideslib::DCRTPoly>& weights,
  size_t cols,
  const fideslib::KeyPair<fideslib::DCRTPoly>& keys);

std::vector<iteration_time_t>
fideslib_training(const std::vector<std::vector<double>>& data, const std::vector<double>& results, std::vector<double>& weights, size_t iterations);

std::pair<std::vector<iteration_time_t>, double>
fideslib_inference(const std::vector<std::vector<double>>& data, const std::vector<double>& results, const std::vector<double>& weights);

#endif // FHE_HPP