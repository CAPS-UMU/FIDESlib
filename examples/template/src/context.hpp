#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <openfhe.h>

/**
 * Presupuesto de niveles para el bootstrapping.
 */
const std::vector<uint32_t> level_budget = {1, 1};
/** 
 * Dimensión del anillo utilizdo 
 */
constexpr uint32_t ring_dim = 1 << 12;
/*
 * Slots utilizados. 
 */
constexpr uint32_t num_slots = ring_dim / 2;

/**
 * Genera un contexto de OpenFHE.
 * @return Contexto de OpenFHE.
 */
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> generate_context();

#endif //CONTEXT_HPP
