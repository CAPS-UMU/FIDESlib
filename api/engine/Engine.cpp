#include "engine/Engine.hpp"

#include <openfhe.h>

#include <string>

namespace fideslib {

// Default implementations: every operation throws until a backend overrides it, so a backend under
// construction compiles (and reports a precise error at runtime) instead of having to stub the whole
// interface up front. teardown() is the one exception — see below.

void Engine::notImplemented(const char* op) const {
	OPENFHE_THROW(std::string(op) + " is not implemented by the " + name() + " backend");
}

// ---- Negation ----
Ciphertext<DCRTPoly> Engine::evalNegate(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalNegate");
}

void Engine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalNegateInPlace");
}

// ---- Addition ----
Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalAdd");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalAddInPlace");
}

Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalAdd");
}

Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalAdd");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalAddInPlace");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalAddInPlace");
}

Ciphertext<DCRTPoly> Engine::evalAddMany(CryptoContextImpl<DCRTPoly>&, const std::vector<Ciphertext<DCRTPoly>>&) {
	notImplemented("evalAddMany");
}

void Engine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>&, std::vector<Ciphertext<DCRTPoly>>&) {
	notImplemented("evalAddManyInPlace");
}

// ---- Subtraction ----
Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, Plaintext&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, double, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSubInPlace");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalSubInPlace");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, double, Ciphertext<DCRTPoly>&) {
	notImplemented("evalSubInPlace");
}

// ---- Multiplication ----
Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalMult");
}

Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalMult");
}

Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalMult");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalMultInPlace");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalMultInPlace");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalMultInPlace");
}

// ---- Square ----
Ciphertext<DCRTPoly> Engine::evalSquare(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSquare");
}

void Engine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalSquareInPlace");
}

// ---- Rotation ----
Ciphertext<DCRTPoly> Engine::evalRotate(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t) {
	notImplemented("evalRotate");
}

void Engine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int32_t) {
	notImplemented("evalRotateInPlace");
}

std::shared_ptr<void> Engine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalFastRotationPrecompute");
}

Ciphertext<DCRTPoly> Engine::evalFastRotation(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const int32_t, const uint32_t, const std::shared_ptr<void>&) {
	notImplemented("evalFastRotation");
}

Ciphertext<DCRTPoly> Engine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const int32_t, const std::shared_ptr<void>&, bool) {
	notImplemented("evalFastRotationExt");
}

std::vector<Ciphertext<DCRTPoly>>
Engine::evalFastRotation(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const std::vector<int32_t>&, const uint32_t, const std::shared_ptr<void>&) {
	notImplemented("evalFastRotation");
}

std::vector<Ciphertext<DCRTPoly>>
Engine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const std::vector<int32_t>&, const std::shared_ptr<void>&, bool) {
	notImplemented("evalFastRotationExt");
}

// ---- Chebyshev series ----
Ciphertext<DCRTPoly> Engine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, std::vector<double>&, double, double) {
	notImplemented("evalChebyshevSeries");
}

void Engine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, std::vector<double>&, double, double) {
	notImplemented("evalChebyshevSeriesInPlace");
}

// ---- Rescale ----
Ciphertext<DCRTPoly> Engine::rescale(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("rescale");
}

void Engine::rescaleInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("rescaleInPlace");
}

// ---- Accumulate (sum reduction) ----
Ciphertext<DCRTPoly> Engine::accumulateSum(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int, int) {
	notImplemented("accumulateSum");
}

void Engine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int) {
	notImplemented("accumulateSumInPlace");
}

void Engine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, int) {
	notImplemented("accumulateSumInPlace");
}

// ---- Bootstrapping ----
BootstrapSetupPolicy Engine::bootstrapSetupPolicy(bool, bool, int32_t) const {
	notImplemented("bootstrapSetupPolicy");
}

void Engine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>&, const PrivateKey<DCRTPoly>&, uint32_t) {
	notImplemented("evalBootstrapKeyGen");
}

Ciphertext<DCRTPoly> Engine::evalBootstrap(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, uint32_t, uint32_t, bool) {
	notImplemented("evalBootstrap");
}

void Engine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, uint32_t, uint32_t, bool) {
	notImplemented("evalBootstrapInPlace");
}

// ---- Convolution transform (BSGS linear transform) ----
void Engine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, const std::vector<Plaintext>&, const std::vector<int>&, int, int) {
	notImplemented("convolutionTransformInPlace");
}

void Engine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, const std::vector<Plaintext>&, Plaintext&, const std::vector<int>&, int, int, int) {
	notImplemented("specialConvolutionTransformInPlace");
}

// ---- Host readback ----
void Engine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("recoverHostCiphertext");
}

// ---- Device residency ----
void Engine::loadContext(CryptoContextImpl<DCRTPoly>&, const PublicKey<DCRTPoly>&) {
	notImplemented("loadContext");
}

void Engine::loadPlaintext(CryptoContextImpl<DCRTPoly>&, Plaintext&) {
	notImplemented("loadPlaintext");
}

void Engine::loadCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("loadCiphertext");
}

// ---- Ciphertext backend hooks ----
std::any Engine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("cloneCiphertextBackend");
}

size_t Engine::ciphertextLevel(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextLevel");
}

size_t Engine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextNoiseScaleDeg");
}

void Engine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&, size_t) {
	notImplemented("setCiphertextSlots");
}

void Engine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&, size_t) {
	notImplemented("setCiphertextLevel");
}

// ---- Context backend state ----
bool Engine::isContextLoaded() const {
	notImplemented("isContextLoaded");
}

void Engine::synchronize() const {
	notImplemented("synchronize");
}

// The context destructor calls teardown() unconditionally, so the default must succeed (a backend with
// no device state simply has nothing to tear down) — a throw here would terminate the program.
void Engine::teardown() {
}

void Engine::setDevices(const std::vector<int>&) {
	notImplemented("setDevices");
}

std::vector<int> Engine::devices() const {
	notImplemented("devices");
}

} // namespace fideslib
