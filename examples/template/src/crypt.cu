#include "crypt.hpp"

#include <execution>

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>encrypt_data(const std::vector<std::vector<double>> &data, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
			 const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_data(0);
	encrypted_data.reserve(data.size());
	std::for_each_n(std::execution::par, data.begin(), data.size(), [&](auto &x) {
		const lbcrypto::Plaintext plaintext = cc->MakeCKKSPackedPlaintext(x);
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext = cc->Encrypt(pk, plaintext);
		encrypted_data.push_back(ciphertext);
	});
	return encrypted_data;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt_data(std::vector<double> &data,
													  const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
													  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {
	const lbcrypto::Plaintext plaintext = cc->MakeCKKSPackedPlaintext(data);
	const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext = cc->Encrypt(pk, plaintext);
	return ciphertext;
}
