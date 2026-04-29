// Serialization headers - required for cereal type registration.
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include "CryptoContext.hpp"
#include "PrivateKey.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"

namespace fideslib::Serial {

bool SerializeToFile(const std::string& filename, const fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype) {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(obj->cpu);

	bool res = false;
	switch (sertype) {
	case SerType::BINARY: {
		res = lbcrypto::Serial::SerializeToFile(filename, context, lbcrypto::SerType::BINARY);
		break;
	}
	case SerType::JSON: {
		res = lbcrypto::Serial::SerializeToFile(filename, context, lbcrypto::SerType::JSON);
		break;
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}

	if (!res) {
		return false;
	}

	// Serialize GPU device information.
	std::string filename_dev = filename + ".dev";
	std::ofstream devFile(filename_dev, std::ios::binary);
	if (!devFile.is_open()) {
		std::cerr << "Failed to open device file for writing: " << filename_dev << std::endl;
		return false;
	}

	std::string devData = std::to_string(obj->devices.size()) + " { ";
	for (const auto& dev : obj->devices) {
		devData += std::to_string(dev) + " ";
	}
	devData += "}\n";
	devFile.write(devData.c_str(), devData.size());
	// Serialize autoload settings.
	std::string autoLoadData = "AutoLoadCiphertexts: " + std::to_string(obj->auto_load_ciphertexts) + "\n";
	devFile.write(autoLoadData.c_str(), autoLoadData.size());
	std::string autoLoadPtData = "AutoLoadPlaintexts: " + std::to_string(obj->auto_load_plaintexts) + "\n";
	devFile.write(autoLoadPtData.c_str(), autoLoadPtData.size());
	// Serialize rotation keys list if applicable.
	std::string rotationData = "RotationIndexes: { ";
	for (const auto& index : obj->rotation_indexes) {
		rotationData += std::to_string(index) + " ";
	}
	rotationData += "}\n";
	devFile.write(rotationData.c_str(), rotationData.size());
	// Serialize key distribution setting if applicable.
	std::string keyDistData = "KeyDist: " + std::to_string(obj->keyDist) + "\n";
	devFile.write(keyDistData.c_str(), keyDistData.size());
	// Serialize bootstrap slots if applicable.
	std::string bootstrapData = "BootstrapSlots: { ";
	for (const auto& slot : obj->slots_bootstrap) {
		bootstrapData += std::to_string(slot) + " ";
	}
	bootstrapData += "}\n";
	devFile.write(bootstrapData.c_str(), bootstrapData.size());
	devFile.close();
	return true;
}

bool SerializeToFile(const std::string& filename, const fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype) {
	auto& pk = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(obj->pimpl);

	switch (sertype) {
	case SerType::BINARY: {
		return lbcrypto::Serial::SerializeToFile(filename, pk, lbcrypto::SerType::BINARY);
	}
	case SerType::JSON: {
		return lbcrypto::Serial::SerializeToFile(filename, pk, lbcrypto::SerType::JSON);
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}
}

bool SerializeToFile(const std::string& filename, const fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype) {
	auto& sk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(obj->pimpl);

	switch (sertype) {
	case SerType::BINARY: {
		return lbcrypto::Serial::SerializeToFile(filename, sk, lbcrypto::SerType::BINARY);
	}
	case SerType::JSON: {
		return lbcrypto::Serial::SerializeToFile(filename, sk, lbcrypto::SerType::JSON);
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}
}

bool DeserializeFromFile(const std::string& filename, fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype) {

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context;

	bool res;
	switch (sertype) {
	case SerType::BINARY: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, context, lbcrypto::SerType::BINARY);
		break;
	}
	case SerType::JSON: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, context, lbcrypto::SerType::JSON);
		break;
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}

	if (!res) {
		return false;
	}

	CryptoContextImpl<DCRTPoly> gpu_context;
	gpu_context.cpu						 = std::make_any<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(context);
	gpu_context.devices					 = std::vector<int>();
	gpu_context.multiplicative_depth	 = context->GetCryptoParameters()->GetElementParams()->GetParams().size() - 1;
	auto ptr							 = std::make_shared<CryptoContextImpl<DCRTPoly>>(std::move(gpu_context));
	ptr->self_reference					 = std::weak_ptr<CryptoContextImpl<DCRTPoly>>(ptr);
	obj									 = ptr;

	// Deserialize GPU device information if available.
	std::ifstream devFile(filename + ".dev", std::ios::binary);
	if (!devFile.is_open()) {
		return false;
	}

	// First line: device IDs
	std::string line;
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		size_t numDevices;
		iss >> numDevices;
		char brace;
		iss >> brace; // Read '{'
		ptr->devices.clear();
		for (size_t i = 0; i < numDevices; ++i) {
			int devId;
			iss >> devId;
			ptr->devices.push_back(devId);
		}
	}
	// Second line: AutoLoadCiphertexts
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label; // Read "AutoLoadCiphertexts:"
		int flag;
		iss >> flag;
		ptr->auto_load_ciphertexts = (flag != 0);
	}
	// Third line: AutoLoadPlaintexts
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label; // Read "AutoLoadPlaintexts:"
		int flag;
		iss >> flag;
		ptr->auto_load_plaintexts = (flag != 0);
	}
	// Fourth line: RotationIndexes
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label; // Read "RotationIndexes:"
		char brace;
		iss >> brace; // Read '{'
		ptr->rotation_indexes.clear();
		int index;
		while (iss >> index) {
			ptr->rotation_indexes.push_back(index);
		}
	}
	// Fifth line: KeyDist
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label; // Read "KeyDist:"
		int dist;
		iss >> dist;
		ptr->keyDist = (SecretKeyDist)dist;
	}
	// Sixth line: Boostrap slots 
	if (std::getline(devFile, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label; // Read "BootstrapSlots:"
		char brace;
		iss >> brace; // Read '{'
		ptr->slots_bootstrap.clear();
		uint32_t slot;
		while (iss >> slot) {
			ptr->slots_bootstrap.push_back(slot);
		}
	}
	devFile.close();

	return res;
}

bool DeserializeFromFile(const std::string& filename, fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype) {

	lbcrypto::PublicKey<lbcrypto::DCRTPoly> pk;

	bool res;
	switch (sertype) {
	case SerType::BINARY: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, pk, lbcrypto::SerType::BINARY);
		break;
	}
	case SerType::JSON: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, pk, lbcrypto::SerType::JSON);
		break;
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}

	if (res) {
		PublicKey<DCRTPoly> public_key = std::make_shared<PublicKeyImpl<DCRTPoly>>();
		public_key->pimpl			   = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(pk);
		obj							   = public_key;
	}

	return res;
}

bool DeserializeFromFile(const std::string& filename, fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype) {

	lbcrypto::PrivateKey<lbcrypto::DCRTPoly> sk;

	bool res;
	switch (sertype) {
	case SerType::BINARY: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, sk, lbcrypto::SerType::BINARY);
		break;
	}
	case SerType::JSON: {
		res = lbcrypto::Serial::DeserializeFromFile(filename, sk, lbcrypto::SerType::JSON);
		break;
	}
	default: std::cerr << "Unsupported serialization type" << std::endl; return false;
	}

	if (res) {
		PrivateKey<DCRTPoly> private_key = std::make_shared<PrivateKeyImpl<DCRTPoly>>();
		private_key->pimpl				 = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(sk);
		obj								 = private_key;
	}

	return res;
}

} // namespace fideslib::Serial