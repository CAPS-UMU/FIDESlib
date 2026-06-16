#include "controller.hpp"
#include "utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fideslib.hpp>
#include <fideslib/Definitions.hpp>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <numeric>

void resnet::generate_context(experiment_settings e) {

	fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;

	this->num_slots = 1 << 14;

	parameters.SetSecurityLevel(fideslib::HEStd_NotSet);
	parameters.SetNumLargeDigits(e.digits_hks);
	parameters.SetRingDim(1 << e.log_ring_dim);
	parameters.SetBatchSize(this->num_slots);
	parameters.SetKeySwitchTechnique(fideslib::HYBRID);

	this->level_budget = std::vector<uint32_t>({ e.cts_levels, e.stc_levels });

	parameters.SetScalingModSize(e.log_primes);
	parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
	parameters.SetFirstModSize(e.log_scale_factor);

	parameters.SetCiphertextAutoload(this->auto_load_ciphertexts);
	parameters.SetPlaintextAutoload(this->auto_load_plaintexts);

	const char* env_devices = std::getenv("FIDESLIB_USE_NUM_GPUS");
	if (env_devices) {
		int num_gpus = std::stoi(env_devices);
		this->devices.resize(num_gpus);
		std::iota(this->devices.begin(), this->devices.end(), 0);
	}

	parameters.SetBackend(fideslib::Backend::CUDA);
	parameters.SetSecretKeyDist(e.secret_key_dist);
	parameters.SetPlaintextAutoload(e.autoload);

	this->relu_degree		= e.relu_degree;
	this->parameters_folder = e.parameters_folder;
	this->circuit_depth		= e.depth;
	this->prescaled			= e.prescale;
	this->partial_load		= e.by_layer_loading;
	this->auto_load_ciphertexts = e.autoload;
	this->auto_load_plaintexts	= e.autoload;
	this->verbose				= e.verbose;

	if (this->devices.empty() && this->prescaled) {
		std::cout << "Prescale only on GPU mode." << std::endl;
		exit(1);
	}

	std::cout << "Ciphertexts depth: " << this->circuit_depth << std::endl;

	parameters.SetMultiplicativeDepth(this->circuit_depth);

	context = GenCryptoContext(parameters);
	if (!this->devices.empty()) {
		context->SetCudaDevices(this->devices);
	}

	std::cout << "Context built, generating keys..." << std::endl;

	context->Enable(fideslib::PKE);
	context->Enable(fideslib::KEYSWITCH);
	context->Enable(fideslib::LEVELEDSHE);
	context->Enable(fideslib::ADVANCEDSHE);
	context->Enable(fideslib::FHE);

	key_pair = context->KeyGen();

	context->EvalMultKeyGen(key_pair.secretKey);

	std::cout << "Generated." << std::endl;
}

void resnet::generate_rotations(experiment_settings e) {

	std::cout << "Generating rotation keys..." << std::endl;

	for (size_t layer_idx = 1; layer_idx < num_layers; ++layer_idx) {
		if (resnet::conv_image_widths[layer_idx] == 0)
			continue; // Skip if not a conv layer (though all seem to be conv or related)

		size_t bStep = 0;
		int stride	 = resnet::conv_rotation_indexes[layer_idx];
		size_t gStep = 0;

		if (layer_idx == 7 || layer_idx == 14) {
			// convbnsx
			bStep = resnet::weights_inner_shape[layer_idx];
			gStep = resnet::weights_outer_shape[layer_idx] / 2;
		} else if (layer_idx == 8) {
			// convbndx
			bStep = 1;
			gStep = resnet::weights_outer_shape[layer_idx] / 2;
		} else {
			// convbn
			bStep = resnet::weights_inner_shape[layer_idx];
			gStep = resnet::weights_outer_shape[layer_idx];
		}

		auto new_rots = context->GetConvolutionTransformRotationIndices(0, static_cast<int>(bStep), stride, static_cast<uint32_t>(gStep));
		this->rotations.insert(this->rotations.end(), new_rots.begin(), new_rots.end());
	}

	// Remove duplicates
	std::sort(this->rotations.begin(), this->rotations.end());
	this->rotations.erase(std::unique(this->rotations.begin(), this->rotations.end()), this->rotations.end());

	context->EvalRotateKeyGen(key_pair.secretKey, this->rotations);
}

void resnet::serialize_context(experiment_settings e) {
	if (!e.serialize)
		return;

	struct stat sb{};
	std::string folder = e.parameters_folder;

	if (stat(("../" + folder).c_str(), &sb) == 0) {
		std::string command = "rm -r ../" + folder;
		[[maybe_unused]] int a = system(command.c_str());
	}
	mkdir(("../" + folder).c_str(), 0777);

	write_to_file("../" + folder + "/relu_degree.txt", std::to_string(this->relu_degree));
	write_to_file("../" + folder + "/level_budget.txt", std::to_string(this->level_budget[0]) + "," + std::to_string(this->level_budget[1]));

	std::cout << "Now serializing keys ..." << std::endl;
	std::ofstream multKeyFile("../" + folder + "/mult-keys.txt", std::ios::out | std::ios::binary);

	if (multKeyFile.is_open()) {
		if (!context->SerializeEvalMultKey(multKeyFile, fideslib::SerType::BINARY)) {
			std::cerr << "Error writing EvalMult keys" << std::endl;
			exit(1);
		}
		std::cout << "EvalMult keys have been serialized" << std::endl;
		multKeyFile.close();
	} else {
		std::cerr << "Error serializing EvalMult keys in \"" << "../" + folder + "/mult-keys.txt" << "\"" << std::endl;
		exit(1);
	}

	if (!fideslib::Serial::SerializeToFile("../" + folder + "/crypto-context.txt", context, fideslib::SerType::BINARY)) {
		std::cerr << "Error writing serialization of the crypto context to crypto-context.txt" << std::endl;
	} else {
		std::cout << "Crypto Context have been serialized" << std::endl;
	}

	if (!fideslib::Serial::SerializeToFile("../" + folder + "/public-key.txt", key_pair.publicKey, fideslib::SerType::BINARY)) {
		std::cerr << "Error writing serialization of public key to public-key.txt" << std::endl;
	} else {
		std::cout << "Public Key has been serialized" << std::endl;
	}

	if (!fideslib::Serial::SerializeToFile("../" + folder + "/secret-key.txt", key_pair.secretKey, fideslib::SerType::BINARY)) {
		std::cerr << "Error writing serialization of secret key to secret-key.txt" << std::endl;
	} else {
		std::cout << "Secret Key has been serialized" << std::endl;
	}

	std::ofstream rotationKeyFile("../" + folder + "/rotations.bin", std::ios::out | std::ios::binary);
	if (rotationKeyFile.is_open()) {
		if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, fideslib::SerType::BINARY)) {
			std::cerr << "Error writing rotation keys" << std::endl;
			exit(1);
		}
		std::cout << "Rotation keys have been serialized" << std::endl;
	} else {
		std::cerr << "Error serializing Rotation keys" << "../" + folder + "/rotations.bin" << std::endl;
		exit(1);
	}
}

void resnet::generate_bootstrapping(experiment_settings e) {
	context->EvalBootstrapSetup(this->level_budget, { 16, 16 }, this->num_slots, 0);
	context->EvalBootstrapKeyGen(key_pair.secretKey, this->num_slots);
}

void resnet::deserialize_context(experiment_settings e) {

	this->parameters_folder = e.parameters_folder;
	this->verbose			= e.verbose;

	struct stat sb{};
	if (stat(("../" + parameters_folder).c_str(), &sb) != 0) {
		std::cerr << "The folder \"" << parameters_folder << "\" does not exist. Please, create it or check the name. Aborting. :-(" << std::endl;
		exit(1);
	}

	std::cout << "Reading serialized context..." << std::endl;

	if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/crypto-context.txt", context, fideslib::SerType::BINARY)) {
		std::cerr << "I cannot read serialized data from: " << "../" + parameters_folder + "/crypto-context.txt" << std::endl;
		exit(1);
	}

	fideslib::PublicKey<fideslib::DCRTPoly> clientPublicKey;
	if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/public-key.txt", clientPublicKey, fideslib::SerType::BINARY)) {
		std::cerr << "I cannot read serialized data from public-key.txt" << std::endl;
		exit(1);
	}

	fideslib::PrivateKey<fideslib::DCRTPoly> serverSecretKey;
	if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/secret-key.txt", serverSecretKey, fideslib::SerType::BINARY)) {
		std::cerr << "I cannot read serialized data from secret-key.txt" << std::endl;
		exit(1);
	}

	const char* env_devices = std::getenv("FIDESLIB_DEVICES");
	if (env_devices) {
		std::vector<int> dev_override;
		std::string str_devices(env_devices);
		std::replace(str_devices.begin(), str_devices.end(), ',', ' ');
		std::stringstream ss(str_devices);
		int device_id;
		while (ss >> device_id) {
			dev_override.push_back(device_id);
		}
		this->context->SetCudaDevices(dev_override);
	}

	key_pair.publicKey = clientPublicKey;
	key_pair.secretKey = serverSecretKey;

	std::ifstream multKeyIStream("../" + parameters_folder + "/mult-keys.txt", std::ios::in | std::ios::binary);
	if (!multKeyIStream.is_open()) {
		std::cerr << "Cannot read serialization from " << "mult-keys.txt" << std::endl;
		exit(1);
	}
	if (!context->DeserializeEvalMultKey(multKeyIStream, fideslib::SerType::BINARY)) {
		std::cerr << "Could not deserialize eval mult key file" << std::endl;
		exit(1);
	}

	std::ifstream rotationKeyIStream("../" + parameters_folder + "/rotations.bin", std::ios::in | std::ios::binary);
	if (!rotationKeyIStream.is_open()) {
		std::cerr << "Cannot read serialization from " << "rotations.bin" << std::endl;
		exit(1);
	}
	if (!context->DeserializeEvalAutomorphismKey(rotationKeyIStream, fideslib::SerType::BINARY)) {
		std::cerr << "Could not deserialize rotation key file" << std::endl;
		exit(1);
	}

	this->relu_degree = static_cast<uint32_t>(stoi(read_from_file("../" + parameters_folder + "/relu_degree.txt")));

	this->level_budget[0] = static_cast<uint32_t>(read_from_file("../" + parameters_folder + "/level_budget.txt").at(0) - '0');
	this->level_budget[1] = static_cast<uint32_t>(read_from_file("../" + parameters_folder + "/level_budget.txt").at(2) - '0');

	std::cout << "CtoS: " << level_budget[0] << ", StoC: " << level_budget[1] << std::endl;

	this->circuit_depth = this->context->multiplicative_depth;

	this->auto_load_ciphertexts = context->auto_load_ciphertexts;
	this->auto_load_plaintexts	= context->auto_load_plaintexts;
	this->devices				= this->context->GetCudaDevices();
	this->prescaled				= e.prescale;
	this->partial_load			= e.by_layer_loading;

	std::cout << "Circuit depth: " << circuit_depth << std::endl;
}

void resnet::load_context() {
	std::cout << "ResNet Controller now in GPU mode." << std::endl;
	std::cout << "Devices: ";
	for (auto d : this->devices) {
		std::cout << d << " ";
	}
	std::cout << std::endl << "Auto-load plaintexts: " << (this->auto_load_plaintexts ? "enabled" : "disabled") << std::endl;
	std::cout << "Auto-load ciphertexts: " << (this->auto_load_ciphertexts ? "enabled" : "disabled") << std::endl;

	this->context->LoadContext(this->key_pair.publicKey);
}

void resnet::load_weights_layer0() {

	if (!this->devices.empty() && this->prescaled) {
		this->prescale = this->context->GetPreScaleFactor(this->num_slots);

		std::cout << "Prescale optimization enabled: using a prescale factor of " << this->prescale << "." << std::endl;	
	}

	const uint32_t convbn_l0_levels_weights = this->prescaled ? this->circuit_depth - 2 : this->circuit_depth - 3;
	const uint32_t convbn_l0_levels_bias	= this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l0_slots			= 16384;

	// Load weights as a single flattened vector (same format as other layers)
	std::vector<fideslib::Plaintext> flat_w_l0;
	flat_w_l0.reserve(weights_outer_shape[0] * weights_inner_shape[0]);

	for (size_t j = 0; j < weights_outer_shape[0]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[0]; ++k) {
			auto vals = read_values_from_file("../weights/conv1bn1-ch" + std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin", scales[0] * this->prescale);
			flat_w_l0.push_back(encode(vals, convbn_l0_levels_weights, convbn_l0_slots));
		}
	}
	this->weights[0] = { std::move(flat_w_l0) };
	this->bias[0]	 = { encode(read_values_from_file("../weights/conv1bn1-bias.bin", scales[0] * this->prescale), convbn_l0_levels_bias, convbn_l0_slots) };

	this->mask = this->mask_from_to(0, 1024, convbn_l0_levels_weights);

	this->context->Synchronize();

	std::cout << "Loaded weights for layer 0." << std::endl;
}

void resnet::load_weights_layer1() {

	// Clear l0 weights to free memory.
	if (this->partial_load) {
		this->weights[0].clear();
		this->bias[0].clear();
		this->mask = nullptr;
	}

	const uint32_t convbn_l1_slots			= 16384;
	const uint32_t convbn_l1_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l1_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l1;
	flat_w_l1.reserve(weights_outer_shape[1] * weights_inner_shape[1]);

	for (size_t j = 0; j < weights_outer_shape[1]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[1]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(1) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[1] * this->prescale);

			flat_w_l1.push_back(encode(vals, convbn_l1_levels_weights, convbn_l1_slots));
		}
	}
	this->weights[1] = { std::move(flat_w_l1) };
	this->bias[1]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(1) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[1] * this->prescale),
		 convbn_l1_levels_bias,
		 convbn_l1_slots) };

	this->context->Synchronize();

	std::cout << "Loaded weights for layer 1." << std::endl;
}

void resnet::load_weights_layer2() {

	if (this->partial_load) {
		this->weights[1].clear();
		this->bias[1].clear();
	}

	const uint32_t convbn_l2_slots			= 16384;
	const uint32_t convbn_l2_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l2_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l2;
	flat_w_l2.reserve(weights_outer_shape[2] * weights_inner_shape[2]);

	for (size_t j = 0; j < weights_outer_shape[2]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[2]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(1) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[2] * this->prescale);
			flat_w_l2.push_back(encode(vals, convbn_l2_levels_weights, convbn_l2_slots));
		}
	}
	this->weights[2] = { std::move(flat_w_l2) };
	this->bias[2]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(1) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[2] * this->prescale),
		 convbn_l2_levels_bias,
		 convbn_l2_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 2." << std::endl;
}

void resnet::load_weights_layer3() {

	if (this->partial_load) {
		this->weights[2].clear();
		this->bias[2].clear();
	}

	const uint32_t convbn_l3_slots			= 16384;
	const uint32_t convbn_l3_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l3_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l3;
	flat_w_l3.reserve(weights_outer_shape[3] * weights_inner_shape[3]);

	for (size_t j = 0; j < weights_outer_shape[3]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[3]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(2) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[3] * this->prescale);
			flat_w_l3.push_back(encode(vals, convbn_l3_levels_weights, convbn_l3_slots));
		}
	}
	this->weights[3] = { std::move(flat_w_l3) };
	this->bias[3]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(2) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[3] * this->prescale),
		 convbn_l3_levels_bias,
		 convbn_l3_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 3." << std::endl;
}

void resnet::load_weights_layer4() {

	if (this->partial_load) {
		this->weights[3].clear();
		this->bias[3].clear();
	}

	const uint32_t convbn_l4_slots			= 16384;
	const uint32_t convbn_l4_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l4_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l4;
	flat_w_l4.reserve(weights_outer_shape[4] * weights_inner_shape[4]);

	for (size_t j = 0; j < weights_outer_shape[4]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[4]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(2) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[4] * this->prescale);
			flat_w_l4.push_back(encode(vals, convbn_l4_levels_weights, convbn_l4_slots));
		}
	}
	this->weights[4] = { std::move(flat_w_l4) };
	this->bias[4]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(2) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[4] * this->prescale),
		 convbn_l4_levels_bias,
		 convbn_l4_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 4." << std::endl;
}

void resnet::load_weights_layer5() {

	if (this->partial_load) {
		this->weights[4].clear();
		this->bias[4].clear();
	}

	const uint32_t convbn_l5_slots			= 16384;
	const uint32_t convbn_l5_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l5_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l5;
	flat_w_l5.reserve(weights_outer_shape[5] * weights_inner_shape[5]);

	for (size_t j = 0; j < weights_outer_shape[5]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[5]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(3) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[5] * this->prescale);
			flat_w_l5.push_back(encode(vals, convbn_l5_levels_weights, convbn_l5_slots));
		}
	}
	this->weights[5] = { std::move(flat_w_l5) };
	this->bias[5]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(3) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[5] * this->prescale),
		 convbn_l5_levels_bias,
		 convbn_l5_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 5." << std::endl;
}

void resnet::load_weights_layer6() {

	if (this->partial_load) {
		this->weights[5].clear();
		this->bias[5].clear();
	}

	const uint32_t convbn_l6_slots			= 16384;
	const uint32_t convbn_l6_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l6_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l6;
	flat_w_l6.reserve(weights_outer_shape[6] * weights_inner_shape[6]);

	for (size_t j = 0; j < weights_outer_shape[6]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[6]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(3) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[6] * this->prescale);
			flat_w_l6.push_back(encode(vals, convbn_l6_levels_weights, convbn_l6_slots));
		}
	}
	this->weights[6] = { std::move(flat_w_l6) };
	this->bias[6]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(3) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[6] * this->prescale),
		 convbn_l6_levels_bias,
		 convbn_l6_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 6." << std::endl;
}

void resnet::load_weights_layer7() {

	if (this->partial_load) {
		this->weights[6].clear();
		this->bias[6].clear();
	}

	const uint32_t convbn_l7_slots			= 16384;
	const uint32_t convbn_l7_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l7_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as two flattened vectors (first half and second half) for convbnsx
	size_t half_channels = weights_outer_shape[7] / 2;
	std::vector<fideslib::Plaintext> flat_w_l7_first;
	std::vector<fideslib::Plaintext> flat_w_l7_second;
	flat_w_l7_first.reserve(half_channels * weights_inner_shape[7]);
	flat_w_l7_second.reserve(half_channels * weights_inner_shape[7]);

	// First half
	for (size_t j = 0; j < half_channels; ++j) {
		for (size_t k = 0; k < weights_inner_shape[7]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(4) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[7] * this->prescale);
			flat_w_l7_first.push_back(encode(vals, convbn_l7_levels_weights, convbn_l7_slots));
		}
	}

	// Second half
	for (size_t j = half_channels; j < weights_outer_shape[7]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[7]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(4) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[7] * this->prescale);
			flat_w_l7_second.push_back(encode(vals, convbn_l7_levels_weights, convbn_l7_slots));
		}
	}

	this->weights[7] = { std::move(flat_w_l7_first), std::move(flat_w_l7_second) };
	this->bias[7] = { encode(read_values_from_file("../weights/layer" + std::to_string(4) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias1.bin",
							   scales[7] * this->prescale),
						convbn_l7_levels_bias,
						convbn_l7_slots),
		encode(read_values_from_file(
				 "../weights/layer" + std::to_string(4) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias2.bin", scales[7] * this->prescale),
		  convbn_l7_levels_bias,
		  convbn_l7_slots) };

	std::cout << "Loaded weights for layer 7." << std::endl;

	// Downsampling masks.

	const uint32_t downsample_masks_0_slots		 = 16384;
	const uint32_t downsample_masks_0_full_slots = 16384 * 2;

	this->downsample_masks[0].resize(5 + 16 + 32);

	this->downsample_masks[0][0] = this->mask_first_n(downsample_masks_0_slots, downsample_masks_0_full_slots, this->circuit_depth - 8);
	this->downsample_masks[0][1] = this->mask_second_n(downsample_masks_0_slots, downsample_masks_0_full_slots, this->circuit_depth - 8);
	this->downsample_masks[0][2] = this->gen_mask(2, downsample_masks_0_full_slots, this->circuit_depth - 7);
	this->downsample_masks[0][3] = this->gen_mask(4, downsample_masks_0_full_slots, this->circuit_depth - 6);
	this->downsample_masks[0][4] = this->gen_mask(8, downsample_masks_0_full_slots, this->circuit_depth - 5);
	for (uint32_t i = 5; i < (5 + 16); ++i) {
		this->downsample_masks[0][i] = this->mask_first_n_mod(16, 1024, i - 5, this->circuit_depth - 4);
	}
	for (uint32_t i = (5 + 16); i < (5 + 16 + 32); ++i) {
		this->downsample_masks[0][i] = this->mask_channel(i - (5 + 16), this->circuit_depth - 3);
	}
	std::cout << "All downsampling masks generated." << std::endl;
	this->zero_ct = this->encrypt({ 0.0 }, this->circuit_depth - 4, this->num_slots);

	this->context->Synchronize();

	std::cout << "Generated zero ciphertext." << std::endl;
}

void resnet::load_weights_layer8() {

	if (this->partial_load) {
		this->weights[7].clear();
		this->bias[7].clear();
	}

	const uint32_t convbn_l8_slots			= 16384;
	const uint32_t convbn_l8_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l8_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as two flat vectors for 1x1 convolution (bStep=1)
	size_t half_channels = weights_outer_shape[8] / 2;
	std::vector<fideslib::Plaintext> flat_w_l8_first;
	std::vector<fideslib::Plaintext> flat_w_l8_second;
	flat_w_l8_first.reserve(half_channels); // bStep=1, so just half_channels weights
	flat_w_l8_second.reserve(half_channels);

	// First half: channels 0 to half_channels-1
	for (size_t j = 0; j < half_channels; ++j) {
		auto vals = read_values_from_file("../weights/layer" + std::to_string(4) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
			std::to_string(j) + "-k" + std::to_string(1) + ".bin",
		  scales[8] * this->prescale);
		flat_w_l8_first.push_back(encode(vals, convbn_l8_levels_weights, convbn_l8_slots));
	}

	// Second half: channels half_channels to weights_outer_shape[8]-1
	for (size_t j = half_channels; j < weights_outer_shape[8]; ++j) {
		auto vals = read_values_from_file("../weights/layer" + std::to_string(4) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
			std::to_string(j) + "-k" + std::to_string(1) + ".bin",
		  scales[8] * this->prescale);
		flat_w_l8_second.push_back(encode(vals, convbn_l8_levels_weights, convbn_l8_slots));
	}

	this->weights[8] = { std::move(flat_w_l8_first), std::move(flat_w_l8_second) };
	this->bias[8] = { encode(read_values_from_file("../weights/layer" + std::to_string(4) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias1.bin",
							   scales[8] * this->prescale),
						convbn_l8_levels_bias,
						convbn_l8_slots),
		encode(read_values_from_file(
				 "../weights/layer" + std::to_string(4) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias2.bin", scales[8] * this->prescale),
		  convbn_l8_levels_bias,
		  convbn_l8_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 8." << std::endl;
}

void resnet::load_weights_layer9() {

	if (this->partial_load) {
		this->weights[8].clear();
		this->bias[8].clear();
		this->downsample_masks[0].clear();
	}

	const uint32_t convbn_l9_slots			= 8192;
	const uint32_t convbn_l9_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l9_levels_bias	= this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l9;
	flat_w_l9.reserve(weights_outer_shape[9] * weights_inner_shape[9]);

	for (size_t j = 0; j < weights_outer_shape[9]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[9]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(4) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[9] * this->prescale);
			flat_w_l9.push_back(encode(vals, convbn_l9_levels_weights, convbn_l9_slots));
		}
	}
	this->weights[9] = { std::move(flat_w_l9) };
	this->bias[9]	 = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(4) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[9] * this->prescale),
		 convbn_l9_levels_bias,
		 convbn_l9_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 9." << std::endl;
}

void resnet::load_weights_layer10() {

	if (this->partial_load) {
		this->weights[9].clear();
		this->bias[9].clear();
	}

	const uint32_t convbn_l10_slots			 = 8192;
	const uint32_t convbn_l10_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l10_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l10;
	flat_w_l10.reserve(weights_outer_shape[10] * weights_inner_shape[10]);

	for (size_t j = 0; j < weights_outer_shape[10]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[10]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(5) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[10] * this->prescale);
			flat_w_l10.push_back(encode(vals, convbn_l10_levels_weights, convbn_l10_slots));
		}
	}
	this->weights[10] = { std::move(flat_w_l10) };
	this->bias[10]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(5) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[10] * this->prescale),
		 convbn_l10_levels_bias,
		 convbn_l10_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 10." << std::endl;
}

void resnet::load_weights_layer11() {

	if (this->partial_load) {
		this->weights[10].clear();
		this->bias[10].clear();
	}

	const uint32_t convbn_l11_slots			 = 8192;
	const uint32_t convbn_l11_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l11_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l11;
	flat_w_l11.reserve(weights_outer_shape[11] * weights_inner_shape[11]);

	for (size_t j = 0; j < weights_outer_shape[11]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[11]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(5) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[11] * this->prescale);
			flat_w_l11.push_back(encode(vals, convbn_l11_levels_weights, convbn_l11_slots));
		}
	}
	this->weights[11] = { std::move(flat_w_l11) };
	this->bias[11]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(5) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[11] * this->prescale),
		 convbn_l11_levels_bias,
		 convbn_l11_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 11." << std::endl;
}

void resnet::load_weights_layer12() {

	if (this->partial_load) {
		this->weights[11].clear();
		this->bias[11].clear();
	}

	const uint32_t convbn_l12_slots			 = 8192;
	const uint32_t convbn_l12_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l12_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l12;
	flat_w_l12.reserve(weights_outer_shape[12] * weights_inner_shape[12]);

	for (size_t j = 0; j < weights_outer_shape[12]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[12]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(6) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[12] * this->prescale);
			flat_w_l12.push_back(encode(vals, convbn_l12_levels_weights, convbn_l12_slots));
		}
	}
	this->weights[12] = { std::move(flat_w_l12) };
	this->bias[12]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(6) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[12] * this->prescale),
		 convbn_l12_levels_bias,
		 convbn_l12_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 12." << std::endl;
}

void resnet::load_weights_layer13() {

	if (this->partial_load) {
		this->weights[12].clear();
		this->bias[12].clear();
	}

	const uint32_t convbn_l13_slots			 = 8192;
	const uint32_t convbn_l13_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l13_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l13;
	flat_w_l13.reserve(weights_outer_shape[13] * weights_inner_shape[13]);

	for (size_t j = 0; j < weights_outer_shape[13]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[13]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(6) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[13] * this->prescale);
			flat_w_l13.push_back(encode(vals, convbn_l13_levels_weights, convbn_l13_slots));
		}
	}
	this->weights[13] = { std::move(flat_w_l13) };
	this->bias[13]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(6) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[13] * this->prescale),
		 convbn_l13_levels_bias,
		 convbn_l13_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 13." << std::endl;
}

void resnet::load_weights_layer14() {

	if (this->partial_load) {
		this->weights[13].clear();
		this->bias[13].clear();
	}

	const uint32_t convbn_l14_slots			 = 8192;
	const uint32_t convbn_l14_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l14_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as two flattened vectors (first half and second half) for convbnsx
	size_t half_channels = weights_outer_shape[14] / 2;
	std::vector<fideslib::Plaintext> flat_w_l14_first;
	std::vector<fideslib::Plaintext> flat_w_l14_second;
	flat_w_l14_first.reserve(half_channels * weights_inner_shape[14]);
	flat_w_l14_second.reserve(half_channels * weights_inner_shape[14]);

	// First half
	for (size_t j = 0; j < half_channels; ++j) {
		for (size_t k = 0; k < weights_inner_shape[14]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(7) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[14] * this->prescale);
			flat_w_l14_first.push_back(encode(vals, convbn_l14_levels_weights, convbn_l14_slots));
		}
	}

	// Second half
	for (size_t j = half_channels; j < weights_outer_shape[14]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[14]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(7) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[14] * this->prescale);
			flat_w_l14_second.push_back(encode(vals, convbn_l14_levels_weights, convbn_l14_slots));
		}
	}

	this->weights[14] = { std::move(flat_w_l14_first), std::move(flat_w_l14_second) };
	this->bias[14] = { encode(read_values_from_file("../weights/layer" + std::to_string(7) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias1.bin",
								scales[14] * this->prescale),
						 convbn_l14_levels_bias,
						 convbn_l14_slots),
		encode(read_values_from_file(
				 "../weights/layer" + std::to_string(7) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias2.bin", scales[14] * this->prescale),
		  convbn_l14_levels_bias,
		  convbn_l14_slots) };

	std::cout << "Loaded weights for layer 14." << std::endl;

	// Downsampling masks.

	this->downsample_masks[1].resize(5 + 32 + 64);
	const uint32_t downsample_masks_1_slots		 = 8192;
	const uint32_t downsample_masks_1_full_slots = 8192 * 2;

	this->downsample_masks[1][0] = this->mask_first_n(downsample_masks_1_slots, downsample_masks_1_full_slots, this->circuit_depth - 7);
	this->downsample_masks[1][1] = this->mask_second_n(downsample_masks_1_slots, downsample_masks_1_full_slots, this->circuit_depth - 7);
	this->downsample_masks[1][2] = this->gen_mask(2, downsample_masks_1_full_slots, this->circuit_depth - 6);
	this->downsample_masks[1][3] = this->gen_mask(4, downsample_masks_1_full_slots, this->circuit_depth - 5);
	this->downsample_masks[1][4] = this->gen_mask(8, downsample_masks_1_full_slots, this->circuit_depth - 4);

	for (uint32_t i = 5; i < (5 + 32); ++i) {
		this->downsample_masks[1][i] = this->mask_first_n_mod2(8, 256, i - 5, this->circuit_depth - 3);
	}

	for (uint32_t i = (5 + 32); i < (5 + 32 + 64); ++i) {
		this->downsample_masks[1][i] = this->mask_channel_2(i - (5 + 32), this->circuit_depth - 2);
	}
	this->context->Synchronize();

	std::cout << "All downsampling masks generated." << std::endl;
}

void resnet::load_weights_layer15() {

	if (this->partial_load) {
		this->weights[14].clear();
		this->bias[14].clear();
	}

	const uint32_t convbn_l15_slots			 = 8192;
	const uint32_t convbn_l15_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l15_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as two flat vectors for 1x1 convolution (bStep=1)
	size_t half_channels = weights_outer_shape[15] / 2;
	std::vector<fideslib::Plaintext> flat_w_l15_first;
	std::vector<fideslib::Plaintext> flat_w_l15_second;
	flat_w_l15_first.reserve(half_channels); // bStep=1, so just half_channels weights
	flat_w_l15_second.reserve(half_channels);

	// First half: channels 0 to half_channels-1
	for (size_t j = 0; j < half_channels; ++j) {
		auto vals = read_values_from_file("../weights/layer" + std::to_string(7) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
			std::to_string(j) + "-k" + std::to_string(1) + ".bin",
		  scales[15] * this->prescale);
		flat_w_l15_first.push_back(encode(vals, convbn_l15_levels_weights, convbn_l15_slots));
	}

	// Second half: channels half_channels to weights_outer_shape[15]-1
	for (size_t j = half_channels; j < weights_outer_shape[15]; ++j) {
		auto vals = read_values_from_file("../weights/layer" + std::to_string(7) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
			std::to_string(j) + "-k" + std::to_string(1) + ".bin",
		  scales[15] * this->prescale);
		flat_w_l15_second.push_back(encode(vals, convbn_l15_levels_weights, convbn_l15_slots));
	}

	this->weights[15] = { std::move(flat_w_l15_first), std::move(flat_w_l15_second) };
	this->bias[15] = { encode(read_values_from_file("../weights/layer" + std::to_string(7) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias1.bin",
								scales[15] * this->prescale),
						 convbn_l15_levels_bias,
						 convbn_l15_slots),
		encode(read_values_from_file(
				 "../weights/layer" + std::to_string(7) + "dx-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias2.bin", scales[15] * this->prescale),
		  convbn_l15_levels_bias,
		  convbn_l15_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 15." << std::endl;
}

void resnet::load_weights_layer16() {

	if (this->partial_load) {
		this->weights[15].clear();
		this->bias[15].clear();
		this->downsample_masks[1].clear();
	}

	const uint32_t convbn_l16_slots			 = 4096;
	const uint32_t convbn_l16_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l16_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l16;
	flat_w_l16.reserve(weights_outer_shape[16] * weights_inner_shape[16]);

	for (size_t j = 0; j < weights_outer_shape[16]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[16]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(7) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[16] * this->prescale);
			flat_w_l16.push_back(encode(vals, convbn_l16_levels_weights, convbn_l16_slots));
		}
	}
	this->weights[16] = { std::move(flat_w_l16) };
	this->bias[16]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(7) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[16] * this->prescale),
		 convbn_l16_levels_bias,
		 convbn_l16_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 16." << std::endl;
}

void resnet::load_weights_layer17() {

	if (this->partial_load) {
		this->weights[16].clear();
		this->bias[16].clear();
	}

	const uint32_t convbn_l17_slots			 = 4096;
	const uint32_t convbn_l17_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l17_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l17;
	flat_w_l17.reserve(weights_outer_shape[17] * weights_inner_shape[17]);

	for (size_t j = 0; j < weights_outer_shape[17]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[17]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(8) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[17] * this->prescale);
			flat_w_l17.push_back(encode(vals, convbn_l17_levels_weights, convbn_l17_slots));
		}
	}
	this->weights[17] = { std::move(flat_w_l17) };
	this->bias[17]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(8) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[17] * this->prescale),
		 convbn_l17_levels_bias,
		 convbn_l17_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 17." << std::endl;
}

void resnet::load_weights_layer18() {

	if (this->partial_load) {
		this->weights[17].clear();
		this->bias[17].clear();
	}

	const uint32_t convbn_l18_slots			 = 4096;
	const uint32_t convbn_l18_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l18_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l18;
	flat_w_l18.reserve(weights_outer_shape[18] * weights_inner_shape[18]);

	for (size_t j = 0; j < weights_outer_shape[18]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[18]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(8) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[18] * this->prescale);
			flat_w_l18.push_back(encode(vals, convbn_l18_levels_weights, convbn_l18_slots));
		}
	}
	this->weights[18] = { std::move(flat_w_l18) };
	this->bias[18]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(8) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[18] * this->prescale),
		 convbn_l18_levels_bias,
		 convbn_l18_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 18." << std::endl;
}

void resnet::load_weights_layer19() {

	if (this->partial_load) {
		this->weights[18].clear();
		this->bias[18].clear();
	}

	const uint32_t convbn_l19_slots			 = 4096;
	const uint32_t convbn_l19_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l19_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l19;
	flat_w_l19.reserve(weights_outer_shape[19] * weights_inner_shape[19]);

	for (size_t j = 0; j < weights_outer_shape[19]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[19]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(9) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[19] * this->prescale);
			flat_w_l19.push_back(encode(vals, convbn_l19_levels_weights, convbn_l19_slots));
		}
	}
	this->weights[19] = { std::move(flat_w_l19) };
	this->bias[19]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(9) + "-conv" + std::to_string(1) + "bn" + std::to_string(1) + "-bias.bin", scales[19] * this->prescale),
		 convbn_l19_levels_bias,
		 convbn_l19_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 19." << std::endl;
}

void resnet::load_weights_layer20() {

	if (this->partial_load) {
		this->weights[19].clear();
		this->bias[19].clear();
	}

	const uint32_t convbn_l20_slots			 = 4096;
	const uint32_t convbn_l20_levels_weights = this->prescaled ? this->circuit_depth - 1 : this->circuit_depth - 2;
	const uint32_t convbn_l20_levels_bias	 = this->prescaled ? this->circuit_depth : this->circuit_depth - 1;

	// Load weights as a single flattened vector
	std::vector<fideslib::Plaintext> flat_w_l20;
	flat_w_l20.reserve(weights_outer_shape[20] * weights_inner_shape[20]);

	for (size_t j = 0; j < weights_outer_shape[20]; ++j) {
		for (size_t k = 0; k < weights_inner_shape[20]; ++k) {
			auto vals = read_values_from_file("../weights/layer" + std::to_string(9) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-ch" +
				std::to_string(j) + "-k" + std::to_string(k + 1) + ".bin",
			  scales[20] * this->prescale);
			flat_w_l20.push_back(encode(vals, convbn_l20_levels_weights, convbn_l20_slots));
		}
	}
	this->weights[20] = { std::move(flat_w_l20) };
	this->bias[20]	  = { encode(
		 read_values_from_file("../weights/layer" + std::to_string(9) + "-conv" + std::to_string(2) + "bn" + std::to_string(2) + "-bias.bin", scales[20] * this->prescale),
		 convbn_l20_levels_bias,
		 convbn_l20_slots) };
	this->context->Synchronize();

	std::cout << "Loaded weights for layer 20." << std::endl;
}

void resnet::load_weights_final() {

	if (this->partial_load) {
		this->weights[20].clear();
		this->bias[20].clear();
	}

	this->weight_final = encode(read_fc_weight("../weights/fc.bin"), 14, 4096);
	this->final_mask   = mask_mod(64, 14, 1.0 / 64.0);
	this->context->Synchronize();

	std::cout << "Loaded weights for final FC layer." << std::endl;
}

void resnet::load_weights() {

	for (const auto& load_func : this->load_funcs) {
		load_func();
	}

	std::cout << "All weights loaded." << std::endl;
}

fideslib::Plaintext resnet::encode(const std::vector<double>& in, uint32_t level, uint32_t plaintext_slots) {

	if (plaintext_slots == 0) {
		plaintext_slots = this->num_slots;
	}

	auto plaintext = this->context->MakeCKKSPackedPlaintext(in, 1, level, nullptr, plaintext_slots);
	plaintext->SetLength(plaintext_slots);

	return plaintext;
}

fideslib::Ciphertext<fideslib::DCRTPoly> resnet::encrypt(const std::vector<double>& in, uint32_t level, uint32_t plaintext_slots) {

	if (plaintext_slots == 0) {
		plaintext_slots = this->num_slots;
	}

	auto plaintext = this->encode(in, level, plaintext_slots);
	return this->context->Encrypt(plaintext, this->key_pair.publicKey);
}

fideslib::Plaintext resnet::decrypt(fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {

	fideslib::Plaintext ptxt;

	this->context->Decrypt(ct, this->key_pair.secretKey, &ptxt);
	return ptxt;
}

void resnet::bootstrap(fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {

	this->context->EvalBootstrapInPlace(ct, 1, 0, this->prescaled);
}

size_t resnet::execute_resnet_inference(const std::string& input_image) {

	std::vector<double> input_image_data = read_image(input_image.c_str());

	bool loaded = false;
	fideslib::Ciphertext<fideslib::DCRTPoly> encrypted_image;
	std::array<std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000>>>, 22> starts, ends;

	for (size_t i = 0; i < 10; ++i) {

	auto intial_level										 = this->prescaled ? this->circuit_depth - 2 : this->circuit_depth - 3;
	encrypted_image = this->encrypt(input_image_data, intial_level, this->num_slots);

	if (this->verbose) {
		this->measure_bootstrap_precision();
	}

	if (!this->partial_load && !loaded) {
		this->load_weights();
		loaded = true;
	}

	// Block 0. Layer 0
	if (this->partial_load && !loaded)
		this->load_funcs[0]();
	starts[0] = std::chrono::high_resolution_clock::now();
	this->print(encrypted_image, "L0 Input");
	this->convbn(0, encrypted_image);
	this->print(encrypted_image, "L0 ConvBN");
	this->bootstrap(encrypted_image);
	this->print(encrypted_image, "L0 Bootstrap");
	if (this->prescaled) {
		encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
	}
	this->relu(encrypted_image, scales[0]);
	this->print(encrypted_image, "L0 ReLU");
	ends[0] = std::chrono::high_resolution_clock::now();
	// Block 1.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[1]();
		starts[1] = std::chrono::high_resolution_clock::now();
		auto in	  = encrypted_image->Clone();

		// Layer 1.
		this->print(encrypted_image, "L1 Input");
		this->convbn(1, encrypted_image);
		this->print(encrypted_image, "L1 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L1 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[1]);
		this->print(encrypted_image, "L1 ReLU");
		ends[1] = std::chrono::high_resolution_clock::now();

		// Layer 2.
		if (this->partial_load && !loaded)
			this->load_funcs[2]();
		starts[2] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L2 Input");
		this->convbn(2, encrypted_image);
		this->print(encrypted_image, "L2 ConvBN");
		this->context->EvalMultInPlace(in, scales[2] * this->prescale);
		this->print(in, "L2 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L2 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L2 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[2]);
		this->print(encrypted_image, "L2 ReLU");
		ends[2] = std::chrono::high_resolution_clock::now();
	}

	// Block 2.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[3]();
		starts[3] = std::chrono::high_resolution_clock::now();
		auto in	  = encrypted_image->Clone();

		// Layer 3.
		this->print(encrypted_image, "L3 Input");
		this->convbn(3, encrypted_image);
		this->print(encrypted_image, "L3 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L3 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[3]);
		this->print(encrypted_image, "L3 ReLU");
		ends[3] = std::chrono::high_resolution_clock::now();

		// Layer 4.
		if (this->partial_load && !loaded)
			this->load_funcs[4]();
		starts[4] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L4 Input");
		this->convbn(4, encrypted_image);
		this->print(encrypted_image, "L4 ConvBN");
		this->context->EvalMultInPlace(in, scales[4] * this->prescale);
		this->print(in, "L4 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L4 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L4 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[4]);
		this->print(encrypted_image, "L4 ReLU");
		ends[4] = std::chrono::high_resolution_clock::now();
	}

	// Block 3.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[5]();
		starts[5] = std::chrono::high_resolution_clock::now();
		auto in	  = encrypted_image->Clone();

		// Layer 5.
		this->print(encrypted_image, "L5 Input");
		this->convbn(5, encrypted_image);
		this->print(encrypted_image, "L5 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L5 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[5]);
		this->print(encrypted_image, "L5 ReLU");
		ends[5] = std::chrono::high_resolution_clock::now();

		// Layer 6.
		if (this->partial_load && !loaded)
			this->load_funcs[6]();
		starts[6] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L6 Input");
		this->convbn(6, encrypted_image);
		this->print(encrypted_image, "L6 ConvBN");
		this->context->EvalMultInPlace(in, scales[6] * this->prescale);
		this->print(in, "L6 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L6 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L6 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[6]);
		this->print(encrypted_image, "L6 ReLU");
		ends[6] = std::chrono::high_resolution_clock::now();
	}

	// Block 4.
	{
		// Layer 7. SX.
		if (this->partial_load && !loaded)
			this->load_funcs[7]();
		starts[7] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L7 Input");
		auto sx = this->convbnsx(7, encrypted_image);
		this->bootstrap(sx[0]);
		this->bootstrap(sx[1]);
		this->print(sx[0], "L7 SX 0 Bootstrap");
		this->print(sx[1], "L7 SX 1 Bootstrap");
		auto sxd = this->downsample(0, std::move(sx));
		if (this->prescaled)
			sxd->SetLevel(this->circuit_depth - 1);
		this->print(sxd, "L7 SX Downsample");
		this->bootstrap(sxd);
		this->print(sxd, "L7 SX After Bootstrap");
		if (this->prescaled) {
			sxd->SetLevel(sxd->GetLevel() + 1);
		}
		this->relu(sxd, scales[7]);
		this->print(sxd, "L7 SX ReLU");
		ends[7] = std::chrono::high_resolution_clock::now();

		// Layer 8. DX.
		if (this->partial_load && !loaded)
			this->load_funcs[8]();
		starts[8] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L8 Input");
		auto dx = this->convbndx(8, std::move(encrypted_image));
		this->bootstrap(dx[0]);
		this->bootstrap(dx[1]);
		this->print(dx[0], "L8 DX 0 Bootstrap");
		this->print(dx[1], "L8 DX 1 Bootstrap");
		auto dxd = this->downsample(0, std::move(dx));
		this->print(dxd, "L8 DX Downsample");
		ends[8] = std::chrono::high_resolution_clock::now();

		// Move back to main flow.
		encrypted_image = std::move(sxd);

		// Layer 9.
		if (this->partial_load && !loaded)
			this->load_funcs[9]();
		starts[9] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L9 Input");
		this->convbn(9, encrypted_image);
		this->print(encrypted_image, "L9 ConvBN");
		this->context->EvalAddInPlace(encrypted_image, dxd);
		this->print(encrypted_image, "L9 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L9 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[9]);
		this->print(encrypted_image, "L9 ReLU");
		ends[9] = std::chrono::high_resolution_clock::now();
	}

	// Block 5.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[10]();
		starts[10] = std::chrono::high_resolution_clock::now();
		auto in	   = encrypted_image->Clone();

		// Layer 10.
		this->print(encrypted_image, "L10 Input");
		this->convbn(10, encrypted_image);
		this->print(encrypted_image, "L10 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L10 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[10]);
		this->print(encrypted_image, "L10 ReLU");
		ends[10] = std::chrono::high_resolution_clock::now();

		// Layer 11.
		if (this->partial_load && !loaded)
			this->load_funcs[11]();
		starts[11] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L11 Input");
		this->convbn(11, encrypted_image);
		this->print(encrypted_image, "L11 ConvBN");
		this->context->EvalMultInPlace(in, scales[11] * this->prescale);
		this->print(in, "L11 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L11 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L11 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[11]);
		this->print(encrypted_image, "L11 ReLU");
		ends[11] = std::chrono::high_resolution_clock::now();
	}

	// Block 6.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[12]();
		starts[12] = std::chrono::high_resolution_clock::now();
		auto in	   = encrypted_image->Clone();

		// Layer 12.
		this->print(encrypted_image, "L12 Input");
		this->convbn(12, encrypted_image);
		this->print(encrypted_image, "L12 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L12 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[12]);
		this->print(encrypted_image, "L12 ReLU");
		ends[12] = std::chrono::high_resolution_clock::now();

		// Layer 13.
		if (this->partial_load && !loaded)
			this->load_funcs[13]();
		starts[13] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L13 Input");
		this->convbn(13, encrypted_image);
		this->print(encrypted_image, "L13 ConvBN");
		this->context->EvalMultInPlace(in, scales[13] * this->prescale);
		this->print(in, "L13 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L13 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L13 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[13]);
		this->print(encrypted_image, "L13 ReLU");
		ends[13] = std::chrono::high_resolution_clock::now();
	}

	// Block 7.
	{
		// Layer 14. SX.
		if (this->partial_load && !loaded)
			this->load_funcs[14]();
		starts[14] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L14 SX Input");
		auto sx = this->convbnsx(14, encrypted_image);
		this->bootstrap(sx[0]);
		this->bootstrap(sx[1]);
		this->context->SetLevel(sx[0], this->circuit_depth - 7);
		this->context->SetLevel(sx[1], this->circuit_depth - 7);
		this->print(sx[0], "L14 SX 0 Bootstrap");
		this->print(sx[1], "L14 SX 1 Bootstrap");
		auto sxd = this->downsample(1, std::move(sx));
		this->print(sxd, "L14 SX Downsample");
		if (this->prescaled)
			sxd->SetLevel(this->circuit_depth - 1);
		this->bootstrap(sxd);
		this->print(sxd, "L14 SX After Bootstrap");
		if (this->prescaled) {
			sxd->SetLevel(sxd->GetLevel() + 1);
		}
		this->relu(sxd, scales[14]);
		this->print(sxd, "L14 SX ReLU");
		ends[14] = std::chrono::high_resolution_clock::now();

		// Layer 15. DX.
		if (this->partial_load && !loaded)
			this->load_funcs[15]();
		starts[15] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L15 DX Input");
		auto dx = this->convbndx(15, std::move(encrypted_image));
		this->bootstrap(dx[0]);
		this->bootstrap(dx[1]);
		this->context->SetLevel(dx[0], this->circuit_depth - 7);
		this->context->SetLevel(dx[1], this->circuit_depth - 7);
		this->print(dx[0], "L15 DX 0 Bootstrap");
		this->print(dx[1], "L15 DX 1 Bootstrap");
		auto dxd = this->downsample(1, std::move(dx));
		this->print(dxd, "L15 DX Downsample");
		ends[15] = std::chrono::high_resolution_clock::now();

		// Move back to main flow.
		encrypted_image = std::move(sxd);

		// Layer 16.
		if (this->partial_load && !loaded)
			this->load_funcs[16]();
		starts[16] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L16 Input");
		this->convbn(16, encrypted_image);
		this->print(encrypted_image, "L16 ConvBN");
		this->context->EvalAddInPlace(encrypted_image, dxd);
		this->print(encrypted_image, "L16 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L16 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[16]);
		this->print(encrypted_image, "L16 ReLU");
		ends[16] = std::chrono::high_resolution_clock::now();
	}

	// Block 8.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[17]();
		starts[17] = std::chrono::high_resolution_clock::now();
		auto in	   = encrypted_image->Clone();

		// Layer 17.
		this->print(encrypted_image, "L17 Input");
		this->convbn(17, encrypted_image);
		this->print(encrypted_image, "L17 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L17 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[17]);
		this->print(encrypted_image, "L17 ReLU");
		ends[17] = std::chrono::high_resolution_clock::now();

		// Layer 18.
		if (this->partial_load && !loaded)
			this->load_funcs[18]();
		starts[18] = std::chrono::high_resolution_clock::now();
		this->print(encrypted_image, "L18 Input");
		this->convbn(18, encrypted_image);
		this->print(encrypted_image, "L18 ConvBN");
		this->context->EvalMultInPlace(in, scales[18] * this->prescale);
		this->print(in, "L18 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L18 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L18 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[18]);
		this->print(encrypted_image, "L18 ReLU");
		ends[18] = std::chrono::high_resolution_clock::now();
	}

	// Block 9.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[19]();
		starts[19] = std::chrono::high_resolution_clock::now();
		auto in	   = encrypted_image->Clone();

		// Layer 19.
		this->print(encrypted_image, "L19 Input");
		this->convbn(19, encrypted_image);
		this->print(encrypted_image, "L19 ConvBN");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L19 Bootstrap");
		if (this->prescaled) {
			encrypted_image->SetLevel(encrypted_image->GetLevel() + 1);
		}
		this->relu(encrypted_image, scales[19]);
		this->print(encrypted_image, "L19 ReLU");
		ends[19] = std::chrono::high_resolution_clock::now();

		// Layer 20.
		if (this->partial_load && !loaded)
			this->load_funcs[20]();
		starts[20] = std::chrono::high_resolution_clock::now();
		this->convbn(20, encrypted_image);
		this->print(encrypted_image, "L20 ConvBN");
		this->context->EvalMultInPlace(in, scales[20] * this->prescale);
		this->print(in, "L20 Input Multiplied");
		this->context->EvalAddInPlace(encrypted_image, in);
		this->print(encrypted_image, "L20 Added");
		this->bootstrap(encrypted_image);
		this->print(encrypted_image, "L20 Bootstrap");
		this->relu(encrypted_image, scales[20]);
		this->print(encrypted_image, "L20 ReLU");
		ends[20] = std::chrono::high_resolution_clock::now();
	}

	// Block 10.
	{
		if (this->partial_load && !loaded)
			this->load_funcs[21]();
		starts[21] = std::chrono::high_resolution_clock::now();
		this->context->AccumulateSumInPlace(encrypted_image, 64, 1);
		this->print(encrypted_image, "L21 After first AccumulateSum");
		this->context->EvalMultInPlace(encrypted_image, this->final_mask);
		this->print(encrypted_image, "L21 After final mask Mult");
		this->context->AccumulateSumInPlace(encrypted_image, 16, 1);
		this->print(encrypted_image, "L21 After second AccumulateSum");
		this->context->EvalRotateInPlace(encrypted_image, -16 + 1);
		this->print(encrypted_image, "L21 After Rotate");
		this->context->EvalMultInPlace(encrypted_image, this->weight_final);
		this->print(encrypted_image, "L21 After final weight Mult");
		this->context->AccumulateSumInPlace(encrypted_image, 64, 64);
		ends[21] = std::chrono::high_resolution_clock::now();
	}

	loaded = this->partial_load ? false : true;
	}

	// Elapsed time in milliseconds
	auto total = std::chrono::duration<double, std::milli>();
	for (size_t i = 0; i < starts.size(); ++i) {
		auto elapsed = std::chrono::duration<double, std::milli>(ends[i] - starts[i]);
		std::cout << "Layer " << i << " time: " << elapsed.count() << " ms" << std::endl;
		total += elapsed;
	}
	std::cout << "Inference time: " << total.count() << " ms" << std::endl;

	{
		fideslib::Plaintext result = this->decrypt(encrypted_image);
		result->SetLength(10);
		result->SetSlots(10);

		if (this->verbose) {
			std::cout << "Decrypted final result: " << result << std::endl;
		}

		auto vals = result->GetRealPackedValue();

		const auto max_element_iterator = std::max_element(vals.begin(), vals.end());
		const auto index_max			= static_cast<size_t>(std::distance(vals.begin(), max_element_iterator));

		return index_max;
	}
}

void resnet::measure_bootstrap_precision() {
	std::cout << "Computing boostrap precision..." << std::endl;

	fideslib::Ciphertext<fideslib::DCRTPoly> initial = encrypt({ 0 }, this->circuit_depth - 2, num_slots);

	auto clone_initial = initial->Clone();
	auto a			   = decrypt(clone_initial);

	//this->context->LoadCiphertext(initial);



	//this->context->EvalBootstrapInPlace(initial, 1, 0, false);

	auto b = decrypt(initial);

	std::cout << "Precision: " << std::to_string(compute_approx_error(a, b)) << std::endl;
}

void resnet::convbn(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {

	int img_width = static_cast<int>(resnet::conv_image_widths[layer_idx]);
	int padding	  = 1;

	std::vector<int> indexes = { -padding - img_width, -img_width, padding - img_width, -padding, 0, padding, -padding + img_width, img_width, padding + img_width };

	if (layer_idx == 0) {
		// Use SpecialConvolutionTransform with flattened weights
		size_t bStep = resnet::weights_inner_shape[layer_idx];
		int stride	 = resnet::conv_rotation_indexes[layer_idx];
		size_t gStep = resnet::weights_outer_shape[layer_idx];

		this->context->SpecialConvolutionTransformInPlace(
		  ct, static_cast<int>(gStep), static_cast<int>(bStep), this->weights[layer_idx][0], this->mask, indexes, stride, stride); // stride and maskRotationStride are the same
		// this->print(ct, "Layer " + std::to_string(layer_idx) + " After Special Conv Transform");
	} else {
		// Use pre-flattened weights directly (loaded during weight loading)
		size_t bStep = resnet::weights_inner_shape[layer_idx];
		int stride	 = resnet::conv_rotation_indexes[layer_idx];
		size_t gStep = resnet::weights_outer_shape[layer_idx];

		this->context->ConvolutionTransformInPlace(ct, static_cast<int>(gStep), static_cast<int>(bStep), this->weights[layer_idx][0], indexes, stride);
		// this->print(ct, "Layer " + std::to_string(layer_idx) + " After Conv Transform");
	}

	this->context->EvalAddInPlace(ct, this->bias[layer_idx][0]);

	// this->print(ct, "Layer " + std::to_string(layer_idx) + " After bias");
}

std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> resnet::convbnsx(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {
	int img_width = static_cast<int>(resnet::conv_image_widths[layer_idx]);
	int padding	  = 1;

	std::vector<int> indexes = { -padding - img_width, -img_width, padding - img_width, -padding, 0, padding, -padding + img_width, img_width, padding + img_width };
	int stride = resnet::conv_rotation_indexes[layer_idx];

	std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> results = { nullptr, nullptr };

	size_t bStep	  = resnet::weights_inner_shape[layer_idx];
	size_t half_gStep = resnet::weights_outer_shape[layer_idx] / 2;

	// Process first half (results[0]) using pre-flattened weights
	results[0] = ct->Clone();
	this->context->ConvolutionTransformInPlace(results[0], static_cast<int>(half_gStep), static_cast<int>(bStep), this->weights[layer_idx][0], indexes, stride);
	this->context->EvalAddInPlace(results[0], this->bias[layer_idx][0]);

	// Process second half (results[1]) using pre-flattened weights
	results[1] = ct->Clone();
	this->context->ConvolutionTransformInPlace(results[1], static_cast<int>(half_gStep), static_cast<int>(bStep), this->weights[layer_idx][1], indexes, stride);
	this->context->EvalAddInPlace(results[1], this->bias[layer_idx][1]);

	return results;
}

std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> resnet::convbndx(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>&& ct) {

	std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> results = { nullptr, nullptr };

	// 1x1 convolution: single kernel position (no spatial dimensions)
	std::vector<int> indexes = { 0 }; // No rotation needed for 1x1
	int stride				 = resnet::conv_rotation_indexes[layer_idx];

	// First half: channels 0 to weights_outer_shape/2
	size_t gStep_half = resnet::weights_outer_shape[layer_idx] / 2;
	size_t bStep	  = 1; // 1x1 convolution has bStep = 1

	// Use ConvolutionTransform for first half (results[0])
	results[0] = ct->Clone();
	this->context->ConvolutionTransformInPlace(results[0], static_cast<int>(gStep_half), static_cast<int>(bStep), this->weights[layer_idx][0], indexes, stride);
	this->context->EvalAddInPlace(results[0], this->bias[layer_idx][0]);

	// Use ConvolutionTransform for second half (results[1])
	results[1] = ct->Clone();
	this->context->ConvolutionTransformInPlace(results[1], static_cast<int>(gStep_half), static_cast<int>(bStep), this->weights[layer_idx][1], indexes, stride);
	this->context->EvalAddInPlace(results[1], this->bias[layer_idx][1]);

	return results;
}

fideslib::Ciphertext<fideslib::DCRTPoly> resnet::downsample(size_t downsample_layer, std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2>&& cts) {

	this->context->EvalMultInPlace(cts[0], this->downsample_masks[downsample_layer][0]);
	// this->print(cts[0], std::string("After first multiplication"));
	this->context->EvalMultInPlace(cts[1], this->downsample_masks[downsample_layer][1]);
	// this->print(cts[1], std::string("After second multiplication"));
	this->context->EvalAddInPlace(cts[0], cts[1]);
	// this->print(cts[0], std::string("After first addition"));

	cts[1] = cts[0]->Clone();
	this->context->EvalRotateInPlace(cts[1], 1);
	// this->print(cts[1], std::string("After first rotation"));
	this->context->EvalAddInPlace(cts[0], cts[1]);
	// this->print(cts[0], std::string("After second addition"));

	this->context->EvalMultInPlace(cts[0], this->downsample_masks[downsample_layer][2]);
	// this->print(cts[0], std::string("After second multiplication"));

	cts[1] = cts[0]->Clone();
	this->context->EvalRotateInPlace(cts[1], 2);
	// this->print(cts[1], std::string("After third rotation"));
	this->context->EvalAddInPlace(cts[0], cts[1]);
	// this->print(cts[0], std::string("After third addition"));
	this->context->EvalMultInPlace(cts[0], this->downsample_masks[downsample_layer][3]);
	// this->print(cts[0], std::string("After third multiplication"));

	cts[1] = cts[0]->Clone();
	this->context->EvalRotateInPlace(cts[1], 4);
	// this->print(cts[1], std::string("After fourth rotation"));
	this->context->EvalAddInPlace(cts[0], cts[1]);
	// this->print(cts[0], std::string("After fourth addition"));

	if (downsample_layer == 0) {
		this->context->EvalMultInPlace(cts[0], this->downsample_masks[downsample_layer][4]);
		// this->print(cts[0], std::string("After fourth multiplication"));

		cts[1] = cts[0]->Clone();
		this->context->EvalRotateInPlace(cts[1], 8);
		// this->print(cts[1], std::string("After fifth rotation"));
		this->context->EvalAddInPlace(cts[0], cts[1]);
		// this->print(cts[0], std::string("After fifth addition"));
	}

	auto downsampled_rows = this->zero_ct->Clone();
	size_t its_down		  = (downsample_layer == 0) ? 16 : 32;
	int step_down		  = (downsample_layer == 0) ? (64 - 16) : (32 - 8);
	for (size_t i = 0; i < its_down; i++) {
		auto masked = cts[0]->Clone();
		this->context->EvalMultInPlace(masked, this->downsample_masks[downsample_layer][5 + i]);
		// this->print(masked, "After downsample mask multiplication " + std::to_string(i));
		this->context->EvalAddInPlace(downsampled_rows, masked);
		// this->print(downsampled_rows, "After downsample addition " + std::to_string(i));
		if (i < (its_down - 1)) {
			this->context->EvalRotateInPlace(cts[0], step_down);
			// this->print(cts[0], "After downsample rotation " + std::to_string(i));
		}
	}
	cts[0] = std::move(downsampled_rows);

	auto downsampled_channels = this->zero_ct->Clone();
	size_t its_channels		  = (downsample_layer == 0) ? 32 : 64;
	int step_channels		  = (downsample_layer == 0) ? (-(1024 - 256)) : (-(256 - 64));
	for (size_t i = 0; i < its_channels; i++) {
		auto masked = cts[0]->Clone();
		this->context->EvalMultInPlace(masked, this->downsample_masks[downsample_layer][i + 5 + its_down]);
		// this->print(masked, "After channel downsample mask multiplication " + std::to_string(i));
		this->context->EvalAddInPlace(downsampled_channels, masked);
		// this->print(downsampled_channels, "After channel downsample addition " + std::to_string(i));
		this->context->EvalRotateInPlace(downsampled_channels, step_channels);
		// this->print(downsampled_channels, "After channel downsample rotation " + std::to_string(i));
	}
	cts[0] = std::move(downsampled_channels);

	int final_rotate_1 = (downsample_layer == 0) ? ((1024 - 256) * 32) : ((256 - 64) * 64);
	this->context->EvalRotateInPlace(cts[0], final_rotate_1);
	// this->print(cts[0], "After final rotate 1");
	auto tmp		   = cts[0]->Clone();
	int final_rotate_2 = (downsample_layer == 0) ? (-8192) : (-4096);
	this->context->EvalRotateInPlace(tmp, final_rotate_2);
	// this->print(tmp, "After final rotate 2");
	this->context->EvalAddInPlace(cts[0], tmp);
	// this->print(cts[0], "After final addition");
	tmp				   = cts[0]->Clone();
	int final_rotate_3 = (downsample_layer == 0) ? (-16384) : (-8192);
	this->context->EvalRotateInPlace(tmp, final_rotate_3);
	// this->print(tmp, "After final rotate 3");
	this->context->EvalAddInPlace(cts[0], tmp);
	// this->print(cts[0], "After final addition 2");

	cts[0]->SetSlots(16384);
	this->num_slots = 16384;

	return cts[0];
}

void resnet::relu(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, double scale) {
	auto prescaled = this->prescaled ? this->prescale : 1.0;
	std::function<double(double)> relu = [scale](double x) -> double {
		if (x < 0)
			return 0;
		else
			return (1 / (scale)) * x;
	};
	auto coeffs = this->context->GetChebyshevCoefficients(relu, -1.0, 1.0, this->relu_degree);
	this->context->EvalChebyshevSeriesInPlace(ct, coeffs, -1.0, 1.0);
}

fideslib::Plaintext resnet::mask_from_to(uint32_t from, uint32_t to, uint32_t level) {
	std::vector<double> vec;

	for (size_t i = 0; i < num_slots; i++) {
		if (i >= from && i < to) {
			vec.push_back(1);
		} else {
			vec.push_back(0);
		}
	}

	return this->encode(vec, level, num_slots);
}

fideslib::Plaintext resnet::gen_mask(const uint32_t n, const uint32_t slots, const uint32_t level) {
	std::vector<double> mask_d;

	int32_t copy_interval = static_cast<int32_t>(n);

	for (int32_t i = 0; i < static_cast<int32_t>(slots); i++) {
		if (copy_interval > 0) {
			mask_d.push_back(1);
		} else {
			mask_d.push_back(0);
		}

		copy_interval--;

		if (copy_interval <= -static_cast<int32_t>(n)) {
			copy_interval = static_cast<int32_t>(n);
		}
	}

	return this->encode(mask_d, level, slots);
}

fideslib::Plaintext resnet::mask_first_n(const uint32_t n, const uint32_t slots, const uint32_t level) {
	std::vector<double> mask_d;

	for (size_t i = 0; i < slots; i++) {
		if (i < n) {
			mask_d.push_back(1);
		} else {
			mask_d.push_back(0);
		}
	}

	return this->encode(mask_d, level, slots);
}

fideslib::Plaintext resnet::mask_second_n(const uint32_t n, const uint32_t slots, const uint32_t level) {
	std::vector<double> mask_d;

	for (size_t i = 0; i < slots; i++) {
		if (i >= n) {
			mask_d.push_back(1);
		} else {
			mask_d.push_back(0);
		}
	}

	return this->encode(mask_d, level, slots);
}

fideslib::Plaintext resnet::mask_first_n_mod(const uint32_t n, const uint32_t padding, const uint32_t pos, const uint32_t level) {
	std::vector<double> mask_d;
	for (uint32_t i = 0; i < 32; i++) {
		for (uint32_t j = 0; j < (pos * n); j++) {
			mask_d.push_back(0);
		}
		for (uint32_t j = 0; j < n; j++) {
			mask_d.push_back(1);
		}
		for (uint32_t j = 0; j < (padding - n - (pos * n)); j++) {
			mask_d.push_back(0);
		}
	}

	return this->encode(mask_d, level, 16384 * 2);
}

fideslib::Plaintext resnet::mask_channel(const uint32_t n, const uint32_t level) {
	// The produced mask has a fixed size of 32 * 1024 = 32768 slots.
	// Layout (as in the original code): n blocks of 1024 zeros,
	// then 256 ones, then 768 zeros, then (31-n) blocks of 1024 zeros.
	const size_t total_slots = 32 * 1024;
	std::vector<double> mask_d(total_slots, 0.0);

	// Set the 256 ones starting at offset n * 1024
	const size_t start = static_cast<size_t>(n) * 1024;
	if (start + 256 <= total_slots) {
		for (size_t i = 0; i < 256; ++i) {
			mask_d[start + i] = this->prescale;
		}
	}

	return this->encode(mask_d, level, 16384 * 2);
}

fideslib::Plaintext resnet::mask_first_n_mod2(const uint32_t n, const uint32_t padding, const uint32_t pos, const uint32_t level) {
	// Each of the 64 blocks has length `padding`. Within each block there are
	// pos*n zeros, then n ones, then the remaining zeros. Total size = 64 * padding.
	const size_t blocks = 64;
	const size_t total	= static_cast<size_t>(blocks) * static_cast<size_t>(padding);
	std::vector<double> mask_d(total, 0.0);

	// Fill ones for each block at the correct offset
	for (size_t block = 0; block < blocks; ++block) {
		const size_t offset = block * static_cast<size_t>(padding) + static_cast<size_t>(pos) * static_cast<size_t>(n);
		for (size_t k = 0; k < static_cast<size_t>(n); ++k) {
			if (offset + k < total)
				mask_d[offset + k] = 1;
		}
	}

	return this->encode(mask_d, level, 8192 * 2);
}

fideslib::Plaintext resnet::mask_channel_2(const uint32_t n, const uint32_t level) {
	// Total slots: 64 * 256 = 16384
	const size_t total_slots = 64 * 256;
	std::vector<double> mask_d(total_slots, 0.0);

	// The original layout inserts n blocks of 256 zeros first (already zero),
	// then 64 ones starting at offset n * 256.
	const size_t ones_start = static_cast<size_t>(n) * 256;
	if (ones_start + 64 <= total_slots) {
		for (size_t i = 0; i < 64; ++i) {
			mask_d[ones_start + i] = this->prescale;
		}
	}

	return this->encode(mask_d, level, 8192 * 2);
}

fideslib::Plaintext resnet::mask_mod(const uint32_t n, const uint32_t level, const double custom_val) {
	// Preallocate and fill by index. Protect against n == 0 to avoid UB.
	const size_t total = static_cast<size_t>(num_slots);
	std::vector<double> vec(total, 0.0);
	if (n == 0)
		return encode(vec, level, num_slots);

	for (size_t i = 0; i < total; i += static_cast<size_t>(n)) {
		vec[i] = custom_val;
	}

	return encode(vec, level, num_slots);
}

void resnet::print(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, std::string msg) {

	if (!this->verbose) {
		return;
	}

	std::cout << msg << std::endl;
	std::cout << "\tLevel " << ct->GetLevel() << " , Scale " << ct->GetNoiseScaleDeg() << std::endl;
	fideslib::Plaintext result = this->decrypt(ct);
	//std::cout << "\t" << result << std::endl;
}