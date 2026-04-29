#ifndef RESNET_HPP
#define RESNET_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <fideslib/Definitions.hpp>
#include <vector>

#include "experiments.hpp"

#include <fideslib/fideslib.hpp>

class resnet {
  private:
	// 16384 slots.
	uint32_t num_slots				   = 1 << 14;
	std::vector<uint32_t> level_budget = { 3, 3 };
	uint32_t relu_degree			   = 119;
	std::string parameters_folder	   = "NO_FOLDER";
	uint32_t circuit_depth			   = 0;
	bool verbose					   = false;

	fideslib::CryptoContext<fideslib::DCRTPoly> context;
	fideslib::KeyPair<fideslib::DCRTPoly> key_pair;

	// Devices configuration.
	std::vector<int> devices   = { 0 };
	bool auto_load_plaintexts  = false;
	bool auto_load_ciphertexts = true;

	// Special prescale optimization.
	bool prescaled = false;

	// Rotations needed for ResNet-20 inference.
	std::vector<int> rotations = {
		// Kernel rotations.
		1,
		-1,
		7,
		-7,
		8,
		-8,
		9,
		-9,
		15,
		-15,
		16,
		-16,
		17,
		-17,
		31,
		-31,
		32,
		-32,
		33,
		-33,
		// Convolution rotations.
		-64,
		-128,
		-256,
		-512,
		-1024,
		-2048,
		-4096,
		-8192,
		// Downsample rotations.
		-768,
		24,
		-192,
		// Average pool rotations.
		768
		
	};

	// Weights for ResNet-20 layers.
	static constexpr size_t num_layers = 21;
	static constexpr size_t block_size = 8;
	static constexpr std::array<size_t, num_layers> weights_outer_shape = { 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64 };
	static constexpr std::array<size_t, num_layers> weights_inner_shape = { 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9 };
	static constexpr std::array<double, num_layers> scales = { 0.90, 1.00, 0.52, 0.55, 0.36, 0.63, 0.42, 0.57, 0.40, 0.40, 0.76, 0.37, 0.63, 0.25, 0.63, 0.40, 0.40, 0.57, 0.33, 0.69, 0.10 };

	std::array<std::vector<std::vector<fideslib::Plaintext>>, num_layers> weights;
	std::array<std::vector<fideslib::Plaintext>, num_layers> bias;
	fideslib::Plaintext weight_final = nullptr;

	fideslib::Plaintext mask = nullptr;
	std::array<std::vector<fideslib::Plaintext>, 2> downsample_masks;
	fideslib::Plaintext final_mask = nullptr;

	// Convolution image widths for each layer.
	static constexpr std::array<size_t, num_layers> conv_image_widths = { 32, 32, 32, 32, 32, 32, 32, 32, 0, 16, 16, 16, 16, 16, 16, 0, 8, 8, 8, 8, 8 };
	// Convolution aux rotation indexes for each layer.
	static constexpr std::array<int32_t, num_layers>
	  conv_rotation_indexes = { 1024, -1024, -1024, -1024, -1024, -1024, -1024, -1024, -1024, -256, -256, -256, -256, -256, -256, -256, -64, -64, -64, -64, -64 };

	// Prescale factor.
	double prescale = 1.0;

	// Functions for each layer.
	bool partial_load								 = false;
	std::array<std::function<void()>, 22> load_funcs = { [this]() { this->load_weights_layer0(); },
		[this]() { this->load_weights_layer1(); },
		[this]() { this->load_weights_layer2(); },
		[this]() { this->load_weights_layer3(); },
		[this]() { this->load_weights_layer4(); },
		[this]() { this->load_weights_layer5(); },
		[this]() { this->load_weights_layer6(); },
		[this]() { this->load_weights_layer7(); },
		[this]() { this->load_weights_layer8(); },
		[this]() { this->load_weights_layer9(); },
		[this]() { this->load_weights_layer10(); },
		[this]() { this->load_weights_layer11(); },
		[this]() { this->load_weights_layer12(); },
		[this]() { this->load_weights_layer13(); },
		[this]() { this->load_weights_layer14(); },
		[this]() { this->load_weights_layer15(); },
		[this]() { this->load_weights_layer16(); },
		[this]() { this->load_weights_layer17(); },
		[this]() { this->load_weights_layer18(); },
		[this]() { this->load_weights_layer19(); },
		[this]() { this->load_weights_layer20(); },
		[this]() { this->load_weights_final(); } };

	void load_weights_layer0();
	void load_weights_layer1();
	void load_weights_layer2();
	void load_weights_layer3();
	void load_weights_layer4();
	void load_weights_layer5();
	void load_weights_layer6();
	void load_weights_layer7();
	void load_weights_layer8();
	void load_weights_layer9();
	void load_weights_layer10();
	void load_weights_layer11();
	void load_weights_layer12();
	void load_weights_layer13();
	void load_weights_layer14();
	void load_weights_layer15();
	void load_weights_layer16();
	void load_weights_layer17();
	void load_weights_layer18();
	void load_weights_layer19();
	void load_weights_layer20();
	void load_weights_final();

	// Zero CT for masking.
	fideslib::Ciphertext<fideslib::DCRTPoly> zero_ct;

  public:
	void generate_context(experiment_settings e);
	void generate_rotations(experiment_settings e);
	void generate_bootstrapping(experiment_settings e);
	void generate_precomputations(experiment_settings e);
	void serialize_context(experiment_settings e);
	void deserialize_context(experiment_settings e);

	void load_context();
	void load_weights();

	fideslib::Plaintext encode(const std::vector<double>& in, uint32_t level, uint32_t plaintext_slots);
	fideslib::Ciphertext<fideslib::DCRTPoly> encrypt(const std::vector<double>& in, uint32_t level = 0, uint32_t plaintext_slots = 0);
	fideslib::Plaintext decrypt(fideslib::Ciphertext<fideslib::DCRTPoly>& ct);

	void bootstrap(fideslib::Ciphertext<fideslib::DCRTPoly>& ct);
	void relu(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, double scale);

	size_t execute_resnet_inference(const std::string& input_image);

	void measure_bootstrap_precision();

	fideslib::Plaintext mask_from_to(uint32_t from, uint32_t to, uint32_t level);
	fideslib::Plaintext gen_mask(uint32_t n, uint32_t slots, uint32_t level);
	fideslib::Plaintext mask_first_n(uint32_t n, uint32_t slots, uint32_t level);
	fideslib::Plaintext mask_second_n(uint32_t n, uint32_t slots, uint32_t level);
	fideslib::Plaintext mask_first_n_mod(uint32_t n, uint32_t padding, uint32_t pos, uint32_t level);
	fideslib::Plaintext mask_first_n_mod2(uint32_t n, uint32_t padding, uint32_t pos, uint32_t level);
	fideslib::Plaintext mask_channel(uint32_t n, uint32_t level);
	fideslib::Plaintext mask_channel_2(uint32_t n, uint32_t level);
	fideslib::Plaintext mask_mod(uint32_t n, uint32_t level, double custom_val);

	void convbn(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>& ct);
	std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> convbnsx(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>& ct);
	std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2> convbndx(size_t layer_idx, fideslib::Ciphertext<fideslib::DCRTPoly>&& ct);
	fideslib::Ciphertext<fideslib::DCRTPoly> downsample(size_t downsample_layer, std::array<fideslib::Ciphertext<fideslib::DCRTPoly>, 2>&& cts);

	void print(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, std::string msg);
};

#endif // RESNET_HPP