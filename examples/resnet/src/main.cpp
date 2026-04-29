#include <iostream>
#include <unistd.h>
#include "utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"
#define RESET_COLOR "\033[0m"



#include "controller.hpp"
#include "experiments.hpp"
#include "args.hpp"

int main(int argc, char* argv[]) {

	// Parse arguments.
	args program_args = check_arguments(argc, argv);

	// Controller for ResNet.
	resnet resnet_controller;

	std::cout << "----- ResNet-20 Experiment " << program_args.context_index << " -----" << std::endl;

	// Get experiment settings.
	experiment_settings settings = get_experiment_settings(program_args.context_index);
	settings.verbose = program_args.verbose;
		
	// Generate context for the experiment.
	resnet_controller.generate_context(settings);
	// Generate rotation keys and bootstrapping keys.
	resnet_controller.generate_rotations(settings);
	// Generate bootstrapping keys.
	resnet_controller.generate_bootstrapping(settings);

	// Into GPU mode.
	resnet_controller.load_context();
	// Get input image.
	 if (program_args.input_filename.empty()) {
        program_args.input_filename = "../inputs/luis.png";
        if (program_args.verbose) std::cout << "You did not set any input, I use " << GREEN_TEXT << "../inputs/luis.png" << RESET_COLOR << "." << std::endl;
    } else {
        if (program_args.verbose) std::cout << "I am going to encrypt and classify " << GREEN_TEXT << program_args.input_filename << RESET_COLOR << "." << std::endl;
    }

	// Execute ResNet-20 inference.
	auto class_index = resnet_controller.execute_resnet_inference(program_args.input_filename);

	std::cout << "The input image is classified as " << GREEN_TEXT << get_class(class_index) << RESET_COLOR << std::endl;
}
