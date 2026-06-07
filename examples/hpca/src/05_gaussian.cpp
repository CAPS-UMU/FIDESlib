#include <fideslib.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace fideslib;

// =====================================================
// FHE Gaussian Blur using Separable Filter.
//
// ALGORITHM (Plaintext):
// -----------------------
// A 2D Gaussian blur can be decomposed into two 1D passes
// (separable filter property):
//
//   G_2D(x,y) = G_1D(x) * G_1D(y)
//
// For an NxN kernel with weights [w0, w1, ..., w_{N-1}] (symmetric):
// Let H = N/2 (halfKernel). The center weight is w_H.
//
//   Horizontal pass:
//     H[i] = sum_{k=-H}^{H} w_{H+k} * P[i+k]
//          = w_H*P[i] + sum_{k=1}^{H} w_{H-k}*(P[i-k] + P[i+k])
//
//   Vertical pass:
//     V[i] = sum_{k=-H}^{H} w_{H+k} * H[i+k*W]
//          = w_H*H[i] + sum_{k=1}^{H} w_{H-k}*(H[i-k*W] + H[i+k*W])
//
// where W = row width, and i is the linear pixel index.
//
// FHE TRANSLATION:
// ----------------
// In CKKS, pixels are packed into slots. Rotation shifts slots:
//   - Rotate(ct, +k) shifts slot[i] <- slot[i+k] (left neighbor by k)
//   - Rotate(ct, -k) shifts slot[i] <- slot[i-k] (right neighbor by k)
//   - Rotate(ct, +k*W) shifts slot[i] <- slot[i+k*W] (k rows below)
//   - Rotate(ct, -k*W) shifts slot[i] <- slot[i-k*W] (k rows above)
//
// Horizontal pass in FHE:
//   ctH = w_H*ct + sum_{k=1}^{H} w_{H-k}*(Rot(ct,+k) + Rot(ct,-k))
//
// Vertical pass in FHE:
//   ctV = w_H*ctH + sum_{k=1}^{H} w_{H-k}*(Rot(ctH,+k*W) + Rot(ctH,-k*W))
//
// OPTIMIZATIONS:
// --------------
// 1. Hoisted rotations: Precompute digit decomposition once,
//    then apply all rotations with shared precomputation.
//
// 2. Symmetric kernel: Since w_{H-k} == w_{H+k} for Gaussian,
//    we add Rot(+k) + Rot(-k) first, then multiply by w_{H-k}.
//    This halves the number of ct-scalar multiplications.
//
// KERNEL STORAGE:
// ---------------
// Because we use a SEPARABLE filter, the NxN 2D kernel is decomposed
// into two 1D kernels of length N:
//
//   2D kernel (NxN weights)    =>    1D kernel (N weights)
//                                    applied twice:
//                                    horizontal then vertical
//
// The 1D kernel is stored as:
//   kernel1d = [w0, w1, ..., w_{N-1}] where w_{H} is the center weight.
//   Index:      0   1   ...  N-1
//              -H  ...  0   ... +H  (relative position)
//
// Access pattern:
//   - kernel1d[halfKernel]     = center weight (w_H)
//   - kernel1d[halfKernel - k] = weight for offset ±k (symmetric)
//
// =====================================================
Ciphertext<DCRTPoly>
fheGaussianBlur(CryptoContext<DCRTPoly>& cc, const Ciphertext<DCRTPoly>& ctTile, int tileWidth,
				const std::vector<double>& kernel1d, const std::vector<int32_t>& hRotIndices, const std::vector<int32_t>& vRotIndices) {

	// TODO: Get the kernel size, the center weight, and the cyclotomic order.
	
	// ----- Horizontal pass -----

	// TODO: Generate all rotations needed for the horizontal pass.

	// TODO: Compute the center term: ct * kernel[center].

	// TODO: Compute the symmetric terms: (Rot(+k) + Rot(-k)) * w[k]. Accumulate the results.
	
	// ----- Vertical pass -----

	// TODO: Generate all rotations needed for the vertical pass.

	// TODO: Compute the center term: ct * kernel[center].

	// TODO: Compute the symmetric terms: (Rot(+k) + Rot(-k)) * w[k]. Accumulate the results.

	return ctTile;
}

// =====================================================
// Generate rotation indices for FHE Gaussian blur.
//
// This function computes the rotation indices needed for the horizontal
// and vertical passes of the separable Gaussian blur:
//   - Horizontal rotation indices: {+1, -1, +2, -2, ...} for neighbor access
//   - Vertical rotation indices: {+W, -W, +2W, -2W, ...} for row access
//
// Parameters:
//   - kernelSize: Size of the Gaussian kernel (must be odd)
//   - tileWidth: Width of the tile (used for vertical stride)
//   - hRotIndices: Output vector for horizontal rotation indices
//   - vRotIndices: Output vector for vertical rotation indices
// =====================================================
void generateRotationIndices(int kernelSize, int tileWidth,
							 std::vector<int32_t>& hRotIndices,
							 std::vector<int32_t>& vRotIndices) {
	int halfKernel = kernelSize / 2;

	// Horizontal rotation indices: {+1, -1, +2, -2, ...}
	hRotIndices.clear();
	for (int k = 1; k <= halfKernel; ++k) {
		hRotIndices.push_back(k);  // Left neighbor.
		hRotIndices.push_back(-k); // Right neighbor.
	}

	// Vertical rotation indices: {+W, -W, +2W, -2W, ...}
	vRotIndices.clear();
	for (int k = 1; k <= halfKernel; ++k) {
		vRotIndices.push_back(k * tileWidth);  // Row below.
		vRotIndices.push_back(-k * tileWidth); // Row above.
	}
}

// =====================================================
// Compute 1D Gaussian kernel.
// This function generates the kernel values based on the sigma and kernel
// size (size x size).
// =====================================================
std::vector<double> computeGaussianKernel1D(double sigma, int size) {
	std::vector<double> kernel(size);
	int halfSize = size / 2;
	double sum	 = 0.0;

	for (int i = 0; i < size; ++i) {
		int x	  = i - halfSize;
		kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
		sum += kernel[i];
	}

	// Normalize.
	for (int i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}

	return kernel;
}

// =====================================================
// Image structure.
// =====================================================
struct Image {
	std::vector<uint8_t> data;
	int width;
	int height;
	int channels;
};

// =====================================================
// Load image using stb.
// =====================================================
Image loadImage(const std::string& filename) {
	Image img;
	int w, h, c;
	uint8_t* data = stbi_load(filename.c_str(), &w, &h, &c, 1); // Load as grayscale.
	if (!data) {
		std::cerr << "Failed to load image: " << filename << std::endl;
		exit(1);
	}
	img.width	 = w;
	img.height	 = h;
	img.channels = 1;
	img.data.assign(data, data + w * h);
	stbi_image_free(data);
	return img;
}

// =====================================================
// Save image using stb.
// =====================================================
void saveImage(const std::string& filename, const Image& img) {
	stbi_write_png(filename.c_str(), img.width, img.height, img.channels, img.data.data(), img.width * img.channels);
}

// =====================================================
// Extract a tile from the image with border padding.
// =====================================================
std::vector<double> extractTile(const Image& img, int tileX, int tileY, int tileWidth, int tileHeight, int border) {
	int totalWidth	= tileWidth + 2 * border;
	int totalHeight = tileHeight + 2 * border;
	std::vector<double> tile(totalWidth * totalHeight);

	int startX = tileX * tileWidth - border;
	int startY = tileY * tileHeight - border;

	for (int y = 0; y < totalHeight; ++y) {
		for (int x = 0; x < totalWidth; ++x) {
			int imgX = startX + x;
			int imgY = startY + y;

			// Clamp to image bounds.
			imgX = std::max(0, std::min(imgX, img.width - 1));
			imgY = std::max(0, std::min(imgY, img.height - 1));

			tile[y * totalWidth + x] = static_cast<double>(img.data[imgY * img.width + imgX]) / 255.0;
		}
	}

	return tile;
}

// =====================================================
// Insert a tile back into the image (without border).
// =====================================================
void insertTile(Image& img, const std::vector<double>& tile, int tileX, int tileY, int tileWidth, int tileHeight, int border) {
	int totalWidth = tileWidth + 2 * border;

	int startX = tileX * tileWidth;
	int startY = tileY * tileHeight;

	for (int y = 0; y < tileHeight; ++y) {
		for (int x = 0; x < tileWidth; ++x) {
			int imgX = startX + x;
			int imgY = startY + y;

			if (imgX < img.width && imgY < img.height) {
				int tileIdx						  = (y + border) * totalWidth + (x + border);
				double val						  = tile[tileIdx];
				val								  = std::max(0.0, std::min(1.0, val));
				img.data[imgY * img.width + imgX] = static_cast<uint8_t>(val * 255.0);
			}
		}
	}
}

// =====================================================
// Apply horizontal convolution. This is the reference implementation.
// =====================================================
std::vector<double> applyHorizontalConv(const std::vector<double>& tile, int width, int height, const std::vector<double>& kernel) {
	int kernelSize = static_cast<int>(kernel.size());
	int halfKernel = kernelSize / 2;
	std::vector<double> result(width * height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double sum = 0.0;
			for (int k = 0; k < kernelSize; ++k) {
				int srcX = std::max(0, std::min(x + k - halfKernel, width - 1));
				sum += tile[y * width + srcX] * kernel[k];
			}
			result[y * width + x] = sum;
		}
	}

	return result;
}

// =====================================================
// Apply vertical convolution. This is the reference implementation.
// =====================================================
std::vector<double> applyVerticalConv(const std::vector<double>& tile, int width, int height, const std::vector<double>& kernel) {
	int kernelSize = static_cast<int>(kernel.size());
	int halfKernel = kernelSize / 2;
	std::vector<double> result(width * height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double sum = 0.0;
			for (int k = 0; k < kernelSize; ++k) {
				int srcY = std::max(0, std::min(y + k - halfKernel, height - 1));
				sum += tile[srcY * width + x] * kernel[k];
			}
			result[y * width + x] = sum;
		}
	}

	return result;
}

int main(int argc, char* argv[]) {
	std::cout << "FHE Gaussian Blur Filter" << std::endl;

	// Default parameters for the filter.
	std::string inputFile  = "input.png";
	std::string outputFile = "output.png";
	double sigma		   = 10.0; // Sigma controls the amount of blur.
	int kernelSize		   = 21;   // Kernel size controls the size of the blur.
	int tileSize		   = 64; // Tile dimension. Must divide the image in order to fit the ciphertext.

	if (argc >= 2)
		inputFile = argv[1];
	if (argc >= 3)
		outputFile = argv[2];
	if (argc >= 4)
		sigma = std::stod(argv[3]);

	std::cout << "Input:  " << inputFile << std::endl;
	std::cout << "Output: " << outputFile << std::endl;
	std::cout << "Sigma:  " << sigma << std::endl;
	std::cout << "Kernel: " << kernelSize << "x" << kernelSize << std::endl;

	// Compute 1D Gaussian kernel.
	auto kernel1d = computeGaussianKernel1D(sigma, kernelSize);
	std::cout << "1D Kernel: ";
	for (const auto& k : kernel1d)
		std::cout << std::fixed << std::setprecision(4) << k << " ";
	std::cout << std::endl;

	// Load input image.
	Image img = loadImage(inputFile);
	std::cout << "Image size: " << img.width << "x" << img.height << std::endl;

	// Output image.
	Image outputImg;
	outputImg.width	   = img.width;
	outputImg.height   = img.height;
	outputImg.channels = 1;
	outputImg.data.resize(img.width * img.height);

	// Calculate tiles.
	int border		   = kernelSize / 2;
	int tileWithBorder = tileSize + 2 * border;
	tileWithBorder	   = 1 << static_cast<int>(std::floor(std::log2(tileWithBorder)));
	tileSize		   = tileWithBorder - 2 * border;
	int tilesX		   = (img.width + tileSize - 1) / tileSize;
	int tilesY		   = (img.height + tileSize - 1) / tileSize;
	int totalTiles	   = tilesX * tilesY;
	int tilePixels	   = tileWithBorder * tileWithBorder;

	std::cout << "Tiles: " << tilesX << "x" << tilesY << " = " << totalTiles << std::endl;
	std::cout << "Tile size with border: " << tileWithBorder << "x" << tileWithBorder << std::endl;
	std::cout << "Pixels per tile: " << tilePixels << std::endl;

	// =====================================================
	// Setup FHE context.
	// =====================================================

	uint32_t batchSize			 = tilePixels;
	uint32_t multDepth			 = 4;
	uint32_t ring_dim			 = 1 << 14;
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;

	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;
	uint32_t dnum	  = 3;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
	parameters.SetRingDim(ring_dim);
	parameters.SetMultiplicativeDepth(multDepth);
	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBatchSize(batchSize);
	parameters.SetBackend(Backend::CUDA);
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "Ring dimension: " << cc->GetRingDimension() << std::endl;

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	// Rotation keys for horizontal (1, -1) and vertical (row offsets).
	std::vector<int32_t> rotIndices;
	for (int k = 1; k <= kernelSize / 2; ++k) {
		rotIndices.push_back(k);
		rotIndices.push_back(-k);
		rotIndices.push_back(k * tileWithBorder);
		rotIndices.push_back(-k * tileWithBorder);
	}
	cc->EvalRotateKeyGen(keys.secretKey, rotIndices);

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Precompute rotation indices for blur kernel.
	// =====================================================
	std::vector<int32_t> hRotIndices;
	std::vector<int32_t> vRotIndices;
	generateRotationIndices(kernelSize, tileWithBorder, hRotIndices, vRotIndices);

	// =====================================================
	// Extract and pre-encrypt all tiles.
	// =====================================================
	std::cout << "Encrypting " << totalTiles << " tiles..." << std::endl;

	std::vector<std::vector<double>> tiles(totalTiles);
	std::vector<Ciphertext<DCRTPoly>> encryptedTiles(totalTiles);

	for (int ty = 0; ty < tilesY; ++ty) {
		for (int tx = 0; tx < tilesX; ++tx) {
			int idx	   = ty * tilesX + tx;
			tiles[idx] = extractTile(img, tx, ty, tileSize, tileSize, border);
			Plaintext ptxt = cc->MakeCKKSPackedPlaintext(tiles[idx]);
			encryptedTiles[idx] = cc->Encrypt(keys.publicKey, ptxt);
		}
	}
	cc->Synchronize();
	std::cout << "Encryption complete." << std::endl;

	// =====================================================
	// Process tiles.
	// =====================================================
	std::cout << "Processing " << totalTiles << " tiles..." << std::endl;

	std::vector<Ciphertext<DCRTPoly>> blurredCiphertexts(totalTiles);

	auto startTime = std::chrono::high_resolution_clock::now();

	for (int ty = 0; ty < tilesY; ++ty) {
		for (int tx = 0; tx < tilesX; ++tx) {
			int idx					= ty * tilesX + tx;
			blurredCiphertexts[idx] = fheGaussianBlur(cc, encryptedTiles[idx], tileWithBorder, kernel1d, hRotIndices, vRotIndices);
		}
	}

	cc->Synchronize();
	auto endTime						= std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = endTime - startTime;

	std::cout << "FHE computation time: " << std::fixed << std::setprecision(2) << totalTime.count() << " s" << std::endl;

	// =====================================================
	// Decrypt results and insert into output image.
	// =====================================================
	std::cout << "Decrypting results..." << std::endl;

	for (int ty = 0; ty < tilesY; ++ty) {
		for (int tx = 0; tx < tilesX; ++tx) {
			int idx = ty * tilesX + tx;
			Plaintext ptxtResult;
			cc->Decrypt(keys.secretKey, blurredCiphertexts[idx], &ptxtResult);
			ptxtResult->SetLength(tiles[idx].size());
			auto blurredTile = ptxtResult->GetRealPackedValue();
			insertTile(outputImg, blurredTile, tx, ty, tileSize, tileSize, border);
		}
	}

	// Save output image.
	saveImage(outputFile, outputImg);
	std::cout << "Output saved to: " << outputFile << std::endl;

	return 0;
}
