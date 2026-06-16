#include <fideslib.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

// Sigmoid function (LR activation) for reference.
// Definition: sigma(x) = 1 / (1 + e^(-x)).
double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

// Manual polynomial approximation of sigmoid.
// Uses the degree-3 Taylor-like approximation.
Ciphertext<DCRTPoly> evaluateManualSigmoid(CryptoContext<DCRTPoly>& cc, const Ciphertext<DCRTPoly>& ctX) {
	// Coefficients for the degree-3 approximation of sigmoid.
	// Definition: sigma(x) ~= 0.5 + 0.25*x - (1/48)*x^3

	// TODO: Implement manual sigmoid evaluation.
	// Hint: There exists an efficient EvalSquare function.
		// Compute x^2 = x * x.
	auto ctX2 = cc->EvalSquare(ctX);

	// Compute x^3 = x^2 * x.
	auto ctX3 = cc->EvalMult(ctX2, ctX);

	// Compute 0.25 * x.
	auto term1 = cc->EvalMult(ctX, 0.25);

	// Compute -(1/48) * x^3 = -0.0208333... * x^3.
	auto term2 = cc->EvalMult(ctX3, -1.0 / 48.0);

	// Compute 0.25*x - (1/48)*x^3.
	auto sumTerms = cc->EvalAdd(term1, term2);

	// Add 0.5 constant.
	auto result = cc->EvalAdd(sumTerms, 0.5);

	return result;
}

// Chebyshev polynomial approximation of sigmoid.
Ciphertext<DCRTPoly> evaluateChebyshev(CryptoContext<DCRTPoly>& cc, const Ciphertext<DCRTPoly>& ctX, double lowerBound, double upperBound, size_t degree) {
	// Get Chebyshev coefficients for the sigmoid function.
	std::function<double(double)> sigmoidFunc = sigmoid;
	auto coeffs								  = cc->GetChebyshevCoefficients(sigmoidFunc, lowerBound, upperBound, degree);

	// Evaluate using Chebyshev series.
	auto result = cc->EvalChebyshevSeries(ctX, coeffs, lowerBound, upperBound);

	return result;
}

int main() {
	// =====================================================
	// Step 1: Define our parameters.
	// =====================================================

	uint32_t multDepth			 = 8;
	uint32_t batchSize			 = 16;
	uint32_t ring_dim			 = 1 << 12;
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

	// =====================================================
	// Step 2: Generate the CryptoContext.
	// =====================================================

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "CKKS scheme using ring dimension: " << cc->GetRingDimension() << std::endl << std::endl;

	// =====================================================
	// Step 3: Key Generation.
	// =====================================================

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	// =====================================================
	// Step 4: Load the context on the GPU.
	// =====================================================

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Step 5: Create input data.
	// =====================================================

	std::vector<double> xValues;
	double lowerBound = -2.0;
	double upperBound = 2.0;
	size_t numPoints  = batchSize;

	for (size_t i = 0; i < numPoints; ++i) {
		double x = lowerBound + (upperBound - lowerBound) * static_cast<double>(i) / (numPoints - 1);
		xValues.push_back(x);
	}

	std::cout << "\nInput x values: ";
	for (const auto& x : xValues) {
		std::cout << std::fixed << std::setprecision(2) << x << " ";
	}
	std::cout << std::endl;

	// Compute expected sigmoid values
	std::vector<double> expectedY;
	for (const auto& x : xValues) {
		expectedY.push_back(sigmoid(x));
	}

	// =====================================================
	// Step 6: Encrypt the input.
	// =====================================================

	Plaintext ptxtX = cc->MakeCKKSPackedPlaintext(xValues);
	auto ctX		= cc->Encrypt(keys.publicKey, ptxtX);

	// =====================================================
	// Step 7: Manual sigmoid polynomial evaluation.
	// =====================================================

	auto ctManualResult = evaluateManualSigmoid(cc, ctX);
	std::cout << "Manual sigmoid evaluation." << std::endl;
	std::cout << "\t Input level: " << ctX->GetLevel() << std::endl;
	std::cout << "\t Output level: " << ctManualResult->GetLevel() << std::endl;

	Plaintext ptxtManualResult;
	cc->Decrypt(keys.secretKey, ctManualResult, &ptxtManualResult);
	ptxtManualResult->SetLength(numPoints);
	auto manualResults = ptxtManualResult->GetRealPackedValue();

	// =====================================================
	// Step 8: Chebyshev polynomial evaluation.
	// =====================================================

	// Set degree of chebyshev polynomial. 
	// NOTE: You can try from 3 up to 25.
	size_t chebyDegree = 9;
	auto ctChebResult  = evaluateChebyshev(cc, ctX, lowerBound, upperBound, chebyDegree);
	std::cout << "Chebyshev evaluation." << std::endl;
	std::cout << "\t Input level: " << ctX->GetLevel() << std::endl;
	std::cout << "\t Output level: " << ctChebResult->GetLevel() << std::endl;

	Plaintext ptxtChebResult;
	cc->Decrypt(keys.secretKey, ctChebResult, &ptxtChebResult);
	ptxtChebResult->SetLength(numPoints);
	auto chebResults = ptxtChebResult->GetRealPackedValue();

	// =====================================================
	// Step 9: Output results.
	// =====================================================

	std::cout << std::endl << "==== Results Comparison ====" << std::endl << std::endl;

	std::cout << std::setw(10) << "x" << std::setw(12) << "Expected" << std::setw(12) << "Manual" << std::setw(12) << "Chebyshev" << std::setw(12)
			  << "Err Manual" << std::setw(12) << "Err Cheb" << std::endl;
	std::cout << std::string(70, '-') << std::endl;

	for (size_t i = 0; i < numPoints; ++i) {
		double errManual = std::abs(manualResults[i] - expectedY[i]);
		double errCheb	 = std::abs(chebResults[i] - expectedY[i]);

		std::cout << std::fixed << std::setprecision(4) << std::setw(10) << xValues[i] << std::setw(12) << expectedY[i] << std::setw(12) << manualResults[i]
				  << std::setw(12) << chebResults[i] << std::setw(12) << errManual << std::setw(12) << errCheb << std::endl;
	}

	// =====================================================
	// Step 10: Export to CSV for graphing.
	// =====================================================

	std::ofstream csvFile("polynomial_results.csv");
	if (csvFile.is_open()) {
		csvFile << "x,expected,manual,chebyshev\n";
		for (size_t i = 0; i < numPoints; ++i) {
			csvFile << std::fixed << std::setprecision(6) << xValues[i] << "," << expectedY[i] << "," << manualResults[i] << "," << chebResults[i] << "\n";
		}
		csvFile.close();
		std::cout << "\nResults exported to polynomial_results.csv" << std::endl;
	} else {
		std::cerr << "\nError: Could not open CSV file for writing." << std::endl;
	}

	std::cout << "\nPrecision estimates:" << std::endl;
	std::cout << "  Manual result: " << ptxtManualResult->GetLogPrecision() << " bits" << std::endl;
	std::cout << "  Chebyshev result: " << ptxtChebResult->GetLogPrecision() << " bits" << std::endl;

	return 0;
}