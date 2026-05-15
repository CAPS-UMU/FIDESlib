//
// Created by seyda on 5/19/25.
//
#include "PolyApprox.cuh"

// Softmax
std::vector<double> cheb_coeff_exp_softmax_1   = { 2.53213,
	  1.13032,
	  0.271495,
	  0.0443368,
	  0.00547424,
	  0.000542926,
	  4.49773e-05,
	  3.19844e-06,
	  1.99212e-07,
	  1.10368e-08,
	  5.5059e-10,
	  2.49795e-11,
	  1.03914e-12,
	  3.99088e-14,
	  1.76514e-15,
	  1.96862e-16,
	  -1.32079e-17,
	  1.40569e-16,
	  1.23913e-16,
	  -1.43947e-16,
	  3.08643e-17,
	  -2.166e-16,
	  -3.91206e-17,
	  -3.18494e-16,
	  -6.44747e-16,
	  -8.80072e-17,
	  1.77988e-16,
	  7.41657e-16 };
std::vector<double> cheb_coeff_inv_softmax1_27 = { 5.44283,
	0.596525,
	-0.241373,
	0.141981,
	-0.0968308,
	0.0715189,
	-0.0555137,
	0.0445659,
	-0.0366489,
	0.03068,
	-0.0260309,
	0.0223132,
	-0.0192746,
	0.0167442,
	-0.0146025,
	0.0127633,
	-0.011163,
	0.00975347,
	-0.0084977,
	0.00736661,
	-0.00633693,
	0.00538976,
	-0.00450947,
	0.00368287,
	-0.00289863,
	0.00214682,
	-0.0014185,
	0.000705484 };
std::vector<double> cheb_coeff_inv_softmax2_27 = { 0.302733,
	-0.277884,
	0.218387,
	-0.150772,
	0.0936511,
	-0.0533594,
	0.0283131,
	-0.0141566,
	0.00673172,
	-0.00306667,
	0.00134625,
	-0.000572233,
	0.000236433,
	-9.52668e-05,
	3.75372e-05,
	-1.44967e-05,
	5.49824e-06,
	-2.05145e-06,
	7.54085e-07,
	-2.73437e-07,
	9.79177e-08,
	-3.46627e-08,
	1.21406e-08,
	-4.21048e-09,
	1.44675e-09,
	-4.92363e-10,
	1.64628e-10,
	-4.97485e-11 };
std::vector<double> cheb_coeff_inv_softmax1_59 = { 5.44052,
	0.598837,
	-0.243693,
	0.144314,
	-0.0991835,
	0.0738965,
	-0.0579219,
	0.0470107,
	-0.0391365,
	0.0332167,
	-0.0286235,
	0.0249686,
	-0.0220001,
	0.0195477,
	-0.0174923,
	0.0157481,
	-0.0142523,
	0.0129574,
	-0.0118271,
	0.0108333,
	-0.00995367,
	0.00917049,
	-0.00846942,
	0.00783873,
	-0.00726877,
	0.00675154,
	-0.00628033,
	0.00584947,
	-0.00545416,
	0.00509029,
	-0.00475434,
	0.00444326,
	-0.0041544,
	0.00388546,
	-0.0036344,
	0.00339944,
	-0.00317901,
	0.00297172,
	-0.0027763,
	0.00259164,
	-0.00241675,
	0.00225072,
	-0.00209273,
	0.00194203,
	-0.00179796,
	0.00165988,
	-0.00152724,
	0.00139951,
	-0.00127619,
	0.00115685,
	-0.00104105,
	0.000928406,
	-0.000818538,
	0.000711092,
	-0.000605733,
	0.000502137,
	-0.000399992,
	0.000298997,
	-0.000198859,
	9.9288e-05 };
std::vector<double> cheb_coeff_inv_softmax2_59 = { 0.302733,
	-0.277884,
	0.218387,
	-0.150772,
	0.0936511,
	-0.0533594,
	0.0283131,
	-0.0141566,
	0.00673172,
	-0.00306667,
	0.00134625,
	-0.000572233,
	0.000236433,
	-9.52668e-05,
	3.75372e-05,
	-1.44967e-05,
	5.49824e-06,
	-2.05145e-06,
	7.54085e-07,
	-2.73437e-07,
	9.79177e-08,
	-3.46627e-08,
	1.21406e-08,
	-4.21055e-09,
	1.44696e-09,
	-4.93026e-10,
	1.66656e-10,
	-5.59158e-11,
	1.86297e-11,
	-6.16617e-12,
	2.02789e-12,
	-6.62961e-13,
	2.15683e-13,
	-6.96597e-14,
	2.25494e-14,
	-6.91174e-15,
	2.09602e-15,
	-9.18605e-16,
	-1.9514e-16,
	4.36712e-16,
	-2.89657e-16,
	7.0863e-16,
	-2.92797e-16,
	1.36848e-16,
	-2.50539e-16,
	7.77285e-16,
	-1.69189e-16,
	5.29006e-16,
	-7.29126e-17,
	7.11863e-16,
	-2.14769e-16,
	5.60413e-16,
	1.82282e-17,
	7.83283e-17,
	-1.22015e-16,
	-5.70218e-17,
	2.26928e-16,
	4.88411e-16,
	-5.67106e-16,
	9.2577e-16 };

// LayerNorm
std::vector<double> cheb_coeff_inv_layernorm = { 0.746083,
	-0.491487,
	0.406744,
	-0.356017,
	0.319894,
	-0.2919,
	0.26909,
	-0.249875,
	0.233303,
	-0.218755,
	0.205809,
	-0.19416,
	0.183585,
	-0.173914,
	0.165011,
	-0.156773,
	0.149113,
	-0.141961,
	0.135258,
	-0.128956,
	0.123013,
	-0.117393,
	0.112066,
	-0.107004,
	0.102184,
	-0.0975847,
	0.0931888,
	-0.0889794,
	0.0849418,
	-0.0810629,
	0.0773307,
	-0.0737343,
	0.0702639,
	-0.0669104,
	0.0636656,
	-0.0605218,
	0.057472,
	-0.0545095,
	0.0516284,
	-0.0488229,
	0.0460877,
	-0.0434178,
	0.0408085,
	-0.0382555,
	0.0357544,
	-0.0333013,
	0.0308925,
	-0.0285242,
	0.0261931,
	-0.0238957,
	0.021629,
	-0.0193897,
	0.0171749,
	-0.0149817,
	0.0128071,
	-0.0106485,
	0.0085031,
	-0.00636818,
	0.00424111,
	-0.00211925 };

// tanh: [-20, 20]
std::vector<double> cheb_coeff_tanh = { 1.77636e-16,
	1.27193,
	1.4803e-17,
	-0.420508,
	2.40548e-16,
	0.248214,
	-7.40149e-18,
	-0.173038,
	2.62753e-16,
	0.130341,
	3.21965e-16,
	-0.102516,
	1.70234e-16,
	0.0827991,
	4.07082e-16,
	-0.0680397,
	1.14723e-16,
	0.0565678,
	-4.07082e-17,
	-0.0474136,
	-5.44009e-16,
	0.0399709,
	6.62433e-16,
	-0.0338387,
	1.70234e-16,
	0.0287378,
	3.84877e-16,
	-0.0244661,
	3.14563e-16,
	0.0208721,
	8.69675e-16,
	-0.0178388,
	2.70154e-16,
	0.015274,
	-9.62193e-16,
	-0.0131038,
	4.84797e-16,
	0.0112679,
	3.62673e-16,
	-0.00971691,
	-2.22045e-16,
	0.00840999,
	-2.60902e-16,
	-0.00731326,
	4.81097e-17,
	0.00639867,
	2.59052e-17,
	-0.00564301,
	1.33227e-16,
	0.00502725,
	5.18104e-16,
	-0.00453594,
	-7.03141e-16,
	0.00415682,
	3.79326e-17,
	-0.00388044,
	-5.13941e-16,
	0.00369991,
	9.77921e-16,
	-0.00361075 };

// gelu: [-20, 20]
std::vector<double> cheb_coeff_gelu = { 1,
	0.635822,
	5.56603e-17,
	-0.209827,
	-1.12097e-17,
	0.123397,
	-4.48661e-17,
	-0.0855298,
	2.50288e-17,
	0.063909,
	1.88287e-18,
	-0.0497338,
	2.41997e-17,
	0.0396273,
	4.63997e-17,
	-0.032018,
	-2.72255e-17,
	0.0260762,
	-4.21304e-17,
	-0.021321,
	7.41397e-17,
	0.0174533,
	1.50102e-16,
	-0.014275,
	4.45131e-17,
	0.0116478,
	8.66756e-17,
	-0.00947079,
	-3.74048e-17,
	0.00766665,
	3.93415e-16,
	-0.0061744,
	2.49301e-16,
	0.00494432,
	1.5742e-16,
	-0.00393504,
	4.98387e-17,
	0.00311153,
	-7.53384e-17,
	-0.00244393,
	-7.15577e-16,
	0.00190663,
	-9.94345e-17,
	-0.0014777,
	4.47672e-17,
	0.00113843,
	5.02118e-18,
	-0.000872989,
	1.96112e-17,
	0.000668154,
	1.28816e-16,
	-0.00051305,
	-1.7461e-17,
	0.000398947,
	8.11961e-17,
	-0.000319084,
	-8.42152e-17,
	0.000268526,
	1.72664e-16,
	-0.000244044 };

namespace FIDESlib::CKKS {

void evalFunction(FIDESlib::CKKS::Ciphertext& ctxt, std::vector<double> cheb_coeff, int numSlots, double lower_bound = -1, double upper_bound = 1, bool bts = false) {
	// affine transformation to scale
	if (!(lower_bound == -1.0 && upper_bound == 1.0)) {
		double scale = 2.0 / (upper_bound - lower_bound);
		double shift = -(upper_bound + lower_bound) / (upper_bound - lower_bound);

		if (scale != 1) {
			ctxt.multScalar(scale * (bts && (ctxt.getLevel() + 1 - ctxt.NoiseLevel == 1) ? 1.0 : GetPreScaleFactor(ctxt.cc_, numSlots)));
		}
		ctxt.addScalar(shift);
		lower_bound = -1;
		upper_bound = 1;
	}
	if (bts == true) {
		Bootstrap(ctxt, numSlots, (ctxt.getLevel() + 1 - ctxt.NoiseLevel == 0));
	}
	evalChebyshevSeries(ctxt, cheb_coeff, -1.0, 1.0);
}

void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, int numSlots, double lower_bound, double upper_bound, bool bts) {

	evalFunction(ctxt, cheb_coeff_tanh, numSlots, -20, 20, bts);
}

void EvalTanh_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, int numSlots, double lower_bound, double upper_bound, bool bts) {

	for (size_t i = 0; i < ctxt.size(); i++) {
		for (size_t j = 0; j < ctxt[0].size(); j++) {
			evalTanh(ctxt[i][j], numSlots, lower_bound, upper_bound, bts);
		}
	}
}

void evalGelu(FIDESlib::CKKS::Ciphertext& ctxt, int numSlots) {

	//bool print		= false; // for debugging
	Context& cc_	= ctxt.cc_;
	// ContextData& cc = ctxt.cc;

	Ciphertext tmp(ctxt.cc_);
	tmp.copy(ctxt);
	tmp.multScalar(GetPreScaleFactor(cc_, numSlots));
	evalFunction(ctxt, cheb_coeff_gelu, numSlots, -1, 1); // Approximation to GeLU / x, prescaled
	ctxt.mult(tmp);
	Bootstrap(ctxt, numSlots, true);
	// ctxt.mult(ctxt, tmp); // GeLU
}

void EvalGelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, int numSlots, int test_case) {

	for (size_t i = 0; i < ctxt.size(); i++) {
		for (size_t j = 0; j < ctxt[0].size(); j++) {
			evalGelu(ctxt[i][j], numSlots); // GeLU
		}
	}
}

void EvalSoftmax(FIDESlib::CKKS::Ciphertext& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  FIDESlib::CKKS::Plaintext& mask_token,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_broadcast,
  FIDESlib::CKKS::Plaintext& mask_mean,
  FIDESlib::CKKS::Plaintext& mask_max,
  int numSlots,
  int blockSize,
  int bStepAcc,
  int token_length,
  bool bts,
  int test_case,
  int layerNo,
  int long_input,
  double num_sigma) {
	//bool print = false; // for debugging

	//Context& cc_	= ctxt.cc_;
	// ContextData& cc = ctxt.cc;

	auto context = ctxt_cpu->GetCryptoContext(); // for debugging

	int accum_size = 1 << static_cast<int>(std::ceil(std::log2(token_length)));

	// Exponential
	evalFunction(ctxt, cheb_coeff_exp_softmax_1, numSlots, -1, 1); // prescaled

	ctxt.mult(ctxt, ctxt); // x^2
	if (bts) {
		Bootstrap(ctxt, numSlots);
	}

	for (int i = 0; i < 3; i++) {
		ctxt.mult(ctxt, ctxt); // x^16
	}
	ctxt.multPt(mask_token);

	Ciphertext scores_sum(ctxt.cc_);
	scores_sum.copy(ctxt);
	FIDESlib::CKKS::Accumulate(scores_sum, bStepAcc, 1, accum_size);
	scores_sum.multPt(mask_broadcast[2]);
	Broadcast(scores_sum, bStepAcc, 1, accum_size);

	if (bts) {
		Bootstrap(scores_sum, numSlots);
	}

	Ciphertext scores_sum_x(scores_sum.cc_);
	scores_sum_x.copy(scores_sum);

	scores_sum.addScalar(-(1e4 + 1) / (1e4 - 1));

	// // 1/x step 1
	evalFunction(scores_sum, cheb_coeff_inv_softmax1_59, numSlots, -1, 1); // prescaled
	if (bts) {
		Bootstrap(scores_sum, numSlots);
	}

	if (layerNo == 0) {
		scores_sum.addScalar(0.5); // TODO: remove
	}
	//  // 1/x step 2
	evalFunction(scores_sum, cheb_coeff_inv_softmax2_59, numSlots, 1, 3, bts);
	if (bts) {
		Bootstrap(scores_sum, numSlots);
	}

	double scale = 2.0 / (1e4 - 1);
	scores_sum_x.multScalar(1 / scale);

	// NewtonRaphson
	NewtonRaphsonInv(scores_sum_x, scores_sum, 3, ctxt, ctxt_cpu, privateKey);

	{
		// ctxt.copy(scores_sum_x);
		if (bts) {
			Bootstrap(ctxt, numSlots);
		}
		return;
	}
	if (bts) {
		Bootstrap(ctxt, numSlots);
	}

	// ctxt.mult(ctxt, scores_sum); // embedded in NewtonRaphsonInv
}

void EvalSoftmax_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  FIDESlib::CKKS::Plaintext& mask_token,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_broadcast,
  FIDESlib::CKKS::Plaintext& mask_mean,
  FIDESlib::CKKS::Plaintext& mask_max,
  int numSlots,
  int blockSize,
  int bStepAcc,
  int token_length,
  bool bts,
  int test_case,
  int layerNo,
  int long_input,
  double num_sigma) {

	for (size_t i = 0; i < ctxt.size(); i++) {
		for (size_t j = 0; j < ctxt[0].size(); j++) {
			EvalSoftmax(
			  ctxt[i][j], ctxt_cpu, privateKey, mask_token, mask_broadcast, mask_mean, mask_max, numSlots, blockSize, bStepAcc, token_length, bts, test_case, layerNo, long_input, num_sigma);
		}
	}
}

void EvalLayerNorm(FIDESlib::CKKS::Ciphertext& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_ln,
  FIDESlib::CKKS::Plaintext& mask_row,
  int numSlots,
  int blockSize,
  FIDESlib::CKKS::Plaintext& weight,
  FIDESlib::CKKS::Plaintext& bias,
  bool bts,
  const int bStepAcc) {
	//bool print = false;

	Context& cc_	= ctxt.cc_;
	// ContextData& cc = ctxt.cc;

	auto context = ctxt_cpu->GetCryptoContext();

	Ciphertext sum(ctxt.cc_);
	sum.copy(ctxt);

	sum.dropToLevel(mask_ln[0].c0.getLevel() + mask_ln[1].NoiseLevel - 1);
	Accumulate(sum, bStepAcc, 1, blockSize);

	sum.multPt(mask_ln[0]);

	Broadcast(sum, bStepAcc, 1, blockSize);

	FIDESlib::CKKS::Ciphertext var(cc_);

	var.copy(ctxt);
	var.sub(sum); // ctxt - mean = var
	var.mult(var, var);

	Ciphertext sum_var(ctxt.cc_);
	sum_var.copy(var);
	Accumulate(sum_var, bStepAcc, 1, blockSize);

	sum_var.multPt(mask_ln[1]);
	Broadcast(sum_var, bStepAcc, 1, blockSize);
	if (bts)
		Bootstrap(sum_var, numSlots);

	Ciphertext sum_var_x(cc_);
	sum_var_x.copy(sum_var);

	double upper_bound = 100;
	double lower_bound = 0.01;

	sum_var.addScalar(-(upper_bound + lower_bound) / (upper_bound - lower_bound));

	evalFunction(sum_var, cheb_coeff_inv_layernorm, numSlots, -1, 1, bts);
	if (bts)
		Bootstrap(sum_var, numSlots);

	double scale = 2.0 / (upper_bound - lower_bound);
	sum_var_x.multScalar(0.5 / scale);
	NewtonRaphsonInvSqrt(sum_var_x, sum_var, 2);

	ctxt.sub(sum); // ctxt - mean

	Plaintext weight_masked(cc_);
	weight_masked.multPt(weight, mask_row, true);

	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();
	ctxt.dropToLevel(weight_masked.c0.getLevel());
	ctxt.multPt(weight_masked);
	ctxt.mult(ctxt, sum_var);

	Plaintext bias_masked(cc_);
	bias_masked.multPt(bias, mask_row, true);
	ctxt.addPt(bias_masked);
}

void EvalLayerNorm_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_ln,
  FIDESlib::CKKS::Plaintext& mask_row,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
  int numSlots,
  int blockSize,
  const int bStepAcc,
  bool bts) {

	for (size_t i = 0; i < ctxt.size(); i++) {
		for (size_t j = 0; j < ctxt[0].size(); j++) {
			EvalLayerNorm(ctxt[i][j], ctxt_cpu, privateKey, mask_ln, mask_row, numSlots, blockSize, weight[i][j], bias[i][j], bts, bStepAcc);
		}
	}
}

void NewtonRaphsonInv(FIDESlib::CKKS::Ciphertext& ctxt,
  FIDESlib::CKKS::Ciphertext& initial,
  int num_iter,
  FIDESlib::CKKS::Ciphertext& final,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey) {

	//bool print = false;

	FIDESlib::CKKS::Context& cc = ctxt.cc_;
	auto context				= ctxt_cpu->GetCryptoContext();

	FIDESlib::CKKS::Ciphertext ctxt_tmp(cc), ctxt_y(cc), ctxt_z(cc);

	ctxt_y.copy(initial); // y
	ctxt_z.copy(ctxt);	  // x
	ctxt_y.mult(initial); // y^2
	ctxt_z.mult(ctxt_y);  // xy^2

	ctxt_tmp.copy(initial); // y
	ctxt_tmp.add(ctxt_tmp); // 2y

	ctxt_tmp.dropToLevel(ctxt_z.getLevel());

	ctxt_tmp.sub(ctxt_z); // y' = 2y - xy^2

	for (int iter = 1; iter < num_iter - 1; iter++) {
		ctxt_z.copy(ctxt);				 // x
		ctxt_y.mult(ctxt_tmp, ctxt_tmp); // y^2
		ctxt_z.mult(ctxt_z, ctxt_y);	 // xy^2

		ctxt_tmp.add(ctxt_tmp); // 2y
		ctxt_tmp.dropToLevel(ctxt_z.getLevel());

		ctxt_tmp.sub(ctxt_tmp, ctxt_z); // 2y - xy^2
	}

	// last iteration: embeds the multiplication by final, to save 1 level in Softmax
	ctxt_z.copy(ctxt);	// x
	ctxt_z.mult(final); // x * final

	ctxt_y.mult(ctxt_tmp, ctxt_tmp); // y^2
	ctxt_z.mult(ctxt_z, ctxt_y);	 // xy^2 * final

	ctxt_tmp.add(ctxt_tmp); // 2y
	ctxt_tmp.mult(final);	// 2y * final
	ctxt_tmp.dropToLevel(ctxt_z.getLevel());

	ctxt_tmp.sub(ctxt_tmp, ctxt_z); // (2y - xy^2) * final

	final.copy(ctxt_tmp);
}

void NewtonRaphsonInvSqrt(FIDESlib::CKKS::Ciphertext& ctxt, FIDESlib::CKKS::Ciphertext& initial, int num_iter) {

	FIDESlib::CKKS::Context& cc = ctxt.cc_;

	FIDESlib::CKKS::Ciphertext ctxt_x(cc), ctxt_y(cc);
	FIDESlib::CKKS::Ciphertext ctxt_y_sq(cc), ctxt_y_cu(cc), ctxt_xy_cu(cc), ctxt_tmp1(cc), ctxt_tmp2(cc);

	ctxt_x.copy(ctxt);	  // x
	ctxt_y.copy(initial); // y0

	for (int iter = 0; iter < num_iter; iter++) {
		ctxt_y_sq.copy(ctxt_y);
		ctxt_y_sq.mult(ctxt_y_sq, ctxt_y); // y^2

		ctxt_xy_cu.copy(ctxt_x);
		ctxt_xy_cu.mult(ctxt_y); // xy
		ctxt_xy_cu.dropToLevel(ctxt_y_sq.getLevel());

		ctxt_xy_cu.mult(ctxt_xy_cu, ctxt_y_sq); // x * y^3

		ctxt_tmp1.copy(ctxt_y);
		ctxt_tmp1.multScalar(1.5); // 1.5 * y

		// ctxt_tmp2.multScalar(0.5);                                       // 0.5 * x * y^3
		ctxt_tmp1.dropToLevel(ctxt_xy_cu.getLevel());

		ctxt_y.sub(ctxt_tmp1, ctxt_xy_cu); // y = 1.5y - 0.5xy^3
	}

	initial.copy(ctxt_y); // return y
}
} // namespace FIDESlib::CKKS
