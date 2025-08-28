#include "PolyApprox.cuh"

// Softmax
// exp: [-2, 2]
std::vector<double> cheb_coeff_exp_softmax = {22.6038, 19.5189, 12.8444, 6.67455, 2.83255, 1.00945, 0.30893, 0.0826599, 0.0196199, 0.00418051, 0.000807578, 0.000142616, 2.31874e-05, 3.49196e-06, 4.89675e-07, 6.42347e-08, 7.91444e-09, 9.19231e-10, 1.0097e-10, 1.05187e-11, 1.04334e-12, 9.62492e-14, 1.38012e-14, 3.17005e-15, 2.97591e-16, 4.20728e-15, 3.33981e-15, -1.15174e-15};
// 1/x step 1: [1, 1e4]
std::vector<double> cheb_coeff_inv_softmax1 = {15.2662, 3.04038, -1.00686, 0.537829, -0.344851, 0.24387, -0.183359, 0.143772, -0.116229, 0.0961676, -0.0810316, 0.0692861, -0.0599607, 0.0524146, -0.0462096, 0.0410368, -0.0366729, 0.0329531, -0.029753, 0.0269775, -0.0245527, 0.0224202, -0.0205335, 0.0188551, -0.0173546, 0.0160069, -0.0147912, 0.0136901, -0.0126892, 0.011776, -0.0109402, 0.0101726, -0.00946568, 0.00881261, -0.00820761, 0.0076456, -0.00712214, 0.00663329, -0.00617558, 0.00574594, -0.00534162, 0.00496015, -0.00459934, 0.00425718, -0.00393188, 0.0036218, -0.00332545, 0.00304144, -0.00276854, 0.00250556, -0.00225144, 0.00200515, -0.00176577, 0.00153238, -0.00130416, 0.00108029, -0.000860006, 0.000642551, -0.000427205, 0.000213254};
// 1/x step 2: [1, 10]
std::vector<double> cheb_coeff_inv_softmax2 = {0.210884, -0.201002, 0.176099, -0.144081, 0.111611, -0.0827377, 0.0591755, -0.0410896, 0.0278333, -0.0184621, 0.0120277, -0.00771463, 0.00488133, -0.00305179, 0.00188779, -0.00115673, 0.00070276, -0.000423683, 0.000253654, -0.000150898, 8.92474e-05, -5.25035e-05, 3.07357e-05, -1.79111e-05, 1.03937e-05, -6.0078e-06, 3.45998e-06, -1.98587e-06, 1.13616e-06, -6.48079e-07, 3.68631e-07, -2.09124e-07, 1.18339e-07, -6.68079e-08, 3.76318e-08, -2.11524e-08, 1.18657e-08, -6.64345e-09, 3.71282e-09, -2.07139e-09, 1.15372e-09, -6.41587e-10, 3.5625e-10, -1.97527e-10, 1.0937e-10, -6.0477e-11, 3.33993e-11, -1.84223e-11, 1.01497e-11, -5.58507e-12, 3.07035e-12, -1.6857e-12, 9.24767e-13, -5.06924e-13, 2.77302e-13, -1.51643e-13, 8.26928e-14, -4.3749e-14, 2.21008e-14, -8.72618e-15};
// sqrt(x)
// std::vector<double> cheb_coeff_squareroot = {4.03512, 1.33466, -0.262842, 0.110326, -0.0597802, 0.0369779, -0.0248124, 0.0175913, -0.0129753, 0.00985927, -0.00766574, 0.00606939, -0.0048757, 0.00396263, -0.00325056, 0.00268574, -0.00223088, 0.00185942, -0.00155204, 0.00129439, -0.00107563, 0.000887367, -0.000723054, 0.000577433, -0.000446209, 0.000325788, -0.000213075, 0.000105324};
std::vector<double> cheb_coeff_squareroot = {4.02777, 1.34082, -0.267314, 0.114015, -0.0629426, 0.0397433, -0.0272604, 0.0197758, -0.0149357, 0.0116253, -0.00926059, 0.00751176, -0.00618084, 0.00514318, -0.00431709, 0.00364721, -0.00309488, 0.00263245, -0.00223965, 0.00190132, -0.00160591, 0.00134445, -0.00110982, 0.000896257, -0.000699028, 0.000514111, -0.000338015, 0.000167612};
std::vector<double> cheb_coeff_squareroot_13 = {1.27626, 0.421811, -0.082865, 0.0346236, -0.0186233, 0.0113904, -0.00751481, 0.00519526, -0.00369067, 0.00264969, -0.00188728, 0.00129734, -0.000813979, 0.000392673};

// LayerNorm
std::vector<double> cheb_coeff_squareroot_ln = {5.72578, 1.87097, -0.358808, 0.144356, -0.0731997, 0.0405698, -0.0222362, 0.00995092};

// sqrt(x)
std::vector<double> cheb_coeff_inv_layernorm_1_1 = {12.736, 4.24097, -0.846262, 0.361496, -0.200004, 0.126656, -0.0871998, 0.0635537, -0.0482738, 0.0378351, -0.0303914, 0.0248999, -0.0207346, 0.017502, -0.0149441, 0.0128863, -0.0112069, 0.00981909, -0.00865952, 0.00768112, -0.00684831, 0.00613381, -0.00551641, 0.00497942, -0.00450957, 0.00409621, -0.00373069, 0.00340593, -0.00311612, 0.00285641, -0.00262278, 0.00241181, -0.00222064, 0.00204682, -0.00188827, 0.00174318, -0.00161, 0.00148739, -0.00137418, 0.00126934, -0.00117197, 0.00108128, -0.000996555, 0.000917179, -0.000842591, 0.000772289, -0.000705823, 0.000642784, -0.000582804, 0.000525542, -0.00047069, 0.000417962, -0.000367092, 0.000317834, -0.000269955, 0.000223237, -0.000177469, 0.000132453, -8.79949e-05, 4.39056e-05};
// 1/x
std::vector<double> cheb_coeff_inv_layernorm_1_2 = {2, -1.63636, 1.33884, -1.09542, 0.89625, -0.733296, 0.599969, -0.490884, 0.401632, -0.328608, 0.268861, -0.219977, 0.179982, -0.147258, 0.120483, -0.0985774, 0.0806542, -0.0659898, 0.0539917, -0.044175, 0.0361432, -0.0295717, 0.024195, -0.0197959, 0.0161967, -0.0132518, 0.0108424, -0.00887104, 0.00725812, -0.00593845, 0.00485872, -0.00397531, 0.00325251, -0.00266113, 0.00217726, -0.00178137, 0.00145746, -0.00119242, 0.000975573, -0.000798138, 0.000652952, -0.000534147, 0.000436924, -0.000357355, 0.000292224, -0.0002389, 0.000195228, -0.000159445, 0.000130104, -0.000106019, 8.62186e-05, -6.99014e-05, 5.64084e-05, -4.51946e-05, 3.58069e-05, -2.78658e-05, 2.10507e-05, -1.50861e-05, 9.73107e-06, -4.76919e-06};
// // 1 / sqrt(x)
std::vector<double> cheb_coeff_inv_layernorm = {0.746083, -0.491487, 0.406744, -0.356017, 0.319894, -0.2919, 0.26909, -0.249875, 0.233303, -0.218755, 0.205809, -0.19416, 0.183585, -0.173914, 0.165011, -0.156773, 0.149113, -0.141961, 0.135258, -0.128956, 0.123013, -0.117393, 0.112066, -0.107004, 0.102184, -0.0975847, 0.0931888, -0.0889794, 0.0849418, -0.0810629, 0.0773307, -0.0737343, 0.0702639, -0.0669104, 0.0636656, -0.0605218, 0.057472, -0.0545095, 0.0516284, -0.0488229, 0.0460877, -0.0434178, 0.0408085, -0.0382555, 0.0357544, -0.0333013, 0.0308925, -0.0285242, 0.0261931, -0.0238957, 0.021629, -0.0193897, 0.0171749, -0.0149817, 0.0128071, -0.0106485, 0.0085031, -0.00636818, 0.00424111, -0.00211925};

// gelu: [-20, 20] 
std::vector<double> cheb_coeff_relu = {12.7164, 10, 4.25996, -1.12549e-15, -0.864296, -4.29301e-16, 0.378673, 5.11298e-16, -0.216208, -2.86107e-17, 0.141752, -2.53104e-16, -0.101066, -1.36003e-16, 0.0760924, -1.87066e-16, -0.0594185, -7.45384e-16, 0.0475514, 4.08234e-16, -0.0386772, -7.37563e-16, 0.0317833, 6.37343e-16, -0.0262714, 1.22969e-15, 0.0217706, 1.96066e-15, -0.0180414, 5.59383e-16, 0.0149225, -1.00925e-15, -0.0123007, 7.9624e-17, 0.0100929, 6.5449e-16, -0.00823504, 2.93692e-15, 0.00667598, -1.43119e-15, -0.00537299, 2.2065e-15, 0.00428933, 2.10318e-15, -0.00339274, -2.12901e-15, 0.0026544, 1.60021e-15, -0.00204834, -3.0896e-15, 0.00155104, 1.57273e-15, -0.00114103, 3.13517e-15, 0.000798621, 2.02804e-15, -0.000505584, 2.57601e-15, 0.000244821, 1.13086e-14};

// tanh: [-20, 20]
std::vector<double> cheb_coeff_tanh = {1.77636e-16, 1.27193, 1.4803e-17, -0.420508, 2.40548e-16, 0.248214, -7.40149e-18, -0.173038, 2.62753e-16, 0.130341, 3.21965e-16, -0.102516, 1.70234e-16, 0.0827991, 4.07082e-16, -0.0680397, 1.14723e-16, 0.0565678, -4.07082e-17, -0.0474136, -5.44009e-16, 0.0399709, 6.62433e-16, -0.0338387, 1.70234e-16, 0.0287378, 3.84877e-16, -0.0244661, 3.14563e-16, 0.0208721, 8.69675e-16, -0.0178388, 2.70154e-16, 0.015274, -9.62193e-16, -0.0131038, 4.84797e-16, 0.0112679, 3.62673e-16, -0.00971691, -2.22045e-16, 0.00840999, -2.60902e-16, -0.00731326, 4.81097e-17, 0.00639867, 2.59052e-17, -0.00564301, 1.33227e-16, 0.00502725, 5.18104e-16, -0.00453594, -7.03141e-16, 0.00415682, 3.79326e-17, -0.00388044, -5.13941e-16, 0.00369991, 9.77921e-16, -0.00361075};

// Degree 27 coefficients
std::vector<double> cheb_coeff_exp_softmax_4 = {22.6038, 19.5189, 12.8444, 6.67455, 2.83255, 1.00945, 0.30893, 0.0826599, 0.0196199, 0.00418051, 0.000807578, 0.000142616, 2.31874e-05, 3.49196e-06, 4.89675e-07, 6.42347e-08, 7.91444e-09, 9.19231e-10, 1.0097e-10, 1.05187e-11, 1.04334e-12, 9.62492e-14, 1.38012e-14, 3.17005e-15, 2.97591e-16, 4.20728e-15, 3.33981e-15, -1.15174e-15};
std::vector<double> cheb_coeff_exp_softmax_2 = {4.55917, 3.18127, 1.3779, 0.42548, 0.101457, 0.0196514, 0.00320035, 0.000449278, 5.53987e-05, 6.08837e-06, 6.03393e-07, 5.4444e-08, 4.50826e-09, 3.44903e-10, 2.45199e-11, 1.62784e-12, 1.01355e-13, 5.99781e-15, 5.75283e-16, -1.38733e-16, 2.36558e-16, -4.44402e-16, 3.6722e-16, -2.6676e-18, -7.10034e-16, 3.22581e-16, 4.16053e-16, 2.70171e-16}; 
std::vector<double> cheb_coeff_exp_softmax_1 = {2.53213, 1.13032, 0.271495, 0.0443368, 0.00547424, 0.000542926, 4.49773e-05, 3.19844e-06, 1.99212e-07, 1.10368e-08, 5.5059e-10, 2.49795e-11, 1.03914e-12, 3.99088e-14, 1.76514e-15, 1.96862e-16, -1.32079e-17, 1.40569e-16, 1.23913e-16, -1.43947e-16, 3.08643e-17, -2.166e-16, -3.91206e-17, -3.18494e-16, -6.44747e-16, -8.80072e-17, 1.77988e-16, 7.41657e-16};
std::vector<double> cheb_coeff_exp_softmax_05 = {2.12697, 0.515789, 0.0638123, 0.00529022, 0.000329611, 1.64463e-05, 6.84247e-07, 2.44102e-08, 7.62157e-10, 2.11564e-11, 5.28556e-13, 1.19717e-14, 2.53367e-16, 7.55152e-17, 4.66275e-16, 2.90244e-16, -2.72772e-17, 9.7106e-17, 3.80061e-17, -1.27505e-16, -3.30434e-17, -1.65405e-16, -1.86796e-16, -5.13417e-16, -5.78011e-16, -1.71782e-16, 7.43779e-17, 1.08655e-15};

std::vector<double> cheb_coeff_inv_softmax1_27 = {5.44283, 0.596525, -0.241373, 0.141981, -0.0968308, 0.0715189, -0.0555137, 0.0445659, -0.0366489, 0.03068, -0.0260309, 0.0223132, -0.0192746, 0.0167442, -0.0146025, 0.0127633, -0.011163, 0.00975347, -0.0084977, 0.00736661, -0.00633693, 0.00538976, -0.00450947, 0.00368287, -0.00289863, 0.00214682, -0.0014185, 0.000705484};
std::vector<double> cheb_coeff_inv_softmax2_27 = {0.302733, -0.277884, 0.218387, -0.150772, 0.0936511, -0.0533594, 0.0283131, -0.0141566, 0.00673172, -0.00306667, 0.00134625, -0.000572233, 0.000236433, -9.52668e-05, 3.75372e-05, -1.44967e-05, 5.49824e-06, -2.05145e-06, 7.54085e-07, -2.73437e-07, 9.79177e-08, -3.46627e-08, 1.21406e-08, -4.21048e-09, 1.44675e-09, -4.92363e-10, 1.64628e-10, -4.97485e-11};
std::vector<double> cheb_coeff_inv_softmax_27 = {0.0101598, -0.00976182, 0.00936772, -0.00897736, 0.0085906, -0.00820727, 0.00782722, -0.00745031, 0.00707638, -0.00670528, 0.00633686, -0.00597097, 0.00560748, -0.00524623, 0.00488707, -0.00452987, 0.00417449, -0.00382077, 0.00346858, -0.00311778, 0.00276823, -0.00241978, 0.0020723, -0.00172565, 0.0013797, -0.00103429, 0.000689297, -0.000344579};

std::vector<double> cheb_coeff_inv_layernorm_27 = {0.68943, -0.434804, 0.34997, -0.299093, 0.262758, -0.234492, 0.211346, -0.191734, 0.1747, -0.159626, 0.146086, -0.133776, 0.122469, -0.111994, 0.102214, -0.0930209, 0.0843266, -0.0760575, 0.0681518, -0.0605567, 0.0532264, -0.0461205, 0.0392031, -0.0324418, 0.0258069, -0.0192707, 0.0128073, -0.00639178};
std::vector<double> cheb_coeff_relu_27 = {1, 0.635894, 6.10865e-17, -0.210048, -2.29257e-17, 0.123783, -4.54656e-17, -0.0861079, 1.32285e-17, 0.0647187, -4.28239e-17, -0.0508282, 3.02237e-17, 0.0410745, 3.64694e-17, -0.0339037, 8.43072e-18, 0.0285058, 5.4813e-17, -0.0244228, 4.75268e-19, 0.0213817, -1.36629e-16, -0.0192146, -4.53836e-16, 0.0178187, 8.04664e-17, -0.0171345};
std::vector<double> cheb_coeff_tanh_27 = {1.90324e-16, 1.27231, 3.17207e-17, -0.421657, 3.17207e-17, 0.250161, 7.93016e-17, -0.17583, 8.72318e-17, 0.134049, 5.55112e-17, -0.107231, 8.72318e-17, 0.0886388, -5.63042e-16, -0.0751497, 6.34413e-17, 0.0651265, 1.74464e-16, -0.0576362, 6.74064e-17, 0.0521168, 0, -0.0482184, -3.80648e-16, 0.0457239, 2.5773e-16, -0.044506};
std::vector<double> cheb_coeff_relu_13 = {1, 0.633708, -2.88693e-17, -0.203749, 6.61587e-19, 0.114141, -6.73466e-18, -0.0743652, 1.8037e-17, 0.0525471, 5.47986e-17, -0.0403031, -5.0067e-18, 0.0346936};

std::vector<double> cheb_coeff_inv_layernorm1_27 = {3.06073, 0.335451, -0.135734, 0.0798416, -0.0544519, 0.040218, -0.0312177, 0.0250613, -0.0206092, 0.0172527, -0.0146383, 0.0125477, -0.0108389, 0.00941596, -0.00821158, 0.00717733, -0.0062774, 0.00548478, -0.00477861, 0.00414255, -0.00356352, 0.00303089, -0.00253586, 0.00207103, -0.00163002, 0.00120724, -0.000797683, 0.000396723};
std::vector<double> cheb_coeff_inv_layernorm2_27 = {1404.83, -1376.09, 1295.97, -1178.16, 1038.42, -891.04, 747.025, -613.793, 495.539, -393.953, 308.968, -239.415, 183.538, -139.353, 104.891, -78.3319, 58.0792, -42.7789, 31.3152, -22.788, 16.4843, -11.8469, 8.44538, -5.94998, 4.1088, -2.72895, 1.66075, -0.784642};

namespace FIDESlib::CKKS { 

void evalFunction(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey,
                  std::vector<double> cheb_coeff, int numSlots, double lower_bound = -1, double upper_bound = 1, bool bts = false) {
    // affine transformation to scale
    if (!(lower_bound == -1.0 && upper_bound == 1.0)) {
        double scale = 2.0 / (upper_bound - lower_bound);
        double shift = -(upper_bound + lower_bound) / (upper_bound - lower_bound);

            if (scale != 1){
                ctxt.multScalar(scale);
            }
            ctxt.addScalar(shift);
            lower_bound = -1;
            upper_bound = 1;
        }

        if (bts == true) {
            Bootstrap(ctxt, numSlots);
        }
        evalChebyshevSeries(ctxt, keySwitchingKey, cheb_coeff, lower_bound, upper_bound);
    }

    void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots, double lower_bound, double upper_bound, bool bts){
        
        // evalFunction(ctxt, keySwitchingKey, cheb_coeff_tanh, numSlots, lower_bound, upper_bound, bts);
        evalFunction(ctxt, keySwitchingKey, cheb_coeff_tanh_27, numSlots, lower_bound, upper_bound, bts);
        
    }

    void EvalTanh_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey, 
                int numSlots, double lower_bound, double upper_bound, bool bts){
        
        for (size_t i = 0; i < ctxt.size(); i++) {
            for (size_t j = 0; j < ctxt[0].size(); j++) {
                evalTanh(ctxt[i][j], keySwitchingKey, numSlots, lower_bound, upper_bound, bts);
            }
        }
    }

    void evalRelu(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots){

        // evalFunction(ctxt, keySwitchingKey, cheb_coeff_relu, numSlots, -20, 20);
        Ciphertext tmp(ctxt.cc);
        tmp.copy(ctxt);
        // prescaled
        evalFunction(ctxt, keySwitchingKey, cheb_coeff_relu_13, numSlots, -1, 1);   // Approximation to GeLU / x
        // evalFunction(ctxt, keySwitchingKey, cheb_coeff_relu_27, numSlots, -20, 20, true);   // Approximation to GeLU / x
        Bootstrap(ctxt, numSlots);
    
        ctxt.mult(ctxt, tmp, keySwitchingKey);     // GeLU

    }

    void EvalSoftmax(FIDESlib::CKKS::Ciphertext& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, 
                    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, const KeySwitchingKey& keySwitchingKey,
                    FIDESlib::CKKS::Plaintext& mask_token, FIDESlib::CKKS::Plaintext& mask_broadcast, FIDESlib::CKKS::Plaintext& mask_mean, FIDESlib::CKKS::Plaintext& mask_max,         
                    int numSlots, int blockSize, int bStepAcc, int token_length, bool bts, int test_case, int long_input){
        bool print = false;  // for debugging 

        if (print) std::cout << "SOFTMAX " << std::endl;      

        FIDESlib::CKKS::Context& cc = ctxt.cc;
        auto context = ctxt_cpu->GetCryptoContext();    // for debugging 

        int accum_size = 1 << static_cast<int>(std::ceil(std::log2(token_length)));
        // int accum_size = blockSize;

        if (test_case > 0) {      
            ctxt.multPt(mask_token);

            Ciphertext mean(ctxt.cc);
            mean.copy(ctxt);
            Accumulate(mean, bStepAcc, 1, accum_size);

            Plaintext mask_mean_(cc);
            mask_mean_.copy(mask_broadcast);
            mask_mean_.multScalar(1.0 / token_length, true);
            mean.multPt(mask_mean_);
            Broadcast(mean, bStepAcc, 1, accum_size);

            ////////////////////////////////////
            if (print) {
                std::cout << "mu " << std::endl;
                std::cout << "#: " << mean.getLevel() << " " << mean.NoiseLevel << std::endl;            
                FIDESlib::CKKS::RawCipherText raw_res;
                mean.store(cc, raw_res);
                auto result2(ctxt_cpu->Clone());
                GetOpenFHECipherText(result2, raw_res);

                lbcrypto::Plaintext result_pt;
                context->Decrypt(privateKey, result2, &result_pt);
                result_pt->SetLength(4);
                std::cout << result_pt << std::endl;
            }
            ////////////////////////////////////
            
            Ciphertext var(cc);
            var.copy(ctxt);
            var.sub(mean);  // ctxt - mean = var
            var.mult(var, var, keySwitchingKey);

            Ciphertext sum_var(ctxt.cc);
            sum_var.copy(var);
            Accumulate(sum_var, bStepAcc, 1, accum_size);
            sum_var.multPt(mask_mean_);
            Broadcast(sum_var, bStepAcc, 1, accum_size);

            // Square root
            // evalFunction(sum_var, keySwitchingKey, cheb_coeff_squareroot, numSlots, 0.1, 100, bts); // prescaled
            evalFunction(sum_var, keySwitchingKey, cheb_coeff_squareroot_13, numSlots, 0.001, 1, bts); // prescaled

            ////////////////////////////////////
            if (print) {
                std::cout << "sigma " << std::endl;
                std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
                FIDESlib::CKKS::RawCipherText raw_res;
                sum_var.store(cc, raw_res);
                auto result2(ctxt_cpu->Clone());
                GetOpenFHECipherText(result2, raw_res);

                lbcrypto::Plaintext result_pt;
                context->Decrypt(privateKey, result2, &result_pt);
                result_pt->SetLength(4);
                std::cout << result_pt << std::endl;
            }
            ////////////////////////////////////

            Ciphertext musigma(cc);
            musigma.copy(sum_var);
            musigma.sub(sum_var);      // 0


            musigma.add(sum_var);   // 1σ

            musigma.multPt(mask_max, true);
            
            musigma.add(sum_var);   // 1σ

            // if (test_case == 2) {
            //     musigma.multScalar(0.5, true);
            //     musigma.add(sum_var);   // 1σ
            // }

            musigma.add(mean);      // μ + 2σ

            ////////////////////////////////////
            if (print) {
                std::cout << "musigma " << std::endl;
                std::cout << "#: " << musigma.getLevel() << " " << musigma.NoiseLevel << std::endl;            
                FIDESlib::CKKS::RawCipherText raw_res;
                musigma.store(cc, raw_res);
                auto result2(ctxt_cpu->Clone());
                GetOpenFHECipherText(result2, raw_res);

                lbcrypto::Plaintext result_pt;
                context->Decrypt(privateKey, result2, &result_pt);
                result_pt->SetLength(4);
                std::cout << result_pt << std::endl;
            }
            ////////////////////////////////////

            ctxt.sub(musigma);
        }

        if (print) std::cout << "accum_size: " << accum_size << std::endl;      

        // Exponential
        evalFunction(ctxt, keySwitchingKey, cheb_coeff_exp_softmax_1, numSlots, -1, 1, bts); // prescaled

         ////////////////////////////////////
        if (print) {
            std::cout << "Step 1 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        if (bts){ Bootstrap(ctxt, numSlots); }
        for (int i=0; i < 4; i++){
            ctxt.mult(ctxt, ctxt, keySwitchingKey);    
        }
        if (bts){ Bootstrap(ctxt, numSlots); }  // ? necessary ?

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 2 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        ctxt.multPt(mask_token);
        Ciphertext scores_sum(ctxt.cc);
        scores_sum.copy(ctxt);
        FIDESlib::CKKS::Accumulate(scores_sum, bStepAcc, 1, accum_size);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 3-1 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        scores_sum.multPt(mask_broadcast); 

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 3-2 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Broadcast(scores_sum, bStepAcc, 1, accum_size);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 3-3 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext scores_sum_x(scores_sum.cc);
        scores_sum_x.copy(scores_sum);

        // 1/x step 1
        evalFunction(scores_sum, keySwitchingKey, cheb_coeff_inv_softmax1_27, numSlots, 1, 1e4, bts);
        if (bts){ Bootstrap(scores_sum, numSlots); }

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 4 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // // 1/x step 2
        evalFunction(scores_sum, keySwitchingKey, cheb_coeff_inv_softmax2_27, numSlots, 1, 3, bts);
        if (bts){ Bootstrap(scores_sum, numSlots); }

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 5 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // NewtonRaphson
        NewtonRaphsonInv(scores_sum_x, scores_sum, keySwitchingKey, 1, ctxt, ctxt_cpu, privateKey);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 6 " << std::endl;
            std::cout << "#: " << scores_sum.getLevel() << " " << scores_sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            scores_sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        ctxt.mult(ctxt, scores_sum, keySwitchingKey);
   
        ////////////////////////////////////
        if (print) {
            std::cout << "Step 7 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // ctxt.copy(scores_sum);
    }
    

    void EvalSoftmax_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, const KeySwitchingKey& keySwitchingKey, 
                FIDESlib::CKKS::Plaintext& mask_token, FIDESlib::CKKS::Plaintext& mask_broadcast, FIDESlib::CKKS::Plaintext& mask_mean, FIDESlib::CKKS::Plaintext& mask_max, int numSlots, int blockSize, int bStepAcc, int token_length, bool bts, int test_case, int long_input){
        
        for (size_t i = 0; i < ctxt.size(); i++) {
            for (size_t j = 0; j < ctxt[0].size(); j++) {
                EvalSoftmax(ctxt[i][j], ctxt_cpu, privateKey, keySwitchingKey, mask_token, mask_broadcast, mask_mean, mask_max, numSlots, blockSize, bStepAcc, token_length, bts, test_case, long_input);
            }
        }
    }

    void EvalGelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots){
        
        for (size_t i = 0; i < ctxt.size(); i++) {
            for (size_t j = 0; j < ctxt[0].size(); j++) {
                evalRelu(ctxt[i][j], keySwitchingKey, numSlots);    // GeLU
            }
        }
    }

    void EvalLayerNorm(FIDESlib::CKKS::Ciphertext& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, 
                    const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                    int numSlots, int blockSize, FIDESlib::CKKS::Plaintext& weight, FIDESlib::CKKS::Plaintext& bias, bool bts, const int bStepAcc) {
        bool print = false;

        FIDESlib::CKKS::Context& cc = ctxt.cc;
        auto context = ctxt_cpu->GetCryptoContext();

        if (print) std::cout << "LAYERNORM " << std::endl;      

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 0 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext sum(ctxt.cc);
        sum.copy(ctxt);

        Accumulate(sum, bStepAcc, 1, blockSize);


        ////////////////////////////////////
        if (print) {
            std::cout << "Step 1 " << std::endl;
            std::cout << "#: " << sum.getLevel() << " " << sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////
        
        sum.multPt(mask_ln[0]);

        Broadcast(sum, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 2 " << std::endl;
            std::cout << "#: " << sum.getLevel() << " " << sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        FIDESlib::CKKS::Ciphertext var(cc);

        var.copy(ctxt);
        var.sub(sum);  // ctxt - mean = var
        var.mult(var, var, keySwitchingKey);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 3 " << std::endl;
            std::cout << "#: " << var.getLevel() << " " << var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext sum_var(ctxt.cc);
        sum_var.copy(var);
        Accumulate(sum_var, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 4 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        sum_var.multPt(mask_ln[1]);
        Broadcast(sum_var, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 5 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////


        // std::cout << "weight: " << weight.c0.getLevel() << ", " << weight.NoiseLevel << std::endl;
        // std::cout << "bias: " << bias.c0.getLevel() << ", " << bias.NoiseLevel << std::endl;
        // std::cout << "mask_row: " << mask_row.c0.getLevel() << ", " << mask_row.NoiseLevel << std::endl;


        if (bts) Bootstrap(sum_var, numSlots);

        Ciphertext sum_var_x(cc);
        sum_var_x.copy(sum_var);
        sum_var.addScalar(-(100+0.01)/(100-0.01));

        // evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm1_27, numSlots, -1, 1, bts); 
        evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm_27, numSlots, -1, 1); 
        if (bts) Bootstrap(sum_var, numSlots);   

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 6 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm2_27, numSlots, 0.1, 2.1, bts); 
        // if (bts) Bootstrap(sum_var, numSlots);   

        double scale = 2.0 / (1e2 - 0.01);
        sum_var_x.multScalar(0.5 / scale);
        NewtonRaphsonInvSqrt(sum_var_x, sum_var, keySwitchingKey, 1);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 7 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////


        ctxt.sub(sum); // ctxt - mean

        // ctxt.mult(ctxt, sum_var, keySwitchingKey);

        Plaintext weight_masked(cc);
        weight_masked.copy(weight);
        weight_masked.multPt(weight_masked, mask_row, true);
        ctxt.multPt(weight_masked);

        ctxt.mult(ctxt, sum_var, keySwitchingKey);

        Plaintext bias_masked(cc);
        bias_masked.copy(bias);
        bias_masked.multPt(bias_masked, mask_row, true);
        ctxt.addPt(bias_masked);
    }

    void EvalLayerNorm_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
                            const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                            const int bStepAcc, bool bts) {

        for (size_t i = 0; i < ctxt.size(); i++) {
            for (size_t j = 0; j < ctxt[0].size(); j++) {
                EvalLayerNorm(ctxt[i][j], ctxt_cpu, privateKey, keySwitchingKey, mask_ln, mask_row, numSlots, blockSize, weight[i][j], bias[i][j], bts, bStepAcc);
            }
        }
    }



    void evalSqrtTaylor(FIDESlib::CKKS::Ciphertext& x, const KeySwitchingKey& ksk) {
        Context& cc = x.cc;

        // h = x - 1
        Ciphertext h(cc); h.copy(x);
        h.addScalar(-1.0);

        // h2 = h*h  (ct-ct mult #1)
        Ciphertext h2(cc);
        h2.mult(h, h, ksk);          

        // e1 = 1 + 0.5*h   
        Ciphertext e1(cc); e1.copy(h);
        e1.multScalar(0.5);
        e1.addScalar(1.0);

        // e2 = -1/8 + (1/16)*h  
        Ciphertext e2(cc); e2.copy(h);
        e2.multScalar(0.0625);        
        e2.addScalar(-0.125);     

        // prod = h2 * e2 
        Ciphertext prod(cc);
        e2.mult(h2, e2, ksk);
        e2.add(e1);
        x.copy(e2);
    }

    void NewtonRaphsonInv(FIDESlib::CKKS::Ciphertext& ctxt, FIDESlib::CKKS::Ciphertext& initial, const KeySwitchingKey& keySwitchingKey, int num_iter, 
                        FIDESlib::CKKS::Ciphertext& final, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey) {

        bool print = false;

        FIDESlib::CKKS::Context& cc = ctxt.cc;
        auto context = ctxt_cpu->GetCryptoContext();    

        FIDESlib::CKKS::Ciphertext ctxt_tmp(cc), ctxt_y(cc), ctxt_z(cc);

        ctxt_y.copy(initial);                 // y
        ctxt_z.copy(ctxt);                    // x
        ctxt_z.mult(initial, keySwitchingKey);           // x.y
        ctxt_z.mult(initial, keySwitchingKey);           // xy^2

        ctxt_tmp.copy(initial);  // y
        ctxt_tmp.add(ctxt_tmp); // 2y

        ctxt_tmp.dropToLevel(ctxt_z.getLevel());

        ctxt_tmp.sub(ctxt_z);  

        // for (int iter = 1; iter < num_iter; iter++) {
        //     ctxt_z.copy(ctxt_x);                           // x
        //     ctxt_y.mult(ctxt_y, ctxt_y, keySwitchingKey);  // y^2
        //     ctxt_z.mult(ctxt_z, ctxt_y, keySwitchingKey);  // xy^2

        //     ctxt_tmp.copy(ctxt_y);      // y
        //     ctxt_tmp.multScalar(2);     // 2y
        //     ctxt_tmp.dropToLevel(ctxt_z.getLevel());

        //     ctxt_y.sub(ctxt_tmp, ctxt_z);     // 2y - xy^2
        // }
        initial.copy(ctxt_tmp);
    }


    void NewtonRaphsonInvSqrt(FIDESlib::CKKS::Ciphertext& ctxt,
                FIDESlib::CKKS::Ciphertext& initial,
                const FIDESlib::CKKS::KeySwitchingKey& keySwitchingKey,
                int num_iter) {

        FIDESlib::CKKS::Context& cc = ctxt.cc;


        FIDESlib::CKKS::Ciphertext ctxt_x(cc), ctxt_y(cc);
        FIDESlib::CKKS::Ciphertext ctxt_y_sq(cc), ctxt_y_cu(cc), ctxt_xy_cu(cc), ctxt_tmp1(cc), ctxt_tmp2(cc);

        ctxt_x.copy(ctxt);      // x
        ctxt_y.copy(initial);   // y0

        for (int iter = 0; iter < num_iter; iter++) {
            ctxt_y_sq.copy(ctxt_y);
            ctxt_y_sq.mult(ctxt_y_sq, ctxt_y, keySwitchingKey);             // y^2


            ctxt_xy_cu.copy(ctxt_x);
            ctxt_xy_cu.mult(ctxt_y, keySwitchingKey);                      // xy
            ctxt_xy_cu.dropToLevel(ctxt_y_sq.getLevel());           

            ctxt_xy_cu.mult(ctxt_xy_cu, ctxt_y_sq, keySwitchingKey);        // x * y^3

            ctxt_tmp1.copy(ctxt_y);
            ctxt_tmp1.multScalar(1.5);                                       // 1.5 * y

            // ctxt_tmp2.multScalar(0.5);                                       // 0.5 * x * y^3
            ctxt_tmp1.dropToLevel(ctxt_xy_cu.getLevel());

            ctxt_y.sub(ctxt_tmp1, ctxt_xy_cu);                                // y = 1.5y - 0.5xy^3
        }

        initial.copy(ctxt_y); // return y
    }



    void EvalLayerNorm_DelayedInv(FIDESlib::CKKS::Ciphertext& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, 
                    const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                    int numSlots, int blockSize, FIDESlib::CKKS::Plaintext& weight, FIDESlib::CKKS::Plaintext& bias, bool bts, const int bStepAcc, FIDESlib::CKKS::Ciphertext& factor) {
        bool print = false;

        FIDESlib::CKKS::Context& cc = ctxt.cc;
        auto context = ctxt_cpu->GetCryptoContext();

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 0 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext sum(ctxt.cc);
        sum.copy(ctxt);

        Accumulate(sum, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 1 " << std::endl;
            std::cout << "#: " << sum.getLevel() << " " << sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // sum.multPt(mask_ln[0]);
        
        sum.multPt(mask_ln[0]);

        Broadcast(sum, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 2 " << std::endl;
            std::cout << "#: " << sum.getLevel() << " " << sum.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        FIDESlib::CKKS::Ciphertext var(cc);

        var.copy(ctxt);
        var.sub(sum);  // ctxt - mean = var
        var.mult(var, var, keySwitchingKey);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 3 " << std::endl;
            std::cout << "#: " << var.getLevel() << " " << var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext sum_var(ctxt.cc);
        sum_var.copy(var);
        Accumulate(sum_var, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 4 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        sum_var.multPt(mask_ln[0]);
        Broadcast(sum_var, bStepAcc, 1, blockSize);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 5 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        // sqrt(csum_var)
        // evalFunction(sum_var, keySwitchingKey, cheb_coeff_squareroot_ln, numSlots, 0.01, 100, bts);
        evalSqrtTaylor(sum_var, cc.GetEvalKey());
        if (bts) Bootstrap(sum_var, numSlots); 
        // evalSqrtTaylor(sum_var, keySwitchingKey);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 6 " << std::endl;
            std::cout << "#: " << sum_var.getLevel() << " " << sum_var.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            sum_var.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        ctxt.sub(sum); // ctxt - mean

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 7 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Plaintext weight_masked(cc);
        weight_masked.copy(weight);
        weight_masked.multPt(weight_masked, mask_row, true);
        ctxt.multPt(weight_masked);


        ////////////////////////////////////
        if (print) {
            std::cout << "Step 8 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        Ciphertext tmp_bias(cc);
        tmp_bias.copy(sum_var); // sqrt(var)

        Plaintext bias_masked(cc);
        bias_masked.copy(bias);
        bias_masked.multPt(bias_masked, mask_row, true);

        tmp_bias.multPt(bias_masked);
        ctxt.add(tmp_bias);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 9 " << std::endl;
            std::cout << "#: " << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            ctxt.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

        factor.copy(sum_var);

        ////////////////////////////////////
        if (print) {
            std::cout << "Step 10 " << std::endl;
            std::cout << "#: " << factor.getLevel() << " " << factor.NoiseLevel << std::endl;            
            FIDESlib::CKKS::RawCipherText raw_res;
            factor.store(cc, raw_res);
            auto result2(ctxt_cpu->Clone());
            GetOpenFHECipherText(result2, raw_res);

            lbcrypto::Plaintext result_pt;
            context->Decrypt(privateKey, result2, &result_pt);
            result_pt->SetLength(4);
            std::cout << result_pt << std::endl;
        }
        ////////////////////////////////////

    }


    void EvalLayerNorm_Matrix_DelayedInv(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
                            const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                            const int bStepAcc, bool bts, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& factor) {

        factor.reserve(ctxt.size());
        for (size_t i = 0; i < ctxt.size(); i++) {
            std::vector<FIDESlib::CKKS::Ciphertext> row;
            row.reserve(ctxt[0].size());
            for (size_t j = 0; j < ctxt[0].size(); j++) {
                FIDESlib::CKKS::Ciphertext scale_factor(ctxt[0][0].cc);
                EvalLayerNorm_DelayedInv(ctxt[i][j], ctxt_cpu, privateKey, keySwitchingKey, mask_ln, mask_row, numSlots, blockSize, weight[i][j], bias[i][j], bts, bStepAcc, scale_factor);
                row.emplace_back(std::move(scale_factor));
            }
            factor.emplace_back(std::move(row));
        }
    }
}
