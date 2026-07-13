//
// Created by carlosad on 14/09/25.
//

#include "CKKS/openfhe-interface/ParameterSwitch.cuh"

namespace FIDESlib {
namespace CKKS {
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createSwitchableContextBasedOnContext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int limbs, int digits, int hamming_weight) {
	std::shared_ptr<lbcrypto::CryptoParametersCKKSRNS> init_param = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
	auto& init_encode_param										  = init_param->GetEncodingParams();
	auto& init_elem_param										  = init_param->GetElementParams();
	auto& init_elem_P_param										  = init_param->GetParamsP();
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_res;

	lbcrypto::CryptoParametersCKKSRNS param{ *init_param };
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	// parameters.SetNoiseEstimate();
	parameters.SetBatchSize(init_encode_param->GetBatchSize());
	parameters.SetDecryptionNoiseMode(param.GetDecryptionNoiseMode());
	// parameters.SetDesiredPrecision();
	parameters.SetDigitSize(digits);
	parameters.SetExecutionMode(param.GetExecutionMode());
	parameters.SetFirstModSize(init_elem_param->GetParams().at(0)->GetModulus().GetMSB());
	parameters.SetInteractiveBootCompressionLevel(param.GetMPIntBootCiphertextCompressionLevel() /* param.m_MPIntBootCiphertextCompressionLevel*/);
	parameters.SetKeySwitchTechnique(init_param->GetKeySwitchTechnique());
	parameters.SetMaxRelinSkDeg(init_param->GetMaxRelinSkDeg());
	parameters.SetMultiplicativeDepth(limbs - 1);

	parameters.SetNumAdversarialQueries(param.GetNumAdversarialQueries());
	parameters.SetNumLargeDigits(param.GetNumPartQ());
	parameters.SetPREMode(param.GetPREMode());
	parameters.SetRingDim(param.GetElementParams()->GetRingDimension());
	auto elem = init_elem_param->GetParams().at(1)->GetModulus();
	elem.Add(elem.RShift(1)); // To make the modulo greater than 2^scale and not just \approx 2^scale
	int scale = elem.GetMSB() - 1;
	parameters.SetScalingModSize(scale);
	parameters.SetScalingTechnique(param.GetScalingTechnique());
	parameters.SetSecretKeyDist(static_cast<uint32_t>(hamming_weight) == cc->GetRingDimension() / 2 ? lbcrypto::UNIFORM_TERNARY : lbcrypto::SPARSE_TERNARY);
	parameters.SetSecurityLevel(param.GetStdLevel());
	// parameters.SetStandardDeviation()
	parameters.SetStatisticalSecurity(param.GetStatisticalSecurity());

	cc_res = GenCryptoContext(parameters);
	cc_res->Enable(lbcrypto::PKE | lbcrypto::KEYSWITCH | lbcrypto::LEVELEDSHE);

	return cc_res;
}

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>, std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
  std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
  const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& a,
  int hamming_weight_b) {
	lbcrypto::DCRTPoly::TugType tug;
	lbcrypto::DCRTPoly sNew(tug, cca->GetElementParams(), Format::EVALUATION, hamming_weight_b);
	// sparse key used for the modraising step
	auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
	skNew->SetPrivateElement(std::move(sNew));

	// lbcrypto::PrivateKeyImpl < lbcrypto::DCRTPoly >> nkNewLow(ccb);
	// (*skNew);
	skNew->SetKeyTag(a->GetKeyTag());
	auto scaling = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cca->GetCryptoParameters())->GetScalingTechnique();

	std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> atob;
	if (scaling != lbcrypto::FLEXIBLEAUTOEXT) {
		atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(ccb->GetScheme()->KeySwitchGen(a, skNew));
	} else {
		lbcrypto::DCRTPoly saNew = a->GetPrivateElement();
		auto& params			 = ccb->GetElementParams()->GetParams();
		auto& paramsa			 = cca->GetElementParams()->GetParams();

		saNew.SetElementAtIndex(params.size() - 1, saNew.GetAllElements().at(paramsa.size() - 1));
		saNew.DropLastElements(saNew.GetAllElements().size() - ccb->GetElementParams()->GetParams().size());
		saNew.SwitchModulusAtIndex(params.size() - 1, params.back()->GetModulus(), params.back()->GetRootOfUnity());

		auto skaNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cca);
		skaNew->SetPrivateElement(std::move(saNew));
		skaNew->SetKeyTag(a->GetKeyTag());

		lbcrypto::DCRTPoly sbNew = skNew->GetPrivateElement();
		sbNew.SetElementAtIndex(params.size() - 1, sbNew.GetAllElements().at(params.size() - 1));
		sbNew.DropLastElements(sbNew.GetAllElements().size() - ccb->GetElementParams()->GetParams().size());
		sbNew.SwitchModulusAtIndex(params.size() - 1, params.back()->GetModulus(), params.back()->GetRootOfUnity());

		auto skbNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
		skbNew->SetPrivateElement(std::move(sbNew));
		skbNew->SetKeyTag(a->GetKeyTag());

		atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(ccb->GetScheme()->KeySwitchGen(skaNew, skbNew));
	}
	std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> btoa =
	  std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(cca->GetScheme()->KeySwitchGen(skNew, a));

	return { { atob, btoa }, skNew };
}

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>, std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
  std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
  const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& a,
  int hamming_weight_b) {
	return createContextSwitchingKeys(cca, ccb, a.secretKey, hamming_weight_b);
}

} // namespace CKKS
} // namespace FIDESlib