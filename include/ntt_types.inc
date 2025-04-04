//
// Created by carlosad on 21/10/24.
//
#ifdef X
X(uint64_t)
X(uint32_t)
#endif
#ifdef Y
Y(uint64_t, FIDESlib::ALGO::ALGO_NATIVE)
Y(uint64_t, FIDESlib::ALGO::ALGO_NONE)
Y(uint64_t, FIDESlib::ALGO::ALGO_SHOUP)
Y(uint64_t, FIDESlib::ALGO::ALGO_BARRETT)
Y(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64)
Y(uint32_t, FIDESlib::ALGO::ALGO_NATIVE)
Y(uint32_t, FIDESlib::ALGO::ALGO_NONE)
Y(uint32_t, FIDESlib::ALGO::ALGO_SHOUP)
Y(uint32_t, FIDESlib::ALGO::ALGO_BARRETT)
Y(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64)
#endif
#ifdef YY
YY(FIDESlib::ALGO::ALGO_NATIVE)
YY(FIDESlib::ALGO::ALGO_NONE)
YY(FIDESlib::ALGO::ALGO_SHOUP)
YY(FIDESlib::ALGO::ALGO_BARRETT)
YY(FIDESlib::ALGO::ALGO_BARRETT_FP64)
#endif
#ifdef YYY
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
YYY(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
YYY(FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
YYY(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
YYY(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
YYY(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
#endif
#ifdef Z
Z(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE)
Z(uint64_t, true, FIDESlib::ALGO::ALGO_NONE)
Z(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP)
Z(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT)
Z(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64)
Z(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE)
Z(uint32_t, true, FIDESlib::ALGO::ALGO_NONE)
Z(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP)
Z(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT)
Z(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64)
Z(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE)
Z(uint64_t, false, FIDESlib::ALGO::ALGO_NONE)
Z(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP)
Z(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT)
Z(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64)
Z(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE)
Z(uint32_t, false, FIDESlib::ALGO::ALGO_NONE)
Z(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP)
Z(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT)
Z(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64)
#endif
#ifdef VV
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VV(uint64_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint64_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint64_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint64_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint32_t, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint32_t, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint32_t, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VV(uint32_t, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
#endif
#ifdef V
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
V(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
#endif
#ifdef VVV
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_NONE)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_NONE)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_NONE)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_NONE)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_NONE)
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_RESCALE)
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MULTPT)
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_MODDOWN)
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT)
VVV(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
VVV(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::NTT_MODE::NTT_KSK_DOT_ACC)
#endif

#ifdef W
W(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)

W(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)

W(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)

W(uint64_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint64_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
W(uint32_t, false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
#endif
#ifdef WW
WW(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
WW(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
WW(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
WW(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
WW(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
WW(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
WW(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)

WW(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_ROTATE_AND_SAVE)

WW(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)
WW(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_SQUARE_AND_SAVE)

WW(true, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(true, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(true, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(true, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(true, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(false, FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(false, FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(false, FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(false, FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WW(false, FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
#endif

#ifdef WWW
WWW(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_NONE)
WWW(FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_NONE)
WWW(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_NONE)
WWW(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_NONE)
WWW(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_NONE)
WWW(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WWW(FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WWW(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WWW(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WWW(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_SAVE)
WWW(FIDESlib::ALGO::ALGO_NATIVE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WWW(FIDESlib::ALGO::ALGO_NONE, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WWW(FIDESlib::ALGO::ALGO_SHOUP, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WWW(FIDESlib::ALGO::ALGO_BARRETT, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
WWW(FIDESlib::ALGO::ALGO_BARRETT_FP64, FIDESlib::INTT_MODE::INTT_MULT_AND_ACC)
#endif
