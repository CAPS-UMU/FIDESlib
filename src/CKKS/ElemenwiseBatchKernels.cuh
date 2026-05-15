//
// Created by carlosad on 27/09/24.
//

#ifndef GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
#define GPUCKKS_ELEMENWISEBATCHKERNELS_CUH

#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int partition, void** l, void** l1, void** l2, void** l3, void** l4);

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0, void** bc1, void** in, void** aux);

__global__ void mult1Add2_(const __grid_constant__ int partition, void** l, void** l1, void** l2);

template <typename T> __global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid);

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init);

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1, const __grid_constant__ int primeid_init);
template <ALGO algo> __global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid, const uint64_t* shoup_mu);

__global__ void broadcastLimb0_(void** a);
__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0);
__global__ void copy_(void** src, void** dst);
__global__ void copy1D_(void* a, void* b);
__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w, const __grid_constant__ int primeid_init);

__global__ void fusedDotKSK_2_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id, int num_special, int init);
__global__ void hoistedRotateDotKSK_2_(void*** din1,
  void** c0,
  void*** out1,
  void*** sout1,
  void*** out2,
  void*** sout2,
  int n,
  const int* indexes,
  void*** digits,
  int num_d,
  int id,
  int num_special,
  int init,
  void** sc0,
  bool c0_modup);
__global__ void hoistedRotateDotKSKBatched___(void*** in1,
  void*** din1,
  void*** c0,
  void*** sc0,
  void*** out1,
  void*** sout1,
  void*** out2,
  void*** sout2,
  int n,
  const int* indexes,
  void*** digits,
  int num_d,
  int id,
  int num_special,
  int init,
  bool c0_modup);

__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit, const int n);

__global__ void binomialMult_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0, void** d1);
__global__ void binomialMultExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0, void** d1);
__global__ void binomialSquare_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2);
__global__ void binomialSquareExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2);

__global__ void
dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int batch, const int gStep, const int primeidInit, const int n);
__global__ void
dotProductLtBatchedPt2___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int bStep, const int gStep, const int primeidInit, const int n);
__global__ void
dotProductLtBatchedPt3___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int bStep, const int gStep, const int primeidInit, const int n);

__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init);
__global__ void scaleByP_(void** a, const int primeid_init);

__global__ void add___(void*** a, const int primeid_init, const int n);
__global__ void add_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void mult_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void add_scalar_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void mult_scalar_reuse_b___(void*** a, void*** b, void*** b_shoup, const int primeid_init, const int n, const int its);
__global__ void add_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void add_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void copy_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void copy_reuse_negative_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);

__global__ void
binomialDotProdBatched___(const __grid_constant__ int primeid_init, void*** c0, void*** c1, void*** d0, void*** d1, void*** c0_out, void*** c1_out, void*** c2_out, int its, int n, bool ext_in);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
