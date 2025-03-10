//
// Created by carlosad on 21/10/24.
//
#ifndef GPUCKKS_NTTFUSIONS_CUH
#define GPUCKKS_NTTFUSIONS_CUH

#include "ConstantsGPU.cuh"
#include "NTT.cuh"
#include "NTThelper.cuh"

namespace FIDESlib {
template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void mult_and_save_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                                     const T* c1, const T* c1tilde, T* res0, T* res1, const T* kska,
                                                     const T* kskb, const T* c0, const T* c0tilde) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {
        T in1[2], in2[2] = {0, 0}, ksk1[2], ksk2[2], c1_[2], c1tilde_[2], c0_[2], c0tilde_[2];

        if constexpr (sizeof(T) == 8) {
            ((int4*)c1_)[0] = ((int4*)c1)[OFFSET_2T(i)];
            ((int4*)c1tilde_)[0] = ((int4*)c1tilde)[OFFSET_2T(i)];
            in1[0] = modmult<algo>(c1_[0], c1tilde_[0], primeid);
            in1[1] = modmult<algo>(c1_[1], c1tilde_[1], primeid);
            A(i)[j] = in1[0];
            A(i)[j + 1] = in1[1];

            ((int4*)c0_)[0] = ((int4*)c0)[OFFSET_2T(i)];
            ((int4*)c0tilde_)[0] = ((int4*)c0tilde)[OFFSET_2T(i)];

            //  res1 = P * (c0 * c1' + c1 * c0') + kska * c1 * c1'

            in2[0] = modmult<algo>(c1_[0], c0tilde_[0], primeid);
            in2[1] = modmult<algo>(c1_[1], c0tilde_[1], primeid);
            in2[0] = modadd(in2[0], modmult<algo>(c1tilde_[0], c0_[0], primeid), primeid);
            in2[1] = modadd(in2[1], modmult<algo>(c1tilde_[1], c0_[1], primeid), primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
                                                                     /*
            if (OFFSET_2T(i) == 0 && primeid == 0)
                printf("Mult and save c1 %lu\n", in2[0]);
*/
            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modadd(in2[0], modmult<algo>(in1[0], ksk2[0], primeid), primeid);
            in2[1] = modadd(in2[1], modmult<algo>(in1[1], ksk2[1], primeid), primeid);
            ((int4*)res1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            //in2[0] = 0;
            //in2[1] = 0;
            // res0 = result: P * (c0 * c0') + kskb * c1 * c1'

            in2[0] = modmult<algo>(c0_[0], c0tilde_[0], primeid);
            in2[1] = modmult<algo>(c0_[1], c0tilde_[1], primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
                                                                     /*
            if (OFFSET_2T(i) == 0 && primeid == 0)
                printf("Mult and save c0 %lu\n", in2[0]);
*/
            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];
            in2[0] = modadd(modmult<algo>(in1[0], ksk1[0], primeid), in2[0], primeid);
            in2[1] = modadd(modmult<algo>(in1[1], ksk1[1], primeid), in2[1], primeid);

            ((int4*)res0)[OFFSET_2T(i)] = ((int4*)&in2)[0];

        } else {
        }
    }
}

template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void rotate_and_save_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                                       const T* c1, T* res0, T* res1, const T* kska, const T* kskb,
                                                       const T* c0) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {
        T in2[2] = {0, 0}, ksk1[2], ksk2[2], c1_[2], c0_[2];

        if constexpr (sizeof(T) == 8) {
            ((int4*)c1_)[0] = ((int4*)c1)[OFFSET_2T(i)];
            A(i)[j] = c1_[0];
            A(i)[j + 1] = c1_[1];

            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modmult<algo>(c1_[0], ksk2[0], primeid);
            in2[1] = modmult<algo>(c1_[1], ksk2[1], primeid);
            ((int4*)res1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            ((int4*)c0_)[0] = ((int4*)c0)[OFFSET_2T(i)];

            in2[0] = modmult<algo>(c0_[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(c0_[1], C_.P[primeid], primeid);  // TODO shoup

            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];
            in2[0] = modadd(modmult<algo>(c1_[0], ksk1[0], primeid), in2[0], primeid);
            in2[1] = modadd(modmult<algo>(c1_[1], ksk1[1], primeid), in2[1], primeid);

            ((int4*)res0)[OFFSET_2T(i)] = ((int4*)&in2)[0];
        } else {
        }
    }
}

template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void square_and_save_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                                       const T* c1, T* res0, T* res1, const T* kska, const T* kskb,
                                                       const T* c0) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {
        T in2[2] = {0, 0}, in1[2] = {0, 0}, ksk1[2], ksk2[2], c1_[2], c0_[2];

        if constexpr (sizeof(T) == 8) {

            ((int4*)c1_)[0] = ((int4*)c1)[OFFSET_2T(i)];

            in1[0] = modmult<algo>(c1_[0], c1_[0], primeid);
            in1[1] = modmult<algo>(c1_[1], c1_[1], primeid);
            A(i)[j] = in1[0];
            A(i)[j + 1] = in1[1];

            ((int4*)c0_)[0] = ((int4*)c0)[OFFSET_2T(i)];
            in2[0] = modmult<algo>(c0_[0], c0_[0], primeid);
            in2[1] = modmult<algo>(c0_[1], c0_[1], primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];

            in2[0] = modadd(modmult<algo>(in1[0], ksk1[0], primeid), in2[0], primeid);
            in2[1] = modadd(modmult<algo>(in1[1], ksk1[1], primeid), in2[1], primeid);
            ((int4*)res0)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modmult<algo>(c1_[0], c0_[0], primeid);
            in2[1] = modmult<algo>(c1_[1], c0_[1], primeid);
            in2[0] = modadd(in2[0], in2[0], primeid);
            in2[1] = modadd(in2[1], in2[1], primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
            in2[0] = modadd(modmult<algo>(in1[0], ksk2[0], primeid), in2[0], primeid);
            in2[1] = modadd(modmult<algo>(in1[1], ksk2[1], primeid), in2[1], primeid);
            ((int4*)res1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

        } else {
        }
    }
}

template <typename T, ALGO algo_ = ALGO_SHOUP, int M>
__device__ __forceinline__ void mult_and_acc_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                                    const T* c1, const T* c1tilde, T* res0, T* res1, const T* kska,
                                                    const T* kskb, const T* c0, const T* c0tilde) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {

        T in1[2], in2[2], ksk1[2], ksk2[2], c1_[2], c1tilde_[2], c0_[2], c0tilde_[2], r0[2], r1[2];

        if constexpr (sizeof(T) == 8) {
            ((int4*)c1_)[0] = ((int4*)c1)[OFFSET_2T(i)];
            ((int4*)c1tilde_)[0] = ((int4*)c1tilde)[OFFSET_2T(i)];
            in1[0] = modmult<algo>(c1_[0], c1tilde_[0], primeid);
            in1[1] = modmult<algo>(c1_[1], c1tilde_[1], primeid);
            A(i)[j] = in1[0];
            A(i)[j + 1] = in1[1];

            ((int4*)c0_)[0] = ((int4*)c0)[OFFSET_2T(i)];
            ((int4*)c0tilde_)[0] = ((int4*)c0tilde)[OFFSET_2T(i)];

            //  res1 = P * (c0 * c1' + c1 * c0') + kska * c1 * c1'

            in2[0] = modmult<algo>(c1_[0], c0tilde_[0], primeid);
            in2[1] = modmult<algo>(c1_[1], c0tilde_[1], primeid);
            in2[0] = modadd(in2[0], modmult<algo>(c1tilde_[0], c0_[0], primeid), primeid);
            in2[1] = modadd(in2[1], modmult<algo>(c1tilde_[1], c0_[1], primeid), primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
                                                                     /*
            if (OFFSET_2T(i) == 0 && primeid == 0)
                printf("Mult and save c1 %lu\n", in2[0]);
*/
            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modadd(in2[0], modmult<algo>(in1[0], ksk2[0], primeid), primeid);
            in2[1] = modadd(in2[1], modmult<algo>(in1[1], ksk2[1], primeid), primeid);

            ((int4*)r1)[0] = ((int4*)res1)[OFFSET_2T(i)];
            in2[0] = modadd(in2[0], r1[0], primeid);
            in2[1] = modadd(in2[1], r1[1], primeid);
            ((int4*)res1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            //in2[0] = 0;
            //in2[1] = 0;
            // res0 = result: P * (c0 * c0') + kskb * c1 * c1'

            in2[0] = modmult<algo>(c0_[0], c0tilde_[0], primeid);
            in2[1] = modmult<algo>(c0_[1], c0tilde_[1], primeid);
            in2[0] = modmult<algo>(in2[0], C_.P[primeid], primeid);  // TODO shoup
            in2[1] = modmult<algo>(in2[1], C_.P[primeid], primeid);  // TODO shoup
                                                                     /*
            if (OFFSET_2T(i) == 0 && primeid == 0)
                printf("Mult and save c0 %lu\n", in2[0]);
*/
            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];
            in2[0] = modadd(modmult<algo>(in1[0], ksk1[0], primeid), in2[0], primeid);
            in2[1] = modadd(modmult<algo>(in1[1], ksk1[1], primeid), in2[1], primeid);

            ((int4*)r0)[0] = ((int4*)res0)[OFFSET_2T(i)];
            in2[0] = modadd(in2[0], r0[0], primeid);
            in2[1] = modadd(in2[1], r0[1], primeid);
            ((int4*)res0)[OFFSET_2T(i)] = ((int4*)&in2)[0];
        } else {
        }
    }
}

template <typename T, ALGO algo_ = ALGO_SHOUP, int M>
__device__ __forceinline__ void rescale_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                               const int primeid_rescale, const T* res) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    const T q_inv_temp = G_::q_inv[MAXP * primeid_rescale + primeid];
    const T QlQlInvModqlDivqlModq_temp = G_::QlQlInvModqlDivqlModq[MAXP * primeid_rescale + primeid];

    for (int i = 0; i < M; i += 1) {
        T* A = (T*)(buffer + (i << (logBD)));

        T in[2] = {res[OFFSET_T(i)], res[OFFSET_T(i) | 1]};
        assert(primeid_rescale >= 0);
        if constexpr (1) {  // OpenFHE style rescale
            A[j] = modadd(modmult<algo>(q_inv_temp, in[0], primeid),
                          modmult<algo>(QlQlInvModqlDivqlModq_temp, A[j], primeid), primeid);
            A[j | 1] = modadd(modmult<algo>(q_inv_temp, in[1], primeid),
                              modmult<algo>(QlQlInvModqlDivqlModq_temp, A[j | 1], primeid), primeid);
        } else {
            A[j] = modmult<algo>(q_inv_temp, modsub(in[0], A[j], primeid), primeid);
            A[j | 1] = modmult<algo>(q_inv_temp, modsub(in[1], A[j | 1], primeid), primeid);
        }
    }
}

template <typename T, ALGO algo, int M>
__device__ __forceinline__ void moddown_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                               const T* res) {
    for (int i = 0; i < M; i += 1) {
        T* A = (T*)(buffer + (i << (logBD)));

        T in[2] = {res[OFFSET_T(i)], res[OFFSET_T(i) | 1]};

        if (algo != ALGO_SHOUP) {
            A[j] = modmult<algo>(modsub(in[0], A[j], primeid), (T)C_.P_inv[primeid], primeid);
            A[j | 1] = modmult<algo>(modsub(in[1], A[j | 1], primeid), (T)C_.P_inv[primeid], primeid);
        } else {
            A[j] =
                modmult<algo>(modsub(in[0], A[j], primeid), (T)C_.P_inv[primeid], primeid, (T)C_.P_inv_shoup[primeid]);
            A[j | 1] = modmult<algo>(modsub(in[1], A[j | 1], primeid), (T)C_.P_inv[primeid], primeid,
                                     (T)C_.P_inv_shoup[primeid]);
        }
    }
}

template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void multpt_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                              const int primeid_rescale, const T* res, const T* pt) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    const T q_inv_temp = G_::q_inv[MAXP * primeid_rescale + primeid];
    const T QlQlInvModqlDivqlModq_temp = G_::QlQlInvModqlDivqlModq[MAXP * primeid_rescale + primeid];

    for (int i = 0; i < M; i += 1) {
        T* A = (T*)(buffer + (i << (logBD)));

        T in[2] = {res[OFFSET_T(i)], res[OFFSET_T(i) | 1]};

        in[0] = modmult<algo>(in[0], pt[OFFSET_T(i)], primeid);
        in[1] = modmult<algo>(in[1], pt[OFFSET_T(i) | 1], primeid);

        assert(primeid_rescale >= 0);
        if constexpr (1) {  // OpenFHE style rescale
            A[j] = modadd(modmult<algo>(q_inv_temp, in[0], primeid),
                          modmult<algo>(QlQlInvModqlDivqlModq_temp, A[j], primeid), primeid);
            A[j | 1] = modadd(modmult<algo>(q_inv_temp, in[1], primeid),
                              modmult<algo>(QlQlInvModqlDivqlModq_temp, A[j | 1], primeid), primeid);
        } else {
            A[j] = modmult<algo>(q_inv_temp, modsub(in[0], A[j], primeid), primeid);
            A[j | 1] = modmult<algo>(q_inv_temp, modsub(in[1], A[j | 1], primeid), primeid);
        }
    }
}

template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void ksk_dot_fusion(char* buffer, const int logBD, const int j, const int primeid, T* c0,
                                               T* c1, const T* kska, const T* kskb) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {
        T* A = (T*)(buffer + (i << (logBD)));

        T ksk1[2], ksk2[2], in2[2], in1[2];

        if constexpr (sizeof(T) == 8) {

            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modmult<algo>(A[j], ksk2[0], primeid);
            in2[1] = modmult<algo>(A[j + 1], ksk2[1], primeid);
            ((int4*)c1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];
            in1[0] = modmult<algo>(A[j], ksk1[0], primeid);
            in1[1] = modmult<algo>(A[j + 1], ksk1[1], primeid);
            ((int4*)c0)[OFFSET_2T(i)] = ((int4*)&in1)[0];

        } else {
        }
    }
}

template <typename T, ALGO algo_, int M>
__device__ __forceinline__ void ksk_dot_acc_fusion(char* buffer, const int logBD, const int j, const int primeid,
                                                   T* res0, T* res1, const T* kska, const T* kskb) {
    constexpr ALGO algo = algo_ == ALGO_SHOUP ? ALGO_BARRETT : algo_;

    for (int i = 0; i < M; i += 1) {
        T* A = (T*)(buffer + (i << (logBD)));

        T ksk1[2], ksk2[2], in2[2], in1[2], r0[2], r1[2];

        if constexpr (sizeof(T) == 8) {

            ((int4*)ksk2)[0] = ((int4*)kska)[OFFSET_2T(i)];
            in2[0] = modmult<algo>(A[j], ksk2[0], primeid);
            in2[1] = modmult<algo>(A[j + 1], ksk2[1], primeid);
            ((int4*)r1)[0] = ((int4*)res1)[OFFSET_2T(i)];
            in2[0] = modadd(in2[0], r1[0], primeid);
            in2[1] = modadd(in2[1], r1[1], primeid);
            ((int4*)res1)[OFFSET_2T(i)] = ((int4*)&in2)[0];

            ((int4*)ksk1)[0] = ((int4*)kskb)[OFFSET_2T(i)];
            in1[0] = modmult<algo>(A[j], ksk1[0], primeid);
            in1[1] = modmult<algo>(A[j + 1], ksk1[1], primeid);
            ((int4*)r0)[0] = ((int4*)res0)[OFFSET_2T(i)];
            in1[0] = modadd(in1[0], r0[0], primeid);
            in1[1] = modadd(in1[1], r0[1], primeid);
            ((int4*)res0)[OFFSET_2T(i)] = ((int4*)&in1)[0];
        } else {
        }
    }
}

template <typename T, ALGO algo, int M>
__device__ __forceinline__ void forward_negacyclic_scale(char* buffer, const int primeid, T* psi, T* psi_shoup) {

    const int tid = threadIdx.x;

    if constexpr (0) {  // High bandwidth
        for (int i = 0; i < M; i += 1) {
            A(i)
            [tid] = modmult<ALGO_BARRETT>(
                A(i)[tid], ((T*)G_::psi_no[primeid])[tid * (gridDim.x * M) + M * blockIdx.x + i], primeid);
            A(i)
            [tid + blockDim.x] = modmult<ALGO_BARRETT>(
                A(i)[tid + blockDim.x],
                ((T*)G_::psi_no[primeid])[(tid + blockDim.x) * (gridDim.x * M) + M * blockIdx.x + i], primeid);
        }
    } else if constexpr (0) {
        // Now, try to load this from bit-reversed psi array TODO
        T aux[2] = {((T*)G_::psi_no[primeid])[tid * (2 * blockDim.x) + M * blockIdx.x],
                    ((T*)G_::psi_no[primeid])[(tid + blockDim.x) * (2 * blockDim.x) + M * blockIdx.x]};
        T root = C_.root;
        for (int i = 0; i < M; i += 1) {
            if (i > 0) {
                aux[0] = modmult<4>(aux[0], root, primeid);
                aux[1] = modmult<4>(aux[1], root, primeid);
            }
            A(i)[tid] = modmult<4>(A(i)[tid], aux[0], primeid);
            A(i)[tid + blockDim.x] = modmult<4>(A(i)[tid + blockDim.x], aux[1], primeid);
        }
    } else {
        const uint32_t logBD = __clz(blockDim.x);
        // Now, try to load this from bit-reversed psi array
        uint32_t pos1 = tid & (~1);
        pos1 = __brev(pos1);
        pos1 >>= (logBD);

        T aux_3 = ((T*)G_::psi_no[primeid])[(tid & 1) * (gridDim.x * M) + M * blockIdx.x];

        T aux;
        if constexpr (algo == ALGO_SHOUP) {
            aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_shoup[pos1]);
        } else {
            aux = modmult<algo>(aux_3, psi[pos1], primeid);
        }

        const T root = C_.root[primeid];
        T root_shoup;
        if constexpr (algo == ALGO_SHOUP)
            root_shoup = C_.root_shoup[primeid];

        const T fourth_root = psi[1];
        T fourth_root_shoup;
        if constexpr (algo == ALGO_SHOUP)
            fourth_root_shoup = psi_shoup[1];

        for (int i = 0; i < M; i += 1) {
            if (i > 0) {
                //aux = modmult<4>(aux, root, primeid);
                if constexpr (algo == ALGO_SHOUP) {
                    aux = modmult<algo>(aux, root, primeid, root_shoup);
                } else {
                    aux = modmult<algo>(aux, root, primeid);
                }
            }

            T aux2;
            if constexpr (algo == ALGO_SHOUP) {
                aux2 = modmult<algo>(aux, fourth_root, primeid, fourth_root_shoup);
            } else {
                aux2 = modmult<algo>(aux, fourth_root, primeid);
            }

            A(i)[tid] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid], aux, primeid);
            A(i)[tid + blockDim.x] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid + blockDim.x], aux2, primeid);
        }
    }
}

template <typename T, ALGO algo, int M>
__device__ __forceinline__ void backward_negacyclic_scale(char* buffer, const int primeid, T* psi, T* psi_shoup) {

    const int tid = threadIdx.x;

    if constexpr (0) {  // High bandwidth
        for (int i = 0; i < M; i += 1) {
            A(i)
            [tid] = modmult<ALGO_BARRETT>(
                A(i)[tid], ((T*)G_::inv_psi_no[primeid])[tid * (gridDim.x * M) + M * blockIdx.x + i], primeid);
            A(i)
            [tid + blockDim.x] = modmult<ALGO_BARRETT>(
                A(i)[tid + blockDim.x],
                ((T*)G_::inv_psi_no[primeid])[(tid + blockDim.x) * (gridDim.x * M) + M * blockIdx.x + i], primeid);
            A(i)[tid] = modmult<ALGO_SHOUP>(A(i)[tid], C_.N, primeid, C_.N_shoup[primeid]);
            A(i)[tid + blockDim.x] = modmult<ALGO_SHOUP>(A(i)[tid + blockDim.x], C_.N, primeid, C_.N_shoup[primeid]);
        }
    } else if constexpr (0) {
        // Now, try to load this from bit-reversed psi array TODO
        T aux[2] = {((T*)G_::psi_no[primeid])[tid * (2 * blockDim.x) + M * blockIdx.x],
                    ((T*)G_::psi_no[primeid])[(tid + blockDim.x) * (2 * blockDim.x) + M * blockIdx.x]};
        T root = C_.root;
        for (int i = 0; i < M; i += 1) {
            if (i > 0) {
                aux[0] = modmult<4>(aux[0], root, primeid);
                aux[1] = modmult<4>(aux[1], root, primeid);
            }
            A(i)[tid] = modmult<4>(A(i)[tid], aux[0], primeid);
            A(i)[tid + blockDim.x] = modmult<4>(A(i)[tid + blockDim.x], aux[1], primeid);
        }
    } else {
        const uint32_t logBD = __clz(blockDim.x);
        // Now, try to load this from bit-reversed psi array
        uint32_t pos1 = tid & (~1);
        pos1 = __brev(pos1);
        pos1 >>= (logBD);

        T aux_3 = ((T*)G_::inv_psi_no[primeid])[(tid & 1) * (gridDim.x * M) + M * blockIdx.x];

        aux_3 = modmult<ALGO_SHOUP>(aux_3, C_.N, primeid, C_.N_shoup[primeid]);

        T aux;
        if constexpr (algo == ALGO_SHOUP) {
            aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_shoup[pos1]);
        } else {
            aux = modmult<algo>(aux_3, psi[pos1], primeid);
        }

        const T root = C_.inv_root[primeid];
        T root_shoup;
        if constexpr (algo == ALGO_SHOUP)
            root_shoup = C_.inv_root_shoup[primeid];

        T fourth_root = psi[1];
        T fourth_root_shoup;
        if constexpr (algo == ALGO_SHOUP)
            fourth_root_shoup = psi_shoup[1];

        if constexpr (algo == ALGO_SHOUP) {
            assert(fourth_root_shoup == ((T*)G_::inv_psi_shoup[primeid])[1]);
        }
        assert(fourth_root == ((T*)G_::inv_psi[primeid])[1]);

        //fourth_root = ((T*)G_::psi[primeid])[1];

        for (int i = 0; i < M; i += 1) {
            if (i > 0) {
                if constexpr (algo == ALGO_SHOUP) {
                    aux = modmult<algo>(aux, root, primeid, root_shoup);
                } else {
                    aux = modmult<algo>(aux, root, primeid);
                }
            }

            T aux2;
            if constexpr (algo == ALGO_SHOUP) {
                aux2 = modmult<algo>(aux, fourth_root, primeid, fourth_root_shoup);
            } else {
                aux2 = modmult<algo>(aux, fourth_root, primeid);
            }

            A(i)[tid] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid], aux, primeid);
            A(i)[tid + blockDim.x] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid + blockDim.x], aux2, primeid);
        }
    }
}
}  // namespace FIDESlib
#endif  //GPUCKKS_NTTFUSIONS_CUH
