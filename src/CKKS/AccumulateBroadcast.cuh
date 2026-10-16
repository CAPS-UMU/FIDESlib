//
// Created by carlosad on 12/06/25.
//

#ifndef ACCUMULATEBROADCAST_CUH
#define ACCUMULATEBROADCAST_CUH
#include "CKKS/Ciphertext.cuh"
#include <vector>

namespace FIDESlib::CKKS {
/**
 * Compute rotation indices for a rotate and accumulate where:
 *
 *  out_i = \sum_{j = 0}^{size - 1} in_{i%size + j*size}
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, stride, size - should be powers of two
 */
std::vector<int> GetAccumulateRotationIndices(const int bStep, const int stride, const int size);

/**
 * Compute rotation indices for a rotate and broadcast where:
 *
 *  in = {a_0, ... a_initsize, 0, ..., 0}
 *
 *  out_i = in_{i%initsize} iff i < outsize
 *
 *  out_i = 0 iff i >= outsize
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, initsize, outsize - should be powers of two
 */
std::vector<int> GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize);

/**
 * Compute a rotate and accumulate where:
 *
 *  out_i = \sum_{j = 0}^{size - 1} in_{i%size + j*size}
 *
 *  with a bsgs algorithm using baby-step size bStep.
 *
 *  When `startFactor` > 1 the fold starts at factor `startFactor` instead of 1: level s rotates
 *  the current accumulator by {stride*s, 2*stride*s, ..., < min(stride*size, bStep*stride*s)}
 *  for s = startFactor, startFactor*bStep, ... — the start-offset variant used by
 *  `AccumulateSumInPlace(ct, slots, stride, start)`. It follows the same P2b lazy discipline as
 *  the plain fold and is otherwise identical, so start=1 api calls behave the same regardless of
 *  which overload is used.
 *
 *  The fold is metadata-neutral: it never rewrites `ctxt.slots` (O10). Callers that need the
 *  sparse re-interpretation of a full fold (e.g. the bootstrap PartialSum) set `ctxt.slots`
 *  themselves.
 *
 *  bStep, stride, size and startFactor - should be powers of two
 */
void Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size, const int startFactor = 1);

/**
 *  Compute a rotate and broadcast where:
 *  in = {a_0, ... a_initsize, 0, ..., 0}
 *
 *  out_i = in_{i%initsize} iff i < outsize
 *
 *  out_i = 0 iff i >= outsize
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, initsize, outsize - should be powers of two
 */
void Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize);
} // namespace FIDESlib::CKKS
#endif // ACCUMULATEBROADCAST_CUH