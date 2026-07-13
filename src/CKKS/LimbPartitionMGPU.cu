//
// Created by carlosad on 8/06/25.
//

#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "PeerUtils.cuh"
#include "parallel_for.hpp"

namespace FIDESlib::CKKS {

static bool envMemcopyPeer(bool def = false) {
	bool out = def;

	char* res = getenv("FIDESLIB_USE_MEMCPY_PEER");
	if (res && !(0 == std::strcmp(res, ""))) {
		out = atoi(res);
		std::cout << "FIDESLIB_USE_MEMCPY_PEER=" << res << ", set to: " << out << std::endl;
	}
	return out;
}

static bool envPeerAccess(bool def = false) {
	bool out = def;

	char* res = getenv("FIDESLIB_USE_PEER_ACCESS");
	if (res && !(0 == std::strcmp(res, ""))) {
		out = atoi(res);
		std::cout << "FIDESLIB_USE_PEER_ACCESS=" << res << ", set to: " << out << std::endl;
	}
	return out;
}

static bool envGraphCapture(bool def = false) {
	bool out = def;

	char* res = getenv("FIDESLIB_USE_GRAPH_CAPTURE");
	if (res && !(0 == std::strcmp(res, ""))) {
		out = atoi(res);
		std::cout << "FIDESLIB_USE_GRAPH_CAPTURE=" << res << ", set to: " << out << std::endl;
	}
	return out;
}

bool MEMCPY_PEER   = envMemcopyPeer(true);
bool GRAPH_CAPTURE = envGraphCapture(false);
bool PEER_ACCESS   = envPeerAccess(false);

void LimbPartition::rescaleMGPU() {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);

	Stream& stream		  = cc.top_limb_stream[id];
	uint64_t* buffer	  = cc.top_limb_buffer[id];
	VectorGPU<void*>& ptr = cc.top_limbptr[id];

	if (cc.limbGPUid[*level].x == static_cast<uint32_t>(id)) {
		if (limb.size() > cc.limbGPUid[*level].y && PRIMEID(limb[cc.limbGPUid[*level].y]) == *level) {

			LimbImpl& top = limb.at(limbsize - 1);
			stream.wait(s);
			// SWITCH(top, INTT<ALGO_SHOUP>());
			{
				constexpr ALGO algo = ALGO_SHOUP;
				constexpr int M		= 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				int start = 0;
				for (int i = limbsize - 1; i < limbsize; i += cc.batch) {
					cc.top_limb_stream.at(id).wait(s);
					uint32_t num_limbs = 1;

					INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, stream.ptr()>>>(
					  getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), cc.top_limbptr.at(cc.GPUid.size() + id).data);
					INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, stream.ptr()>>>(
					  getGlobals(), cc.top_limbptr.at(cc.GPUid.size() + id).data, PARTITION(id, start + i), cc.top_limbptr.at(id).data);
				}
				// s.wait(cc.top_limb_stream.at(id));
			}
			// STREAM(top).wait(stream);
			// cudaMemcpyAsync(buffer, std::get<U64>(top).v.data, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
			//                 STREAM(top).ptr());
			// stream.wait(STREAM(top));
			/*
			std::cout << "GPU: " << id << " ";
			SWITCH(top, printThisLimb(2));
			std::cout << std::endl;
*/
			// while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
			//     STREAM(limb.back()).wait(stream);
			//     limb.pop_back();
			// }
		} else {
			std::cout << "Cant find the top limb!!!" << id << " " << *level << " " << cc.limbGPUid[*level].y << " " << limb.size() << std::endl;
		}
	} else {
		// stream.wait(s);
	}

	if (cc.GPUid.size() > 1) {
		if (MEMCPY_PEER) {

			int top_gpu = cc.limbGPUid[*level].x;
			if (top_gpu == id)
				stream.record();

			openmp_synchronize();

			if (top_gpu != id) {
				for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
					if (i == static_cast<uint32_t>(id)) {
						// cudaSetDevice(cc.GPUid[i]);
						stream.wait(cc.top_limb_stream[top_gpu]);
						cudaMemcpyPeerAsync(buffer, cc.GPUid[i], cc.top_limb_buffer[top_gpu], cc.GPUid[top_gpu], cc.N * sizeof(uint64_t), stream.ptr());
					}
					CudaCheckErrorModNoSync;
				}
				cudaSetDevice(device);
			}
		} else {

#ifdef NCCL
			if constexpr (0) {
				NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[*level].x, rank, stream.ptr()));
			} else {
				NCCLCHECK(ncclGroupStart());
				if (static_cast<uint32_t>(id) == cc.limbGPUid[*level].x) {
					for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
						if (i != static_cast<uint32_t>(id))
							ncclSend(buffer, cc.N, ncclUint64, i, rank, stream.ptr());
					}
				} else {
					ncclRecv(buffer, cc.N, ncclUint64, cc.limbGPUid[*level].x, rank, stream.ptr());
				}
				NCCLCHECK(ncclGroupEnd());
			}
#else
			assert(false);
#endif
		}
	}
	/* {
		std::vector<uint64_t> data(cc.N, 0);
		cudaMemcpyAsync(data.data(), buffer, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.ptr());
		cudaDeviceSynchronize();
		std::cout << "GPU " << id << " (" << data[0] << " " << data[1] << ")" << std::endl;
	} */

	{
		{
			stream.wait(s);
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			const dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			const dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			const int bytesFirst	  = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
			const int bytesSecond	  = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
			const int size			  = getLimbSize(*level - 1);
			const int batch			  = cc.batch;
			const NTT_MODE mode		  = NTT_RESCALE;

			for (int i = 0; i < size; i += batch) {
				uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

				NTT_<false, algo, mode><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, stream.ptr()>>>(
				  getGlobals(), ptr.data, PARTITION(id, i), auxptr.data + i, nullptr, *level);

				s.wait(stream); // Data dependency on top_limb reaches only to here

				NTT_<true, algo, mode><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, s.ptr()>>>(
				  getGlobals(), auxptr.data + i, PARTITION(id, i), limbptr.data + i, nullptr, *level);
			}
		}
	}
	if (cc.GPUid.size() > 1) {
		openmp_synchronize();
		if (MEMCPY_PEER) {
			if (cc.limbGPUid[*level].x == static_cast<uint32_t>(id)) {
				for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
					if (i != static_cast<uint32_t>(id)) {
						stream.wait(cc.top_limb_stream[i]);
					}
				}
			}
		}
	}
}

void LimbPartition::doubleRescaleMGPU(LimbPartition& partition) {
	const int limbsize = getLimbSize(*level);
	assert(limbsize == getLimbSize(*partition.level));
	cudaSetDevice(device);

	Stream* stream[2] = { &cc.top_limb_stream[id], &cc.top_limb_stream2[id] };

	uint64_t* buffer[2] = { cc.top_limb_buffer[id], cc.top_limb_buffer2[id] };

	VectorGPU<void*>* ptr[2]	= { &cc.top_limbptr[id], &cc.top_limbptr2[id] };
	VectorGPU<void*>* ptraux[2] = { &cc.top_limbptr[cc.GPUid.size() + id], &cc.top_limbptr2[cc.GPUid.size() + id] };

	LimbPartition* part[2] = { this, &partition };
	// parity = !parity;

	for (int i = 0; i < 2; ++i) {
		if (cc.limbGPUid[*part[i]->level].x == static_cast<uint32_t>(id)) {
			if (part[i]->limb.size() > cc.limbGPUid[*part[i]->level].y && PRIMEID(part[i]->limb[cc.limbGPUid[*part[i]->level].y]) == *part[i]->level ||
			  (*part[i]->level == cc.L + 1 && part[i]->SPECIALlimb.size() > 0 && PRIMEID(part[i]->SPECIALlimb.at(cc.limbGPUid[*part[i]->level].y)) == *part[i]->level)) {
				LimbImpl& top = (*part[i]->level == cc.L + 1) ? part[i]->SPECIALlimb.at(0) : part[i]->limb.at(limbsize - 1);
				if (1) {
					stream[i]->wait(part[i]->s);
					constexpr ALGO algo = ALGO_SHOUP;
					constexpr int M		= 4;

					dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
					dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
					int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
					int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

					int start = 0;
					for (int i_ = limbsize - 1; i_ < limbsize; i_ += cc.batch) {
						// stream[i]->wait(i ? partition.s : s);
						uint32_t num_limbs = 1;

						INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, stream[i]->ptr()>>>(getGlobals(),
						  *part[i]->level == cc.L + 1 ? part[i]->SPECIALlimbptr.data + 0 : part[i]->limbptr.data + start + i_,
						  *part[i]->level == cc.L + 1 ? SPECIAL(id, 0) : PARTITION(id, start + i_),
						  ptraux[i]->data);
						INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, stream[i]->ptr()>>>(
						  getGlobals(), ptraux[i]->data, *part[i]->level == cc.L + 1 ? SPECIAL(id, 0) : PARTITION(id, start + i_), ptr[i]->data);
					}

					part[i]->s.wait(stream[i]->ptr());
					// s.wait(cc.top_limb_stream.at(id));
				} else {
					STREAM(top).wait(part[i]->s);
					SWITCH(top, INTT<ALGO_SHOUP>());
					stream[i]->wait(STREAM(top));
					cudaMemcpyAsync(buffer[i], std::get<U64>(top).v.data, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream[i]->ptr());
				}
				/*
			std::cout << "GPU: " << id << " ";
			SWITCH(top, printThisLimb(2));
			std::cout << std::endl;
*/
				// while (part[i]->bufferLIMB == nullptr && part[i]->limb.size() > limbsize - 1) {
				//     STREAM(part[i]->limb.back()).wait(*stream[i]);
				//     part[i]->limb.pop_back();
				// }
			} else {
				std::cout << "Cant find the top limb!!!" << part[i]->id << " " << *part[i]->level << " " << cc.limbGPUid[*part[i]->level].y << " "
						  << part[i]->limb.size() << std::endl;
			}
		} else {
			stream[i]->wait(part[i]->s);
			// stream[i]->record();
		}
	}

	if (cc.GPUid.size() > 1) {
		if (MEMCPY_PEER) {
			// openmp_synchronize();

			int top_gpu = cc.limbGPUid[*level].x;
			if (top_gpu == id) {
				stream[0]->record();
				stream[1]->record();
			}
			openmp_synchronize();

			if (top_gpu != id) {
				for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {

					if (i == static_cast<uint32_t>(id)) {
						{
							// auto& stream_0 = cc.digitStreamForMemcpyPeer[0].at(id).at(i);
							// auto& stream_1 = cc.digitStreamForMemcpyPeer[cc.dnum > 1 ? 1 : 0].at(id).at(i);

							if (limbsize > 0) {
								stream[0]->wait(cc.top_limb_stream[top_gpu]);
								stream[1]->wait(cc.top_limb_stream2[top_gpu]);

								// stream_0.wait(cc.top_limb_stream[i]);
								// stream_1.wait(cc.top_limb_stream2[i]);
								cudaMemcpyPeerAsync(buffer[0], cc.GPUid[i], cc.top_limb_buffer[top_gpu], cc.GPUid[top_gpu], cc.N * sizeof(uint64_t), stream[0]->ptr());
								cudaMemcpyPeerAsync(buffer[1], cc.GPUid[i], cc.top_limb_buffer2[top_gpu], cc.GPUid[top_gpu], cc.N * sizeof(uint64_t), stream[1]->ptr());

								CudaCheckErrorModNoSync;
								stream[0]->record();
								stream[1]->record();
								CudaCheckErrorModNoSync;
								cudaSetDevice(cc.GPUid[top_gpu]);
								CudaCheckErrorModNoSync;
								cc.top_limb_stream[top_gpu].wait(*stream[0]);
								cc.top_limb_stream2[top_gpu].wait(*stream[1]);
								CudaCheckErrorModNoSync;
							}
						}
					}
					CudaCheckErrorModNoSync;
				}

				cudaSetDevice(device);
			}
		} else {
#ifdef NCCL
			{
				NCCLCHECK(ncclGroupStart());
				if (static_cast<uint32_t>(id) == cc.limbGPUid[*level].x) {
					for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
						if (i != static_cast<uint32_t>(id)) {
							ncclSend(buffer[0], cc.N, ncclUint64, i, rank, stream[0]->ptr());
							ncclSend(buffer[1], cc.N, ncclUint64, i, rank, stream[1]->ptr());
						}
					}
				} else {
					ncclRecv(buffer[0], cc.N, ncclUint64, cc.limbGPUid[*part[0]->level].x, rank, stream[0]->ptr());
					ncclRecv(buffer[1], cc.N, ncclUint64, cc.limbGPUid[*part[1]->level].x, rank, stream[1]->ptr());
				}
				NCCLCHECK(ncclGroupEnd());
			}
#else
			assert(false);
#endif
		}
		/* {
			std::vector<uint64_t> data(cc.N, 0);
			cudaMemcpyAsync(data.data(), buffer, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.ptr());
			cudaDeviceSynchronize();
			std::cout << "GPU " << id << " (" << data[0] << " " << data[1] << ")" << std::endl;
		} */
	}
	for (int j = 0; j < 2; ++j) {
		// stream[j]->wait(part[j]->s);
		{
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			const dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			const dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			const int bytesFirst	  = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
			const int bytesSecond	  = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
			const int size			  = getLimbSize(*part[j]->level - 1 - (*part[j]->level == cc.L + 1 && cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT));
			const int batch			  = cc.batch;
			const NTT_MODE mode		  = NTT_RESCALE;

			int top_gpu = cc.limbGPUid[*level].x;

			for (int i = 0; i < size; i += batch) {
				uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

				NTT_<false, algo, mode>
				  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, id == top_gpu ? part[j]->s.ptr() : stream[j]->ptr()>>>(
					getGlobals(), ptr[j]->data, PARTITION(part[j]->id, i), part[j]->auxptr.data + i, nullptr, *part[j]->level);

				CudaCheckErrorModNoSync;
				if (id == top_gpu) {
					stream[j]->wait(part[j]->s);
				} else {
					part[j]->s.wait(*stream[j]);
				}
				CudaCheckErrorModNoSync;

				NTT_<true, algo, mode><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, part[j]->s.ptr()>>>(
				  getGlobals(), part[j]->auxptr.data + i, PARTITION(part[j]->id, i), part[j]->limbptr.data + i, nullptr, *part[j]->level);
			}
		}
	}

	/*
	if (cc.GPUid.size() > 1) {
		if (MEMCPY_PEER) {
			openmp_synchronize();
			if (cc.limbGPUid[*level].x == id) {
				for (int i = 0; i < cc.GPUid.size(); ++i) {
					if (i != id) {
						{
							stream[0]->wait(cc.top_limb_stream[i]);
							stream[1]->wait(cc.top_limb_stream2[i]);
						}
					}
					CudaCheckErrorModNoSync;
				}
			}
		}
	}
	*/
} // namespace FIDESlib::CKKS

void LimbPartition::dotKSKfusedMGPU(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a, const LimbPartition& ksk_b, const LimbPartition& src) {
	cudaSetDevice(device);

	struct vector_gpu {
		void*** data{ nullptr };
		int size;
	};

	vector_gpu digits{ .size = cc.dnum * 6 };
	cudaMallocAsync(&digits.data, digits.size * sizeof(void**), s.ptr());
	// VectorGPU<void**> digits(s, cc.dnum * 6, device);
	std::vector<void**> h_digits(cc.dnum * 6, nullptr);
	LimbPartition& out1 = *this;
	s.wait(ksk_a.getS());
	s.wait(ksk_b.getS());
	s.wait(src.getS());
	s.wait(out2.s);
	const int limbsize = *level + 1;

	if constexpr (1) {
		size_t i = 0;
		for (; i < src.DIGITlimb.size(); ++i) {
			{
				int start = 0;
				for (uint32_t j = 0; j < i; ++j)
					start += DECOMPmeta[j].size();
				int size = std::min((int)DECOMPmeta[i].size(), (int)limbsize - start);
				if (size <= 0)
					break;
			}

			h_digits[i]				  = digitSrc.DIGITlimbptr.at(i).data;
			h_digits[i + cc.dnum]	  = ksk_a.DIGITlimbptr.at(i).data;
			h_digits[i + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(i).data;
			h_digits[i + 3 * cc.dnum] = src.limbptr.data;
			h_digits[i + 4 * cc.dnum] = ksk_a.limbptr.data;
			h_digits[i + 5 * cc.dnum] = ksk_b.limbptr.data;
		}

		int num_special = 0;
		while (num_special < (int)DIGITmeta.at(0).size() && DIGITmeta.at(0).at(num_special).id > cc.L)
			num_special++;
		int num_limbs = 0;
		while (num_limbs < (int)meta.size() && meta.at(num_limbs).id <= *level)
			num_limbs++;

		if (num_special + num_limbs > 0) {
			cudaMemcpyAsync(digits.data, h_digits.data(), cc.dnum * 6 * sizeof(void**), cudaMemcpyDefault, s.ptr());
			fusedDotKSK_2_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)num_special + num_limbs }, 128, 0, s.ptr()>>>(
			  out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data, digits.data, i, id, num_special, 0);
		}
	}
	cudaFreeAsync(digits.data, s.ptr());
	// digits.free(s);

	src.getS().wait(s);
	out2.s.wait(s);
	ksk_a.getS().wait(s);
	ksk_b.getS().wait(s);
}

void LimbPartition::fusedHoistRotate(int n,
  std::vector<int> indexes,
  std::vector<LimbPartition*>& c0,
  std::vector<LimbPartition*>& c1,
  const std::vector<LimbPartition*>& ksk_a,
  const std::vector<LimbPartition*>& ksk_b,
  const LimbPartition& src_c0,
  const LimbPartition& src_c1,
  bool c0_modup) {
	cudaSetDevice(device);

	struct vector_gpu {
		void*** data{ nullptr };
		int size;
	};

	vector_gpu digits{ .size = n * cc.dnum * 6 + cc.dnum * 6 + 4 * n + n };
	cudaMallocAsync(&digits.data, digits.size * sizeof(void**), s.ptr());

	// VectorGPU<void**> digits(s, n * cc.dnum * 6 + cc.dnum * 6 + 4 * n + n, device);
	std::vector<void**> h_digits(n * cc.dnum * 6 + cc.dnum * 6 + 4 * n + n, nullptr);

	int offset_c1		  = n * cc.dnum * 6;
	int offset_output_c0  = n * cc.dnum * 6 + cc.dnum * 6;
	int offset_output_c1  = n * cc.dnum * 6 + cc.dnum * 6 + n;
	int offset_output_c0s = n * cc.dnum * 6 + cc.dnum * 6 + n * 2;
	int offset_output_c1s = n * cc.dnum * 6 + cc.dnum * 6 + n * 3;
	int offset_indexes	  = n * cc.dnum * 6 + cc.dnum * 6 + n * 4;

	LimbPartition& src = *this;
	s.wait(src_c0.getS());
	s.wait(src_c1.getS());
	for (int i = 0; i < n; ++i) {
		// s.wait(ksk_a[i]->s);
		// s.wait(ksk_b[i]->s);
		s.wait(c0[i]->s);
		s.wait(c1[i]->s);
	}
	const int limbsize = *level + 1;

	if constexpr (1) {

		for (int k = 0; k < n; ++k) {

			for (size_t i = 0; i < src.DIGITlimb.size(); ++i) {
				{
					int start = 0;
					for (uint32_t j = 0; j < i; ++j)
						start += DECOMPmeta[j].size();
					int size = std::min((int)DECOMPmeta[i].size(), (int)limbsize - start);
					if (size <= 0)
						break;
				}

				// h_digits[offset_c1 + i] = src.DIGITlimbptr.at(i).data;
				h_digits[k * cc.dnum * 6 + i + cc.dnum]		= ksk_a[k]->DIGITlimbptr.at(i).data;
				h_digits[k * cc.dnum * 6 + i + 2 * cc.dnum] = ksk_b[k]->DIGITlimbptr.at(i).data;
				// h_digits[offset_c1 + i + 3 * cc.dnum] = src_c1.limbptr.data;
				h_digits[k * cc.dnum * 6 + i + 4 * cc.dnum] = ksk_a[k]->limbptr.data;
				h_digits[k * cc.dnum * 6 + i + 5 * cc.dnum] = ksk_b[k]->limbptr.data;
			}
		}

		size_t i = 0;
		{

			for (; i < src.DIGITlimb.size(); ++i) {
				{
					int start = 0;
					for (uint32_t j = 0; j < i; ++j)
						start += DECOMPmeta[j].size();
					int size = std::min((int)DECOMPmeta[i].size(), (int)limbsize - start);
					if (size <= 0)
						break;
				}

				h_digits[offset_c1 + i] = src.DIGITlimbptr.at(i).data;
				// h_digits[i + cc.dnum] = ksk_a.DIGITlimbptr.at(i).data;
				// h_digits[i + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(i).data;
				h_digits[offset_c1 + i + 3 * cc.dnum] = src_c1.limbptr.data;
				// h_digits[i + 4 * cc.dnum] = ksk_a.limbptr.data;
				// h_digits[i + 5 * cc.dnum] = ksk_b.limbptr.data;
			}
		}

		for (int k = 0; k < n; ++k) {
			h_digits[offset_output_c0 + k]		 = c0[k]->limbptr.data;
			h_digits[offset_output_c1 + k]		 = c1[k]->limbptr.data;
			h_digits[offset_output_c0s + k]		 = c0[k]->SPECIALlimbptr.data;
			h_digits[offset_output_c1s + k]		 = c1[k]->SPECIALlimbptr.data;
			((int*)&h_digits[offset_indexes])[k] = indexes[k];
		}

		int num_special = 0;
		while (num_special < (int)DIGITmeta.at(0).size() && DIGITmeta.at(0).at(num_special).id > cc.L)
			num_special++;
		int num_limbs = 0;
		while (num_limbs < (int)meta.size() && meta.at(num_limbs).id <= *level)
			num_limbs++;

		cudaMemcpyAsync(digits.data, h_digits.data(), h_digits.size() * sizeof(void**), cudaMemcpyDefault, s.ptr());

		hoistedRotateDotKSK_2_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)num_special + num_limbs }, 128, sizeof(uint64_t) * 128 * i, s.ptr()>>>(digits.data + offset_c1,
		  src_c0.limbptr.data,
		  digits.data + offset_output_c1,
		  digits.data + offset_output_c1s,
		  digits.data + offset_output_c0,
		  digits.data + offset_output_c0s,
		  n,
		  (int*)(digits.data + offset_indexes),
		  digits.data,
		  i,
		  id,
		  num_special,
		  0,
		  src_c0.SPECIALlimbptr.data,
		  c0_modup);
	}

	src_c1.getS().wait(s);
	src_c0.getS().wait(s);
	for (int i = 0; i < n; ++i) {
		//  ksk_a[i]->s.wait(s);
		//  ksk_b[i]->s.wait(s);
		c0[i]->s.wait(s);
		c1[i]->s.wait(s);
	}
	cudaFreeAsync(digits.data, s.ptr());
	// digits.free(s);
}

void LimbPartition::modup_ksk_moddown_mgpu(LimbPartition& c0,
  const LimbPartition& ksk_a,
  const LimbPartition& ksk_b,
  LimbPartition& auxLimbs1,
  LimbPartition& auxLimbs2,
  const bool moddown,
  const std::vector<uint64_t*>& bufferGather_,
  const std::vector<uint64_t*>& bufferSpecial_c0,
  const std::vector<uint64_t*>& bufferSpecial_c1,
  const std::vector<Stream*>& external_s,
  std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>>& signal,
  std::vector<std::atomic_uint64_t*>& thread_stop,
  const std::vector<Stream*>& external_s0) {
	struct cached_graph {
		cudaGraph_t first;
		cudaGraphExec_t second;
		void*** digits; //(s, cc.dnum * 5, device);
		uint64_t buffKeyA, buffKeyB, buffAux1, buffAux2, buffC0, buffC1;
	};

	static std::map<Parameters, std::map<std::tuple<int, bool>, cached_graph>> map_c_to_map_graph_exec[8];
	static std::atomic_uint64_t skip;

	cudaSetDevice(device);

	constexpr bool PRINT = false;
	bool SELECT			 = id == 1;
	LimbPartition& c1	 = *this;
	int num_d			 = 0;
	{
		int start = 0;
		if constexpr (PRINT)
			std::cout << "/** Compute how many digits are used at this level*/" << std::endl;
		while (num_d < cc.dnum && start < *level + 1) {
			start += DECOMPmeta.at(num_d).size();
			num_d++;
		}
	}
	uint32_t limb_size = 0;
	while (limb_size < meta.size() && meta[limb_size].id <= *level)
		limb_size++;

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Input: ";
			for (size_t i = 0; i < limb_size; ++i) {
				std::cout << meta[i].id;
				SWITCH(limb[i], printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	if constexpr (PRINT)
		std::cout << "/** We try to pipeline the computation of each digit first, splitting independent groups of limbs*/" << std::endl;

	const int digits_per_it = MEMCPY_PEER ? 1 : 1 /*num_d*/; // cc.logN <= 15 ? num_d : cc.logN == 16 ? std::max((num_d + 1) / 2, 1) : 1;

	if constexpr (PRINT)
		std::cout << "GPU " << id << "compute " << num_d << " digits" << std::endl;

	std::vector<void**> h_digits(cc.dnum * 6, nullptr);
	auto& map_exec = map_c_to_map_graph_exec[id][this->cc.param];
	auto exec_old  = map_exec.find(std::tuple<int, bool>{ *level, moddown });

	size_t digits_size = 6 * cc.dnum;
	void*** digits; //(s, cc.dnum * 5, device);
	if (exec_old != map_exec.end()) {
		digits = exec_old->second.digits;
	} else {
		cudaMallocAsync(&digits, 6 * cc.dnum * sizeof(void**), s.ptr());
		// digits = std::make_shared<VectorGPU<void**>>(s, 5 * cc.dnum, device);
	}
	CudaCheckErrorModNoSync;
	for (int32_t j = 0; j < num_d; ++j) {
		h_digits[j]				  = DIGITlimbptr.at(j).data;
		h_digits[j + cc.dnum]	  = ksk_a.DIGITlimbptr.at(j).data;
		h_digits[j + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(j).data;
		h_digits[j + 3 * cc.dnum] = limbptr.data;
		h_digits[j + 4 * cc.dnum] = ksk_a.limbptr.data;
		h_digits[j + 5 * cc.dnum] = ksk_b.limbptr.data;
	}
	cudaMemcpyAsync(digits, h_digits.data(), digits_size * sizeof(void**), cudaMemcpyDefault, s.ptr());

	s.wait(auxLimbs1.s);
	s.wait(auxLimbs2.s);
	s.wait(c0.s);
	s.wait(ksk_a.getS());
	s.wait(ksk_b.getS());
	// cudaMemcpyAsync(digits, h_digits.data(), digits_size * sizeof(void**), cudaMemcpyDefault, s.ptr());
	//  cc.digitStream2.at(0).at(id).wait(s);  // This is for the regular limbs keySWITCH:

	cudaEvent_t ev;

	cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

	constexpr bool SERIAL_MGPU_PRINT = false;
	std::set<Stream*> join_at_end;

	// cudaDeviceSynchronize();
	CudaCheckErrorModNoSync;
	/*
	std::cout << ksk_a.bufferLIMB << " " << ksk_a.bufferGATHER << " " << ksk_a.bufferDECOMPandDIGIT << " "
			  << ksk_a.bufferSPECIAL << std::endl;
	std::cout << ksk_b.bufferLIMB << " " << ksk_b.bufferGATHER << " " << ksk_b.bufferDECOMPandDIGIT << " "
			  << ksk_b.bufferSPECIAL << std::endl;
	std::cout << auxLimbs1.bufferLIMB << " " << auxLimbs1.bufferGATHER << " " << auxLimbs1.bufferDECOMPandDIGIT << " "
			  << auxLimbs1.bufferSPECIAL << std::endl;
	std::cout << auxLimbs2.bufferLIMB << " " << auxLimbs2.bufferGATHER << " " << auxLimbs2.bufferDECOMPandDIGIT << " "
			  << auxLimbs2.bufferSPECIAL << std::endl;
	std::cout << c0.bufferLIMB << " " << c0.bufferGATHER << " " << c0.bufferDECOMPandDIGIT << " " << c0.bufferSPECIAL
			  << std::endl;
	std::cout << c1.bufferLIMB << " " << c1.bufferGATHER << " " << c1.bufferDECOMPandDIGIT << " " << c1.bufferSPECIAL
			  << std::endl;
*/
	if (GRAPH_CAPTURE) {
		if ((!MEMCPY_PEER || (MEMCPY_PEER && id == 0))) {

			if (exec_old != map_exec.end()) {
				auto& graph_data = exec_old->second;
				if (graph_data.buffKeyA == ksk_a.uid && graph_data.buffKeyB == ksk_b.uid && graph_data.buffAux1 == auxLimbs1.uid &&
				  graph_data.buffAux2 == auxLimbs2.uid && graph_data.buffC0 == c0.uid && graph_data.buffC1 == c1.uid) {
					//   std::cout << "Graph for keyswitch is the same, skip capture, level=" << *level
					//             << " moddown=" << moddown << std::endl;
					skip = 1;
				} else {
					//   std::cout << "Graph for keyswitch needs changes, run capture level=" << *level
					//             << " moddown=" << moddown << std::endl;
					skip = 0;
				}
			} else {
				skip = 0;
			}
		}
	} else {
		skip = 0;
	}

	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		if (SERIAL_MGPU_PRINT)
			std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		*thread_stop[id] += 1;
		if (id == 0) {
			for (uint32_t peer = 1; peer < cc.GPUid.size(); ++peer) {
				while (*thread_stop[peer] < *thread_stop[0])
					;

				if (SERIAL_MGPU_PRINT)
					std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << " " << *thread_stop[peer] << std::endl;
				s.wait(*external_s[peer]);
			}
			if (skip == 1) {
				*thread_stop[id] += 1;
				goto skip_capture;
			}
			cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeGlobal /*cudaStreamCaptureModeRelaxed*/);
		} else {
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;

			if (skip == 1) {
				*thread_stop[id] += 1;
				goto skip_capture;
			}
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		}

		if (id != 0) {
			s.wait(*external_s[0]);
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
			*thread_stop[id] += 1;
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		} else {
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
			*thread_stop[id] += 1;
			while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
				;
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		}
	} else if (GRAPH_CAPTURE) {
		if (skip == 1) {
			goto skip_capture;
		}
		openmp_synchronize();
		if (GRAPH_CAPTURE) {
			CudaCheckErrorModNoSync;
			// cudaDeviceSynchronize();
			cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeThreadLocal /*cudaStreamCaptureModeRelaxed*/);
		}
	}

	CudaCheckErrorModNoSync;
	if (MEMCPY_PEER && GRAPH_CAPTURE) {
	} else {
		openmp_synchronize();
	}
	CudaCheckErrorModNoSync;
	if (cc.GPUid.size() > 1) {
		/* This is needed for moddown communication dependencies, if digits_per_it != 0, to make sure, gather buffers exist before memcpypeer*/
		if (MEMCPY_PEER && GRAPH_CAPTURE) {
			for (int i_ = 0; i_ < 2; ++i_) {
				for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
					if (j != static_cast<uint32_t>(id)) {

						// notifyKernel(signal[(cc.dnum + 2) + cc.dnum + i_][j][id].second,
						//              signal[(cc.dnum + 2) + cc.dnum + i_][j][id].first, s.ptr());
					}
				}
			}
			// openmp_synchronize();
			for (int i_ = 0; i_ < 2; ++i_) {

				for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
					if (j != static_cast<uint32_t>(id)) {
						cc.digitStreamForMemcpyPeer[cc.dnum > 1 ? i_ : 0].at(id).at(j).wait(s);
						cc.digitStreamForMemcpyPeer[cc.dnum > 1 ? i_ : 0].at(id).at(j).wait(*external_s[j]);
						// pollingKernel(signal[(cc.dnum + 2) + cc.dnum + i_][id][j].second,
						//               signal[(cc.dnum + 2) + cc.dnum + i_][id][j].first,
						//               cc.digitStreamForMemcpyPeer[cc.dnum > 1 ? i_ : 0].at(id).at(j).ptr());
						// signal[cc.dnum > 1 ? i_ : 0][id][j].first++;
						join_at_end.insert(&cc.digitStreamForMemcpyPeer[cc.dnum > 1 ? i_ : 0].at(id).at(j));
					}
				}
			}
		}
	}

#if 1
	CudaCheckErrorModNoSync;
	for (int d = 0; d < num_d; d += digits_per_it) {
		Stream& stream = cc.digitStream.at(d).at(id);
		stream.wait(s);
	}
	for (int d = 0; d < num_d; d += digits_per_it) {
		int ds = std::min(num_d - d, digits_per_it);

		uint32_t start_d = 0;
		while (start_d < limb_size && meta[start_d].digit < d)
			start_d++;
		uint32_t size_d = 0;
		while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
			size_d++;

		if constexpr (PRINT)
			if (SELECT) {
				std::cout << "GPU " << id << " for digits " << d << ":" << d + digits_per_it << " INTT " << size_d << " limbs starting at limb " << start_d << std::endl;
			}

		Stream& stream = cc.digitStream.at(d).at(id);
		// stream.wait(s);
		if constexpr (PRINT)
			std::cout << "/** Intt */" << std::endl;
		if (size_d > 0) {
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			int gather_offset = 0;
			for (int i = 0; i < id; ++i) {
				gather_offset += cc.meta.at(i).size();
			}

			{
				// std::cout << "Parms INTT first: digit:" << d << " device:" << id << " s:" << stream.ptr() << " glob:" << getGlobals() << " auxptr:" << auxptr.data + start_d
				//		  << " primeid_table_offset:" << PARTITION(id, start_d) << " ptr:" << limbptr.data + start_d << std::endl;
				INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size_d }, blockDimFirst, bytesFirst, stream.ptr()>>>(
				  getGlobals(), limbptr.data + start_d, PARTITION(id, start_d), auxptr.data + start_d);

				CudaCheckErrorModNoSync;
				// std::cout << "Parms INTT second: digit:" << d << " device:" << id << " s:" << stream.ptr() << " glob:" << getGlobals() << " auxptr:" << auxptr.data + start_d
				//		  << " primeid_table_offset:" << PARTITION(id, start_d) << " GATHERptr:" << GATHERptr.data + gather_offset + start_d << std::endl;
				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size_d }, blockDimSecond, bytesSecond, stream.ptr()>>>(
				  getGlobals(), auxptr.data + start_d, PARTITION(id, start_d), GATHERptr.data + gather_offset + start_d);
			}
			CudaCheckErrorModNoSync;
		}

		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out INTT: ";
				for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
					for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
						std::cout << DECOMPmeta[j][i].id;
						SWITCH(DECOMPlimb[j][i], printThisLimb(2));
					}
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}

		if constexpr (PRINT)
			std::cout << "/** Communicate */" << std::endl;
		{

			if (!PEER_ACCESS && cc.GPUid.size() > 1) {

				if (MEMCPY_PEER) {

					{
						CudaCheckErrorModNoSync;
						stream.record();
						for (size_t i = 0; i < cc.GPUid.size(); ++i) {
							if (i != static_cast<size_t>(id)) {
								// notifyKernel(signal[(cc.dnum + 2) + d][i][id].second,
								//              signal[(cc.dnum + 2) + d][i][id].first + 1, stream.ptr());
							}
						}
						if (MEMCPY_PEER && GRAPH_CAPTURE) {
							if (SERIAL_MGPU_PRINT)
								std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
							*thread_stop[id] += 1;
							if (id != 0) {
								while (*thread_stop[id] >= *thread_stop[id - 1])
									;
								if (SERIAL_MGPU_PRINT)
									std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
							} else {
								while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
									;
								if (SERIAL_MGPU_PRINT)
									std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
							}
						} else {
							openmp_synchronize();
						}
						for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
							if (i != static_cast<uint32_t>(id)) {
								/**
								 * We do not synchronize here as all GPUS are synchronized on GPU 0's main stream at the beggining
								 */
								// cc.digitStreamForMemcpyPeer.at(d).at(id).at(i).wait()
								// pollingKernel(signal[(cc.dnum + 2) + d][id][i].second,
								//               signal[(cc.dnum + 2) + d][id][i].first + 1,
								//               cc.digitStreamForMemcpyPeer.at(d).at(id).at(i).ptr());
								// signal[d][i][id].first++;
								// ensure gather buffer is created before peer memcpy, we use s as the dependency will be older than on stream
								// join_at_end.insert(&cc.digitStreamForMemcpyPeer.at(d).at(id).at(i));
							}
						}
					}
					{

						int start = 0;
						for (size_t i = 0; i < cc.GPUid.size(); ++i) {

							uint32_t limb_size_i = 0;
							while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
								limb_size_i++;
							uint32_t start_d_i = 0;
							while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
								start_d_i++;
							uint32_t size_d_i = 0;
							while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
								size_d_i++;
							if constexpr (PRINT)
								if (SELECT) {
									std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it << " communicate " << size_d_i << " limbs" << std::endl;
								}
							if (size_d_i > 0) {

								if (i == static_cast<uint32_t>(id)) {
									for (size_t j = 0; j < cc.GPUid.size(); ++j) {

										if (j != i) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(i).at(j);
											stream_.wait(stream);
											if (0 && GRAPH_CAPTURE) {
												CudaCheckErrorModNoSync;
												transferKernel((float*)(bufferGATHER + cc.N * (start + start_d_i)),
												  (float*)(bufferGather_[j] + cc.N * (start + start_d_i)),
												  sizeof(uint64_t) * size_d_i * cc.N / sizeof(float),
												  stream_.ptr(),
												  device,
												  cc.GPUid[j]);
												CudaCheckErrorModNoSync;
												notifyKernel(signal[d][j][i].second, signal[d][j][i].first + 2, stream_.ptr());
												CudaCheckErrorModNoSync;
												join_at_end.insert(&stream_);
											} else {
												CudaCheckErrorModNoSync;
												transferKernel((float*)(bufferGATHER + cc.N * (start + start_d_i)),
												  (float*)(bufferGather_[j] + cc.N * (start + start_d_i)),
												  sizeof(uint64_t) * size_d_i * cc.N / sizeof(float),
												  stream_.ptr(),
												  device,
												  cc.GPUid[j]);
												// cudaMemcpyPeerAsync(
												//     bufferGather_[j] + cc.N * (start + start_d_i), cc.GPUid[j],
												//     bufferGATHER + cc.N * (start + start_d_i), cc.GPUid[i],
												//     sizeof(uint64_t) * size_d_i * cc.N, stream_.ptr());
												CudaCheckErrorModNoSync;
												stream_.record();
											}
										}
									}
								}
							}
							start += cc.meta[i].size();
						}
					}
					if (MEMCPY_PEER && GRAPH_CAPTURE) {
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						*thread_stop[id] += 1;
						if (id != 0) {
							while (*thread_stop[id] >= *thread_stop[id - 1])
								;
							if (SERIAL_MGPU_PRINT)
								std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						} else {
							while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
								;
							if (SERIAL_MGPU_PRINT)
								std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						}
					} else {
						openmp_synchronize();
					}

					{
						int start = 0;
						for (size_t i = 0; i < cc.GPUid.size(); ++i) {

							uint32_t limb_size_i = 0;
							while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
								limb_size_i++;
							uint32_t start_d_i = 0;
							while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
								start_d_i++;
							uint32_t size_d_i = 0;
							while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
								size_d_i++;

							if (size_d_i > 0) {
								for (size_t j = 0; j < cc.GPUid.size(); ++j) {
									if (j != i) {
										if (j == static_cast<size_t>(id)) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(i).at(j);
											stream.wait(stream_);
										}
									}
								}
							}
							start += cc.meta[i].size();
						}
					}
				} else {
#ifdef NCCL
					NCCLCHECK(ncclGroupStart());

					int start = 0;
					for (size_t i = 0; i < cc.GPUid.size(); ++i) {
						uint32_t limb_size_i = 0;
						while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
							limb_size_i++;
						uint32_t start_d_i = 0;
						while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
							start_d_i++;
						uint32_t size_d_i = 0;
						while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
							size_d_i++;

						if constexpr (PRINT)
							if (SELECT) {
								std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it << " communicate " << size_d_i << " limbs" << std::endl;
							}

						if (size_d_i > 0) {
							NCCLCHECK(ncclBroadcast(
							  /*bufferLIMB + cc.N * start_d*/ bufferGATHER + cc.N * (start + start_d_i),
							  bufferGATHER + cc.N * (start + start_d_i),
							  size_d_i * cc.N,
							  ncclUint64,
							  i,
							  rank,
							  stream.ptr()));
						}
						start += cc.meta[i].size();
					}
					NCCLCHECK(ncclGroupEnd());

#else
					assert(false);
#endif
				}
			} else if (cc.GPUid.size() > 1) {
				stream.record();

				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					if (SERIAL_MGPU_PRINT)
						std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					}
				} else {
					openmp_synchronize();
				}
				for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
					if (static_cast<uint32_t>(id) != i)
						stream.wait(cc.digitStream[d][i]);
				}
				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					if (SERIAL_MGPU_PRINT)
						std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					}
				} else {
					openmp_synchronize();
				}
			}
		}
		//}

		CudaCheckErrorModNoSync;
		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out INTT after communicate: ";
				for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
					for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
						std::cout << DECOMPmeta[j][i].id;
						SWITCH(DECOMPlimb[j][i], printThisLimb(2));
					}
					std::cout << std::endl;
				}
				cudaDeviceSynchronize();
			}
			CudaCheckErrorModNoSync;
		}

		CudaCheckErrorModNoSync;
		// for (int d = 0; d < num_d; d += digits_per_it) {
		// int ds = std::min(num_d - d, digits_per_it);

		// uint32_t start_d = 0;
		// while (start_d < limb_size && meta[start_d].digit < d)
		//	start_d++;
		// uint32_t size_d = 0;
		// while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
		//	size_d++;

		// Stream& stream = cc.digitStream.at(d).at(id);

		if constexpr (PRINT)
			std::cout << "/** Conv */" << std::endl;

		for (int d_ = d; d_ < d + ds; ++d_) {
			Stream& stream1 = cc.digitStream.at(d_).at(id);
			CudaCheckErrorModNoSync;
			stream1.wait(stream);
			CudaCheckErrorModNoSync;

			int start = 0;
			for (int j = 0; j < d_; ++j)
				start += DECOMPlimb.at(j).size();
			int size = std::min((int)DECOMPlimb.at(d_).size(), *level + 1 - start);

			if (size <= 0) {
				std::cerr << "void modup, aborting" << std::endl;
				exit(-1);
			}

			if constexpr (PRINT)
				if (SELECT) {
					std::cout << cc.precom.constants[id].num_primeid_digit_to[d_][*level] << "<- num_prime_id_digit_to: " << d_ << std::endl;
					std::cout << cc.precom.constants[id].num_primeid_digit_from[d_][*level] << "<- num_prime_id_digit_from: " << d_ << std::endl;
				}

			CudaCheckErrorModNoSync;
			if (!PEER_ACCESS) {
				dim3 blockSize{ 64, 2 };
				dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
				int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;

				DecompAndModUpConv<ALGO_SHOUP>
				  <<<gridSize, blockSize, shared_bytes, stream1.ptr()>>>(DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_], getGlobals());
			} else {
				dim3 blockSize{ 64, 2 };
				dim3 gridSize{ (uint32_t)cc.N / blockSize.x / 2 };
				int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x * 2;
				if (d_ > 0)
					cudaStreamWaitEvent(stream1.ptr(), ev);
				DecompAndModUpConv_spec2<ALGO_SHOUP>
				  <<<gridSize, blockSize, shared_bytes, stream1.ptr()>>>(DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_], getGlobals());
				cudaEventRecord(ev, stream1.ptr());
			}
			CudaCheckErrorModNoSync;
			cc.digitStream2.at(d_).at(id).wait(stream1); /** Get dependency for limb NTTs later */
			if constexpr (PRINT)
				std::cout << "/** NTT special limbs */" << std::endl;
			{
				uint32_t size		= cc.splitSpecialMeta.at(id).size();
				constexpr ALGO algo = ALGO_SHOUP;
				constexpr int M		= 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				if (size > 0) {
					NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size }, blockDimFirst, bytesFirst, stream1.ptr()>>>(
					  getGlobals(), DIGITlimbptr[d_].data, DIGIT(d_, 0), c0.DIGITlimbptr[d_].data);

					NTT_<true, algo, NTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size }, blockDimSecond, bytesSecond, stream1.ptr()>>>(
					  getGlobals(), c0.DIGITlimbptr[d_].data, DIGIT(d_, 0), DIGITlimbptr[d_].data);
				}
			}
		}
	}
	for (int d = 0; d < num_d; ++d) {
		s.wait(cc.digitStream.at(d).at(id));
	}
	CudaCheckErrorModNoSync;

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Out ModUp after NTT specials: ";
			for (size_t j = 0; j < DIGITlimb.size(); ++j) {
				for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
					std::cout << DIGITmeta[j][i].id;
					SWITCH(DIGITlimb[j][i], printThisLimb(2));
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
		CudaCheckErrorModNoSync;
	}
	CudaCheckErrorModNoSync;
	if constexpr (PRINT)
		std::cout << "/** We ksk only special limbs and start ModDown as soon as possible */" << std::endl;

	CudaCheckErrorModNoSync;
	if constexpr (PRINT)
		std::cout << "/** ksk */" << std::endl;
	{

		LimbPartition& out1 = *this;
		LimbPartition& out2 = c0;

		if constexpr (1) {
			int num_special = cc.splitSpecialMeta.at(id).size();

			if (num_special > 0) {

				fusedDotKSK_2_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)num_special }, 128, 0, s.ptr()>>>(
				  out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data, digits, num_d, id, num_special, 0);
			}
		}

		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out KSK specials: ";
				for (const auto& j : { &out1, &out2 }) {
					for (auto& i : j->SPECIALlimb) {
						SWITCH(i, printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
			CudaCheckErrorModNoSync;
		}
	}

	c0.s.wait(s);
	CudaCheckErrorModNoSync;

	for (int i_ = 0; i_ < 2; ++i_) {
		if (moddown) {
			Stream& stream			 = i_ == 0 ? s : c0.s;
			LimbPartition& out		 = i_ == 0 ? c1 : c0;
			LimbPartition& aux_limbs = i_ == 0 ? auxLimbs1 : auxLimbs2;
			if constexpr (PRINT)
				std::cout << "/** INTT specials*/" << std::endl;
			{
				constexpr ALGO algo = ALGO_SHOUP;
				constexpr int M		= 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				const uint32_t limbs = cc.splitSpecialMeta.at(id).size();

				if (limbs > 0) {
					const int j = cc.splitSpecialMeta.at(id).at(0).id - cc.specialMeta.at(id).at(0).id;

					INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), limbs }, blockDimFirst, bytesFirst, stream.ptr()>>>(
					  getGlobals(), out.SPECIALlimbptr.data + j, SPECIAL(id, j), out.SPECIALauxptr.data + j);

					INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), limbs }, blockDimSecond, bytesSecond, stream.ptr()>>>(
					  getGlobals(), out.SPECIALauxptr.data + j, SPECIAL(id, j), aux_limbs.SPECIALlimbptr.data + j);
				}
			}

			//}

			CudaCheckErrorModNoSync;
			if constexpr (PRINT)
				std::cout << "/** communicate */" << std::endl;
			if (!PEER_ACCESS && cc.GPUid.size() > 1) {
				if (MEMCPY_PEER) {

					// for (int i_ = 0; i_ < 2; ++i_)
					{
						Stream& stream = i_ == 0 ? s : c0.s;
						// LimbPartition& out					  = i_ == 0 ? c1 : c0;
						std::vector<uint64_t*> bufferSpecial_ = i_ == 0 ? bufferSpecial_c1 : bufferSpecial_c0;

						for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
							// Limb<uint64_t>& l = std::get<U64>(
							//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
							const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
							// uint64_t* ptr2 = l.v.data;

							if (static_cast<uint32_t>(id) == i) {
								if (num_limbs > 0) {
									uint64_t* ptr = bufferSpecial_[i] + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N;

									for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
										if (j != i) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(cc.dnum > 1 ? i_ : 0).at(i).at(j);
											stream_.wait(stream);

											if (0 && GRAPH_CAPTURE) {
												CudaCheckErrorModNoSync;
												transferKernel((float*)(ptr),
												  (float*)(bufferSpecial_[j] + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N),
												  sizeof(uint64_t) * cc.N * num_limbs / sizeof(float),
												  stream_.ptr(),
												  device,
												  cc.GPUid[j]);
												CudaCheckErrorModNoSync;
												notifyKernel(signal[cc.dnum + i_][j][i].second, signal[cc.dnum + i_][j][i].first + 3, stream_.ptr());
												CudaCheckErrorModNoSync;
												join_at_end.insert(&stream_);
											} else {
												CudaCheckErrorModNoSync;
												transferKernel((float*)(ptr),
												  (float*)(bufferSpecial_[j] + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N),
												  sizeof(uint64_t) * cc.N * num_limbs / sizeof(float),
												  stream_.ptr(),
												  device,
												  cc.GPUid[j]);
												//   cudaMemcpyPeerAsync(
												//       bufferSpecial_[j] +
												//           (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N,
												//       cc.GPUid[j], ptr, cc.GPUid[i], sizeof(uint64_t) * cc.N * num_limbs,
												//       stream_.ptr());
												CudaCheckErrorModNoSync;
												stream_.record();
											}
										}
									}
								}
							}
						}
					}
					if (MEMCPY_PEER && GRAPH_CAPTURE) {
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						*thread_stop[id] += 1;
						if (id != 0) {
							while (*thread_stop[id] >= *thread_stop[id - 1])
								;
							if (SERIAL_MGPU_PRINT)
								std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						} else {
							while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
								;
							if (SERIAL_MGPU_PRINT)
								std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
						}
					} else {
						openmp_synchronize();
					}
					// for (int i_ = 0; i_ < 2; ++i_)
					{
						Stream& stream = i_ == 0 ? s : c0.s;
						// LimbPartition& out = i_ == 0 ? c1 : c0;
						for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
							// Limb<uint64_t>& l = std::get<U64>(
							//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
							const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
							// uint64_t* ptr2 = l.v.data;

							if (num_limbs > 0) {

								for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
									if (j == static_cast<uint32_t>(id)) {
										if (i != j) {

											if (0 && GRAPH_CAPTURE) {
												CudaCheckErrorModNoSync;
												pollingKernel(signal[cc.dnum + i_][j][i].second, signal[cc.dnum + i_][j][i].first + 3, stream.ptr());
												CudaCheckErrorModNoSync;
												// signal[cc.dnum > 1 ? i_ : 0][j][i].first++;
											} else {
												Stream& stream_ = cc.digitStreamForMemcpyPeer.at(cc.dnum > 1 ? i_ : 0).at(i).at(j);
												stream.wait(stream_);
											}
										}
									}
								}
							}
						}
					}
				} else {
					// for (int i_ = 0; i_ < 2; ++i_)
					{
#ifdef NCCL
						NCCLCHECK(ncclGroupStart());
						Stream& stream = i_ == 0 ? s : c0.s;
						// LimbPartition& out		 = i_ == 0 ? c1 : c0;
						uint64_t* bufferSpecial_ = i_ == 0 ? auxLimbs1.bufferSPECIAL : auxLimbs2.bufferSPECIAL;

						for (size_t j = 0; j < cc.splitSpecialMeta.size(); ++j) {
							// Limb<uint64_t>& l = std::get<U64>(
							//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
							const uint32_t num_limbs = cc.splitSpecialMeta.at(j).size();
							// uint64_t* ptr2 = l.v.data;

							if (num_limbs > 0) {
								uint64_t* ptr = bufferSpecial_ + (cc.splitSpecialMeta.at(j).at(0).id - SPECIALmeta.at(0).id) * cc.N;
								NCCLCHECK(ncclBroadcast(ptr, ptr, cc.N * num_limbs, ncclUint64, (int)j, rank, stream.ptr()));
							}
						}
						NCCLCHECK(ncclGroupEnd());

#else
						assert(false);
#endif
					}
				}
			} else if (cc.GPUid.size() > 1) {
				stream.record();

				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					if (SERIAL_MGPU_PRINT)
						std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					}
				} else {
					openmp_synchronize();
				}

				for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
					if (static_cast<uint32_t>(id) != i)
						stream.wait(i_ == 0 ? *external_s[i] : *external_s0[i]);
				}

				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					if (SERIAL_MGPU_PRINT)
						std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
						if (SERIAL_MGPU_PRINT)
							std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
					}
				} else {
					openmp_synchronize();
				}
			}
		}

		CudaCheckErrorModNoSync;

		if (moddown) {
			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << id << "KSK specials after INTT and communicate: ";
					for (const auto& j : { &c1, &c0 }) {
						for (auto& i : j->SPECIALlimb) {
							SWITCH(i, printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
				CudaCheckErrorModNoSync;
			}

			// for (int i_ = 0; i_ < 2; ++i_)
			{
				Stream& stream			= i_ == 0 ? c1.s : c0.s;
				LimbPartition& auxLimbs = i_ == 0 ? auxLimbs1 : auxLimbs2;
				if constexpr (PRINT)
					std::cout << "/** Conv */" << std::endl;

				if (!PEER_ACCESS) {
					dim3 blockSize{ 64, 2 };

					dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
					int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;
					if (limb_size > 0)
						ModDown2<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr()>>>(
						  auxLimbs.limbptr.data, limb_size, auxLimbs.SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
				} else {

					if (i_ == 1)
						stream.wait(c1.s);
					dim3 blockSize{ 64, 2 };

					dim3 gridSize{ (uint32_t)cc.N / blockSize.x / 2 };
					int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x * 2;
					if (limb_size > 0)
						ModDown3<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr()>>>(
						  auxLimbs.limbptr.data, limb_size, auxLimbs.SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
				}
			}

			CudaCheckErrorModNoSync;
			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << id << "Out Moddown: ";
					for (const auto& j : { &auxLimbs1, &auxLimbs2 }) {
						for (auto& i : j->limb) {
							SWITCH(i, printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}
		}
	}

	CudaCheckErrorModNoSync;
	if constexpr (PRINT)
		std::cout << "/**We delay the call of NTTs post-modup for non special limbs to here*/" << std::endl;
	for (int d = 0; d < num_d; ++d) {

		Stream& stream = cc.digitStream2.at(d).at(id);

		if (limb_size > 0) {
			uint32_t start = cc.splitSpecialMeta.at(id).size();
			uint32_t size  = cc.precom.constants[id].num_primeid_digit_to[d][*level] - start;
			if (size > 0) {
				constexpr ALGO algo = ALGO_SHOUP;
				constexpr int M		= 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				{
					NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size }, blockDimFirst, bytesFirst, stream.ptr()>>>(
					  getGlobals(), DIGITlimbptr[d].data + start, DIGIT(d, start), c0.DIGITlimbptr[d].data + start);

					NTT_<true, algo, NTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size }, blockDimSecond, bytesSecond, stream.ptr()>>>(
					  getGlobals(), c0.DIGITlimbptr[d].data + start, DIGIT(d, start), DIGITlimbptr[d].data + start);
				}
			}
		}
	}
	CudaCheckErrorModNoSync;
	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Out ModUp after NTT all limbs: ";
			for (size_t j = 0; j < DIGITlimb.size(); ++j) {
				for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
					std::cout << DIGITmeta[j][i].id;
					SWITCH(DIGITlimb[j][i], printThisLimb(2));
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
		CudaCheckErrorModNoSync;
	}

	if constexpr (PRINT)
		std::cout << "/** ksk remaining limbs*/" << std::endl;

	{
		Stream& stream = cc.digitStream2.at(0).at(id);
		for (int d = 1; d < num_d; ++d) {
			stream.wait(cc.digitStream2.at(d).at(id));
		}
		LimbPartition& out1 = *this;
		LimbPartition& out2 = c0;

		if constexpr (1) {
			size_t i = num_d;

			int num_special = cc.splitSpecialMeta.at(id).size();
			if (limb_size > 0) {
				for (uint32_t start = 0; start < limb_size; start += cc.batch) {
					uint32_t num = std::min(static_cast<uint32_t>(cc.batch), limb_size - start);
					fusedDotKSK_2_<<<dim3{ (uint32_t)cc.N / 128, num }, 128, 0, stream.ptr()>>>(
					  out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data, digits, i, id, num_special, num_special + start);
				}
			}
		}

		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out KSK limbs: ";
				for (const auto& j : { &out1, &out2 }) {
					for (auto& i : j->limb) {
						SWITCH(i, printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}
	}
	CudaCheckErrorModNoSync;
	if (moddown) {
		for (int i = 0; i < 2; ++i) {
			if constexpr (PRINT)
				std::cout << "/** Last NTT step for moddown*/" << std::endl;
			Stream& stream			= i == 0 ? c1.s : c0.s;
			LimbPartition& out		= i == 0 ? c1 : c0;
			LimbPartition& auxLimbs = i == 0 ? auxLimbs1 : auxLimbs2;

			if (limb_size > 0) {
				constexpr ALGO algo = ALGO_SHOUP;
				constexpr int M		= 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				{
					NTT_<false, algo, NTT_MODDOWN><<<dim3{ cc.N / (blockDimFirst.x * M * 2), limb_size }, blockDimFirst, bytesFirst, stream.ptr()>>>(
					  getGlobals(), auxLimbs.limbptr.data, PARTITION(id, 0), out.auxptr.data);

					stream.wait(cc.digitStream2.at(0).at(id));

					NTT_<true, algo, NTT_MODDOWN><<<dim3{ cc.N / (blockDimSecond.x * M * 2), limb_size }, blockDimSecond, bytesSecond, stream.ptr()>>>(
					  getGlobals(), out.auxptr.data, PARTITION(id, 0), out.limbptr.data);
				}
			}
		}
		CudaCheckErrorModNoSync;
		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out Moddown after submult: ";
				for (const auto& j : { &c1, &c0 }) {
					for (auto& i : j->limb) {
						SWITCH(i, printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
			CudaCheckErrorModNoSync;
		}
	} else {
		s.wait(cc.digitStream2.at(0).at(id));
	}
	if constexpr (PRINT) {
		std::cout << "Going out keyswitch" << std::endl;
	}
	CudaCheckErrorModNoSync;
	s.wait(c0.s);
	CudaCheckErrorModNoSync;
#endif
	// Ensure all communication has ended before the data pointer owners are destroyed or their content is overwritten
	if (cc.GPUid.size() > 1) {
		if (MEMCPY_PEER) {
			for (Stream* stream : join_at_end) {
				s.wait(*stream);
				CudaCheckErrorModNoSync;
			}
		}
	}
	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		if (SERIAL_MGPU_PRINT)
			std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		*thread_stop[id] += 1;
		if (id != 0) {
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
		} else {
			while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
				;
			if (SERIAL_MGPU_PRINT)
				std::cout << "Thread " << id << " checkpoint " << *thread_stop[id] << std::endl;
			for (uint32_t peer = 1; peer < cc.GPUid.size(); ++peer) {
				s.wait(*external_s[peer]);
			}
			CudaCheckErrorModNoSync;
			// dummy_kernel<<<1, 32, 0, s.ptr()>>>();
			CudaCheckErrorModNoSync;
		}
	} else {
		openmp_synchronize();
		CudaCheckErrorModNoSync;
		// dummy_kernel<<<1, 32, 0, s.ptr()>>>();
		CudaCheckErrorModNoSync;
	}

	if (0 && id == 0 && GRAPH_CAPTURE) {
		// === INSPECT BEFORE ENDING ===
		cudaStreamCaptureStatus status;
		cudaGraph_t capturing_graph;

		cudaStreamGetCaptureInfo(s.ptr(), &status, NULL, &capturing_graph, NULL, NULL);

		if (status == cudaStreamCaptureStatusActive) {
			// printGraphDependencies(capturing_graph, "Captured Work Before EndCapture");
			printGraphDependencies2(capturing_graph, "Captured Work Before EndCapture 2");
		}
	}

	CudaCheckErrorModNoSync;
	if ((MEMCPY_PEER && GRAPH_CAPTURE && id == 0) || (!MEMCPY_PEER && GRAPH_CAPTURE)) {
		{
			bool ok = true;
			if (exec_old != map_exec.end()) {
				if (!MEMCPY_PEER)
					openmp_synchronize();
				cudaStreamEndCapture(s.ptr(), &(exec_old->second.first));
				if (!MEMCPY_PEER)
					openmp_synchronize();

				CudaCheckErrorModNoSync;
				cudaGraphExecUpdateResult result;
				cudaGraphExecUpdate(exec_old->second.second, (exec_old->second.first), nullptr, &result);
				// cudaGraphExecUpdate(graph_execs[gpu], new_graph, nullptr, &result);
				// CudaCheckErrorModNoSync;
				if (result != cudaGraphExecUpdateSuccess) {
					ok = false;
					std::cout << "Graph update failed" << std::endl;
				}

			} else {

				ok = false;
				cudaGraph_t graph;
				if (!MEMCPY_PEER)
					openmp_synchronize();
				cudaStreamEndCapture(s.ptr(), &graph);
				CudaCheckErrorModNoSync;
				// cudaDeviceSynchronize();

				// CudaCheckErrorModNoSync;
				if (!MEMCPY_PEER)
					openmp_synchronize();

				exec_old = map_exec
							 .emplace(std::tuple<int, bool>{ *level, moddown },
							   cached_graph{ .first = graph,
								 .second			= nullptr,
								 .digits			= digits,
								 .buffKeyA			= ksk_a.uid,
								 .buffKeyB			= ksk_b.uid,
								 .buffAux1			= auxLimbs1.uid,
								 .buffAux2			= auxLimbs2.uid,
								 .buffC0			= c0.uid,
								 .buffC1			= c1.uid })
							 .first;
			}

			if (!ok) {
				if (exec_old != map_exec.end() && exec_old->second.second != nullptr) {
					cudaGraphExecDestroy(exec_old->second.second);
					// CudaCheckErrorModNoSync;
				}
				cudaGraphExec_t exec;
				// cudaDeviceSynchronize();
				// CudaCheckErrorModNoSync;
				cudaGraphInstantiateWithFlags(&exec, exec_old->second.first, cudaGraphInstantiateFlagUseNodePriority);
				CudaCheckErrorModNoSync;
				if (exec_old != map_exec.end()) {
					exec_old->second.second = exec;
				} else {
					map_exec.emplace(std::tuple<int, bool>{ *level, moddown },
					  cached_graph{ .first = exec_old->second.first,
						.second			   = exec,
						.digits			   = digits,
						.buffKeyA		   = ksk_a.uid,
						.buffKeyB		   = ksk_b.uid,
						.buffAux1		   = auxLimbs1.uid,
						.buffAux2		   = auxLimbs2.uid,
						.buffC0			   = c0.uid,
						.buffC1			   = c1.uid });
					exec_old = map_exec.find(std::tuple<int, bool>{ *level, moddown });
				}
			}
		}
	skip_capture:
		if ((MEMCPY_PEER && GRAPH_CAPTURE && id == 0) || (!MEMCPY_PEER && GRAPH_CAPTURE)) {
			CudaCheckErrorModNoSync;
			cudaGraphLaunch(exec_old->second.second, s.ptr());
			CudaCheckErrorModNoSync;

			// cudaGraphLaunch(exec_old->second.second, s.ptr());
			// CudaCheckErrorModNoSync;
		}
	}

	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		if (skip) {
			openmp_synchronize();
		} else {
			*thread_stop[id] += 1;
		}

		if (id != 0) {
			s.wait(*external_s[0]);
		}
	}
	c0.s.wait(s);
	auxLimbs1.s.wait(s);
	auxLimbs2.s.wait(s);
	ksk_a.getS().wait(s);
	ksk_b.getS().wait(s);
	// digits.free(s);
	cudaEventDestroy(ev);
}

void LimbPartition::modupMGPU(LimbPartition& aux, const std::vector<uint64_t*>& bufferGather_, std::vector<std::atomic_uint64_t*>& thread_stop, std::vector<Stream*>& external_s) {

	struct cached_graph {
		cudaGraph_t first;
		cudaGraphExec_t second;
		uint64_t buffC0, buffC1;
	};

	static std::map<Parameters, std::map<int, cached_graph>> map_c_to_map_graph_exec[8];
	static std::atomic_uint64_t skip;

	auto& map_exec = map_c_to_map_graph_exec[id][this->cc.param];
	auto exec_old  = map_exec.find(*level);

	cudaEvent_t ev;
	cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

	cudaSetDevice(device);
	constexpr bool PRINT = false;
	bool SELECT			 = id == 1;
	LimbPartition& c1	 = *this;
	LimbPartition& c0	 = aux;
	int num_d			 = 0;
	{
		int start = 0;
		if constexpr (PRINT)
			std::cout << "/** Compute how many digits are used at this level*/" << std::endl;
		while (num_d < cc.dnum && start < *level + 1) {
			start += DECOMPmeta.at(num_d).size();
			num_d++;
		}
	}
	uint32_t limb_size = 0;
	while (limb_size < meta.size() && meta[limb_size].id <= *level)
		limb_size++;
	const int digits_per_it = MEMCPY_PEER ? 1 : 1 /*dnum*/; // cc.logN <= 15 ? num_d : cc.logN == 16 ? std::max((num_d + 1) / 2, 1) : 1;
	s.wait(c0.s);
	if (GRAPH_CAPTURE) {
		if ((!MEMCPY_PEER || (MEMCPY_PEER && id == 0))) {

			if (exec_old != map_exec.end()) {
				auto& graph_data = exec_old->second;
				if (graph_data.buffC0 == c0.uid && graph_data.buffC1 == c1.uid) {
					//   std::cout << "Graph for keyswitch is the same, skip capture, level=" << *level
					//             << " moddown=" << moddown << std::endl;
					skip = 1;
				} else {
					//   std::cout << "Graph for keyswitch needs changes, run capture level=" << *level
					//             << " moddown=" << moddown << std::endl;
					skip = 0;
				}
			} else {
				skip = 0;
			}
		}
	} else {
		skip = 0;
	}
	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		*thread_stop[id] += 1;
		if (id == 0) {
			for (uint32_t peer = 1; peer < cc.GPUid.size(); ++peer) {
				while (*thread_stop[peer] < *thread_stop[0])
					;

				s.wait(*external_s[peer]);
			}
			if (skip == 1) {
				*thread_stop[id] += 1;
				goto skip_capture;
			}
			cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeGlobal /*cudaStreamCaptureModeRelaxed*/);
		} else {
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;

			if (skip == 1) {
				*thread_stop[id] += 1;
				goto skip_capture;
			}
		}

		if (id != 0) {
			s.wait(*external_s[0]);
			*thread_stop[id] += 1;
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;
		} else {
			*thread_stop[id] += 1;
			while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
				;
		}
	} else {
		if (skip == 1) {
			goto skip_capture;
		}
		openmp_synchronize();
		if (GRAPH_CAPTURE) {
			CudaCheckErrorModNoSync;
			// cudaDeviceSynchronize();
			cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeThreadLocal /*cudaStreamCaptureModeRelaxed*/);
		}
	}

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Input: ";
			for (size_t i = 0; i < limb_size; ++i) {
				std::cout << meta[i].id;
				SWITCH(limb[i], printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	if constexpr (PRINT)
		std::cout << "/** We try to pipeline the computation of each digit first, splitting independent groups of limbs*/" << std::endl;
	if constexpr (PRINT)
		std::cout << "GPU " << id << "compute " << num_d << " digits" << std::endl;
	for (int d = 0; d < num_d; d += digits_per_it) {
		Stream& stream = cc.digitStream.at(d).at(id);
		stream.wait(s);
	}
	for (int d = 0; d < num_d; d += digits_per_it) {
		int ds			 = std::min(num_d - d, digits_per_it);
		uint32_t start_d = 0;
		while (start_d < limb_size && meta[start_d].digit < d)
			start_d++;
		uint32_t size_d = 0;
		while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
			size_d++;
		if constexpr (PRINT)
			if (SELECT) {
				std::cout << "GPU " << id << " for digits " << d << ":" << d + digits_per_it << " INTT " << size_d << " limbs starting at limb " << start_d << std::endl;
			}
		Stream& stream = cc.digitStream.at(d).at(id);
		// stream.wait(s);
		if constexpr (PRINT)
			std::cout << "/** Intt */" << std::endl;
		if (size_d > 0) {
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;
			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int gather_offset	= 0;
			for (int i = 0; i < id; ++i) {
				gather_offset += cc.meta.at(i).size();
			}
			{
				INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size_d }, blockDimFirst, bytesFirst, stream.ptr()>>>(
				  getGlobals(), limbptr.data + start_d, PARTITION(id, start_d), auxptr.data + start_d);
				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size_d }, blockDimSecond, bytesSecond, stream.ptr()>>>(
				  getGlobals(), auxptr.data + start_d, PARTITION(id, start_d), GATHERptr.data + gather_offset + start_d);
			}
		}
		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out INTT: ";
				for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
					for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
						std::cout << DECOMPmeta[j][i].id;
						SWITCH(DECOMPlimb[j][i], printThisLimb(2));
					}
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}
		if constexpr (PRINT)
			std::cout << "/** Communicate */" << std::endl;
		{
			if (!PEER_ACCESS && cc.GPUid.size() > 1) {

				if (MEMCPY_PEER) {

					for (size_t i = 0; i < cc.GPUid.size(); ++i) {
						cc.digitStreamForMemcpyPeer.at(d).at(i).at(id).wait(
						  s); // ensure gather buffer is created before peer memcpy, we use s as the dependency will be older than on stream
					}
					if (MEMCPY_PEER && GRAPH_CAPTURE) {
						*thread_stop[id] += 1;
						if (id != 0) {
							while (*thread_stop[id] >= *thread_stop[id - 1])
								;
						} else {
							while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
								;
						}
					} else {
						openmp_synchronize();
					}
					{

						int start = 0;
						for (size_t i = 0; i < cc.GPUid.size(); ++i) {

							uint32_t limb_size_i = 0;
							while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
								limb_size_i++;
							uint32_t start_d_i = 0;
							while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
								start_d_i++;
							uint32_t size_d_i = 0;
							while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
								size_d_i++;
							if constexpr (PRINT)
								if (SELECT) {
									std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it << " communicate " << size_d_i << " limbs" << std::endl;
								}
							if (size_d_i > 0) {

								if (i == static_cast<size_t>(id)) {
									for (size_t j = 0; j < cc.GPUid.size(); ++j) {

										if (j != i) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(i).at(j);
											stream_.wait(stream);

											transferKernel((float*)(bufferGATHER + cc.N * (start + start_d_i)),
											  (float*)(bufferGather_[j] + cc.N * (start + start_d_i)),
											  sizeof(uint64_t) * size_d_i * cc.N / sizeof(float),
											  stream_.ptr(),
											  device,
											  cc.GPUid[j]);
											// cudaMemcpyPeerAsync(bufferGather_[j] + cc.N * (start + start_d_i),
											//                     cc.GPUid[j], bufferGATHER + cc.N * (start + start_d_i),
											//                     cc.GPzUid[i], sizeof(uint64_t) * size_d_i * cc.N,
											//                     stream_.ptr());

											// cc.digitStream.at(d).at(j).wait(stream_);
											stream_.record();
											CudaCheckErrorModNoSync;
										}
									}

									for (size_t j = 0; j < cc.GPUid.size(); ++j) {
										if (j != i) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(i).at(j);
											// stream.wait(stream_);
										}
									}
								}
							}
							start += cc.meta[i].size();
						}
					}
					if (MEMCPY_PEER && GRAPH_CAPTURE) {
						*thread_stop[id] += 1;
						if (id != 0) {
							while (*thread_stop[id] >= *thread_stop[id - 1])
								;
						} else {
							while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
								;
						}
					} else {
						openmp_synchronize();
					}
					{
						int start = 0;
						for (size_t i = 0; i < cc.GPUid.size(); ++i) {

							uint32_t limb_size_i = 0;
							while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
								limb_size_i++;
							uint32_t start_d_i = 0;
							while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
								start_d_i++;
							uint32_t size_d_i = 0;
							while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
								size_d_i++;

							if (size_d_i > 0) {
								for (size_t j = 0; j < cc.GPUid.size(); ++j) {
									if (j != i) {
										if (j == static_cast<size_t>(id)) {
											Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(i).at(j);

											stream.wait(stream_);
										}
									}
								}
							}
							start += cc.meta[i].size();
						}
					}
				} else {
#ifdef NCCL
					NCCLCHECK(ncclGroupStart());
					int start = 0;
					for (size_t i = 0; i < cc.GPUid.size(); ++i) {
						uint32_t limb_size_i = 0;
						while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
							limb_size_i++;
						uint32_t start_d_i = 0;
						while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
							start_d_i++;
						uint32_t size_d_i = 0;
						while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
							size_d_i++;
						if constexpr (PRINT)
							if (SELECT) {
								std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it << " communicate " << size_d_i << " limbs" << std::endl;
							}
						if (size_d_i > 0) {

							NCCLCHECK(ncclBroadcast(
							  bufferGATHER + cc.N * (start + start_d_i), bufferGATHER + cc.N * (start + start_d_i), size_d_i * cc.N, ncclUint64, i, rank, stream.ptr()));
						}
						start += cc.meta[i].size();
					}
					NCCLCHECK(ncclGroupEnd());
#else
					assert(false);
#endif
				}
			} else if (cc.GPUid.size() > 1) {
				stream.record();

				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
					}
				} else {
					openmp_synchronize();
				}
				for (size_t i = 0; i < cc.GPUid.size(); ++i) {
					if (static_cast<size_t>(id) != i)
						stream.wait(cc.digitStream[d][i]);
				}
				if (MEMCPY_PEER && GRAPH_CAPTURE) {
					*thread_stop[id] += 1;
					if (id != 0) {
						while (*thread_stop[id] >= *thread_stop[id - 1])
							;
					} else {
						while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
							;
					}
				} else {
					openmp_synchronize();
				}
			}
		}
		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out INTT after communicate: ";
				for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
					for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
						std::cout << DECOMPmeta[j][i].id;
						SWITCH(DECOMPlimb[j][i], printThisLimb(2));
					}
					std::cout << std::endl;
				}
				cudaDeviceSynchronize();
			}
		}
		// for (int d = 0; d < num_d; d += digits_per_it)
		{
			// int ds			 = std::min(num_d - d, digits_per_it);
			// uint32_t start_d = 0;
			// while (start_d < limb_size && meta[start_d].digit < d)
			//	start_d++;
			// uint32_t size_d = 0;
			// while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
			//	size_d++;
			// Stream& stream = cc.digitStream.at(d).at(id);
			if constexpr (PRINT)
				std::cout << "/** Conv */" << std::endl;
			for (int d_ = d; d_ < d + ds; ++d_) {
				Stream& stream1 = cc.digitStream.at(d_).at(id);
				stream1.wait(stream);
			}
			for (int d_ = d; d_ < d + ds; ++d_) {
				Stream& stream1 = cc.digitStream.at(d_).at(id);

				int start = 0;
				for (int j = 0; j < d_; ++j)
					start += DECOMPlimb.at(j).size();
				int size = std::min((int)DECOMPlimb.at(d_).size(), *level + 1 - start);
				if (size <= 0) {
					std::cerr << "void modup, aborting" << std::endl;
					exit(-1);
				}
				if constexpr (PRINT)
					if (SELECT) {
						std::cout << cc.precom.constants[id].num_primeid_digit_to[d_][*level] << "<- num_prime_id_digit_to: " << d_ << std::endl;
						std::cout << cc.precom.constants[id].num_primeid_digit_from[d_][*level] << "<- num_prime_id_digit_from: " << d_ << std::endl;
						/*
							std::cout << host_constants_per_gpu[id].num_primeid_digit_to[d_][level - 1]
									  << "<- num_prime_id_digit_to: " << d_ << std::endl;
							std::cout << host_constants_per_gpu[id].num_primeid_digit_from[d_][level - 1]
									  << "<- num_prime_id_digit_from: " << d_ << std::endl;
									  */
					}

				CudaCheckErrorModNoSync;
				if (d_ > 0)
					cudaStreamWaitEvent(stream1.ptr(), ev);
				CudaCheckErrorModNoSync;
				if (!PEER_ACCESS) {
					dim3 blockSize{ 64, 2 };
					dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
					int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
					DecompAndModUpConv<ALGO_SHOUP>
					  <<<gridSize, blockSize, shared_bytes, stream1.ptr()>>>(DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_], getGlobals());
					cc.digitStream2.at(d_).at(id).wait(stream1); /** Get dependency for limb NTTs later */
				} else {

					dim3 blockSize{ 64, 2 };
					dim3 gridSize{ (uint32_t)cc.N / blockSize.x / 2 };
					int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x * 2;
					DecompAndModUpConv_spec2<ALGO_SHOUP>
					  <<<gridSize, blockSize, shared_bytes, stream1.ptr()>>>(DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_], getGlobals());
					cc.digitStream2.at(d_).at(id).wait(stream1); /** Get dependency for limb NTTs later */
				}
				CudaCheckErrorModNoSync;
				cudaEventRecord(ev, stream1.ptr());
				CudaCheckErrorModNoSync;
				if constexpr (PRINT)
					std::cout << "/** NTT special limbs */" << std::endl;
				{
					// uint32_t size		= cc.splitSpecialMeta.at(id).size();
					uint32_t size		= cc.precom.constants[id].num_primeid_digit_to[d][*level];
					constexpr ALGO algo = ALGO_SHOUP;
					constexpr int M		= 4;
					dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
					dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
					int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
					int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
					if (size > 0) {
						NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size }, blockDimFirst, bytesFirst, stream1.ptr()>>>(
						  getGlobals(), DIGITlimbptr[d_].data, DIGIT(d_, 0), c0.DIGITlimbptr[d_].data);
						NTT_<true, algo, NTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size }, blockDimSecond, bytesSecond, stream1.ptr()>>>(
						  getGlobals(), c0.DIGITlimbptr[d_].data, DIGIT(d_, 0), DIGITlimbptr[d_].data);
					}
				}
			}
		}
		if constexpr (PRINT) {
			if (SELECT) {
				cudaDeviceSynchronize();
				std::cout << "GPU: " << id << "Out ModUp after NTT specials: ";
				for (size_t j = 0; j < DIGITlimb.size(); ++j) {
					for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
						std::cout << DIGITmeta[j][i].id;
						SWITCH(DIGITlimb[j][i], printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}
		if constexpr (PRINT)
			std::cout << "/** We delay the call of NTTs post-modup for non special limbs to here*/" << std::endl;
		// for (int d = 0; d < num_d; ++d)
		if (0) {
			Stream& stream = cc.digitStream2.at(d).at(id);
			if (limb_size > 0) {
				uint32_t start = cc.splitSpecialMeta.at(id).size();
				uint32_t size  = cc.precom.constants[id].num_primeid_digit_to[d][*level] - start;
				if (size > 0) {
					constexpr ALGO algo = ALGO_SHOUP;
					constexpr int M		= 4;
					dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
					dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
					int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
					int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
					{
						NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), size }, blockDimFirst, bytesFirst, stream.ptr()>>>(
						  getGlobals(), DIGITlimbptr[d].data + start, DIGIT(d, start), c0.DIGITlimbptr[d].data + start);
						NTT_<true, algo, NTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), size }, blockDimSecond, bytesSecond, stream.ptr()>>>(
						  getGlobals(), c0.DIGITlimbptr[d].data + start, DIGIT(d, start), DIGITlimbptr[d].data + start);
					}
				}
				s.wait(stream);
			}
		}
	}
	for (int d = 0; d < num_d; ++d) {
		s.wait(cc.digitStream.at(d).at(id));
	}
	c0.s.wait(s);

	if (cc.GPUid.size() > 1) {
		if (!PEER_ACCESS && MEMCPY_PEER) {
			for (int d = 0; d < num_d; d += digits_per_it) {
				for (size_t j = 0; j < cc.GPUid.size(); ++j) {
					if (j != static_cast<size_t>(id)) {
						Stream& stream_ = cc.digitStreamForMemcpyPeer.at(d).at(id).at(j);
						s.wait(stream_);
					}
				}
			}
		}
	}

	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		*thread_stop[id] += 1;
		if (id != 0) {
			while (*thread_stop[id] >= *thread_stop[id - 1])
				;
		} else {
			while (*thread_stop[id] > *thread_stop[cc.GPUid.size() - 1])
				;
			for (uint32_t i = 1; i < external_s.size(); ++i) {
				s.wait(*external_s[i]);
			}
		}
	} else {
		openmp_synchronize();
	}

	if ((MEMCPY_PEER && GRAPH_CAPTURE && id == 0) || (!MEMCPY_PEER && GRAPH_CAPTURE)) {
		{
			bool ok = true;
			if (exec_old != map_exec.end()) {
				if (!MEMCPY_PEER)
					openmp_synchronize();
				cudaStreamEndCapture(s.ptr(), &(exec_old->second.first));
				if (!MEMCPY_PEER)
					openmp_synchronize();

				CudaCheckErrorModNoSync;
				cudaGraphExecUpdateResult result;
				cudaGraphExecUpdate(exec_old->second.second, (exec_old->second.first), nullptr, &result);
				// cudaGraphExecUpdate(graph_execs[gpu], new_graph, nullptr, &result);
				if (result != cudaGraphExecUpdateSuccess) {
					ok = false;
					std::cout << "Graph update failed" << std::endl;
				}

			} else {

				ok = false;
				cudaGraph_t graph;
				if (!MEMCPY_PEER)
					openmp_synchronize();
				cudaStreamEndCapture(s.ptr(), &graph);
				CudaCheckErrorModNoSync;
				if (!MEMCPY_PEER)
					openmp_synchronize();

				exec_old = map_exec.emplace(*level, cached_graph{ .first = graph, .second = nullptr, .buffC0 = c0.uid, .buffC1 = c1.uid }).first;
			}

			CudaCheckErrorModNoSync;
			if (!ok) {
				if (exec_old != map_exec.end() && exec_old->second.second != nullptr) {
					cudaGraphExecDestroy(exec_old->second.second);
					CudaCheckErrorModNoSync;
				}
				cudaGraphExec_t exec;
				// cudaDeviceSynchronize();
				CudaCheckErrorModNoSync;
				cudaGraphInstantiateWithFlags(&exec, exec_old->second.first, cudaGraphInstantiateFlagUseNodePriority);
				CudaCheckErrorModNoSync;
				if (exec_old != map_exec.end()) {
					exec_old->second.second = exec;
				} else {
					map_exec.emplace(*level, cached_graph{ .first = exec_old->second.first, .second = exec, .buffC0 = c0.uid, .buffC1 = c1.uid });
					exec_old = map_exec.find(*level);
				}
			}
		}
	skip_capture:
		if ((MEMCPY_PEER && GRAPH_CAPTURE && id == 0) || (!MEMCPY_PEER && GRAPH_CAPTURE)) {
			CudaCheckErrorModNoSync;
			cudaGraphLaunch(exec_old->second.second, s.ptr());
			CudaCheckErrorModNoSync;
		}
	}

	if (MEMCPY_PEER && GRAPH_CAPTURE) {
		if (skip) {
			openmp_synchronize();
		} else {
			*thread_stop[id] += 1;
		}

		if (id != 0) {
			s.wait(*external_s[0]);
		}
	} else if (cc.GPUid.size() > 0 && PEER_ACCESS) {
		s.record();
		openmp_synchronize();
		for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
			if (static_cast<uint32_t>(id) != i) {
				s.wait(*external_s[i]);
			}
		}
	}
	aux.s.wait(s);

	cudaEventDestroy(ev);
}

void LimbPartition::moddownMGPU(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs, const std::vector<uint64_t*>& bufferSpecial_) {
	cudaSetDevice(device);
	constexpr bool PRINT = false;
	bool SELECT			 = id == 1;
	LimbPartition& c1	 = *this;

	uint32_t limb_size = getLimbSize(*level);
	CudaCheckErrorModNoSync;
	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Input: ";
			for (size_t i = 0; i < limb_size; ++i) {
				std::cout << meta[i].id;
				SWITCH(limb[i], printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	{

		CudaCheckErrorModNoSync;

		Stream& stream = cc.digitStream2.at(0).at(id);
		stream.wait(s);
		stream.wait(auxLimbs.s);
		LimbPartition& out = c1;
		if constexpr (PRINT)
			std::cout << "/** INTT specials*/" << std::endl;
		{
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			const uint32_t limbs = cc.splitSpecialMeta.at(id).size();
			if (limbs > 0) {
				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				const int i = cc.splitSpecialMeta.at(id).at(0).id - cc.specialMeta.at(id).at(0).id;

				INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), limbs }, blockDimFirst, bytesFirst, stream.ptr()>>>(
				  getGlobals(), out.SPECIALlimbptr.data + i, SPECIAL(id, i), out.SPECIALauxptr.data + i);
				CudaCheckErrorModNoSync;
				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), limbs }, blockDimSecond, bytesSecond, stream.ptr()>>>(
				  getGlobals(), out.SPECIALauxptr.data + i, SPECIAL(id, i), auxLimbs.SPECIALlimbptr.data + i);
				CudaCheckErrorModNoSync;
			}

			if constexpr (PRINT)
				std::cout << "/** communicate */" << std::endl;
		}
		if (!PEER_ACCESS && cc.GPUid.size() > 1) {
			if (MEMCPY_PEER) {
				for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
					cc.digitStreamForMemcpyPeer[0].at(j).at(id).wait(s); // to pass dependency up to the communication
				}
				CudaCheckErrorModNoSync;
				openmp_synchronize();

				for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
					// Limb<uint64_t>& l = std::get<U64>(
					//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
					const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
					// uint64_t* ptr2 = l.v.data;

					if (static_cast<uint32_t>(id) == i) {
						if (num_limbs > 0) {
							uint64_t* ptr = auxLimbs.bufferSPECIAL + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N;

							for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
								if (j != i) {
									Stream& stream_ = cc.digitStreamForMemcpyPeer.at(0).at(i).at(j);
									stream_.wait(stream);

									CudaCheckErrorModNoSync;
									cudaMemcpyPeerAsync(bufferSpecial_[j] + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N,
									  cc.GPUid[j],
									  ptr,
									  cc.GPUid[i],
									  sizeof(uint64_t) * cc.N * num_limbs,
									  stream_.ptr());
									CudaCheckErrorModNoSync;
								}
							}

							for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
								if (j != i) {
									Stream& stream_ = cc.digitStreamForMemcpyPeer.at(0).at(id).at(j);
									// stream.wait(stream_);
								}
							}
						}
					}
				}
				openmp_synchronize();

				for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
					// Limb<uint64_t>& l = std::get<U64>(
					//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
					const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
					// uint64_t* ptr2 = l.v.data;

					if (num_limbs > 0) {

						for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
							if (j == static_cast<uint32_t>(id)) {
								if (i != j) {
									Stream& stream_ = cc.digitStreamForMemcpyPeer.at(0).at(i).at(j);
									stream.wait(stream_);
									CudaCheckErrorModNoSync;
								}
							}
						}
					}
				}
			} else {
#ifdef NCCL
				NCCLCHECK(ncclGroupStart());
				for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
					// Limb<uint64_t>& l = std::get<U64>(
					//     this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
					const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
					// uint64_t* ptr2 = l.v.data;
					if (num_limbs > 0) {
						uint64_t* ptr = auxLimbs.bufferSPECIAL + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N;
						NCCLCHECK(ncclBroadcast(ptr, ptr, cc.N * num_limbs, ncclUint64, (int)i, rank, stream.ptr()));
					}
				}
				NCCLCHECK(ncclGroupEnd());
#else
				assert(false);
#endif
			}
		} else if (cc.GPUid.size() > 1) {
			stream.record();

			openmp_synchronize();

			for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
				if (static_cast<uint32_t>(id) != i)
					stream.wait(cc.digitStream2.at(0).at(i));
			}
		}
	}

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "KSK specials after INTT and communicate: ";
			for (const auto& j : { &c1 }) {
				for (auto& i : j->SPECIALlimb) {
					SWITCH(i, printThisLimb(2));
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	if (limb_size > 0) {
		Stream& stream = cc.digitStream2.at(0).at(id);
		// LimbPartition& out = *this;
		if constexpr (PRINT)
			std::cout << "/** Conv */" << std::endl;

		if (!PEER_ACCESS) {
			dim3 blockSize{ 64, 2 };

			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;
			if (limb_size > 0)
				ModDown2<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr()>>>(
				  auxLimbs.limbptr.data, limb_size, auxLimbs.SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
		} else {
			dim3 blockSize{ 64, 2 };

			dim3 gridSize{ (uint32_t)cc.N / blockSize.x / 2 };
			int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x * 2;
			if (limb_size > 0)
				ModDown3<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr()>>>(
				  auxLimbs.limbptr.data, limb_size, auxLimbs.SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
		}
		CudaCheckErrorModNoSync;
	}

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Out Moddown: ";
			for (const auto& j : { &auxLimbs }) {
				for (auto& i : j->limb) {
					SWITCH(i, printThisLimb(2));
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	{
		if constexpr (PRINT)
			std::cout << "/** Last NTT step for moddown*/" << std::endl;
		Stream& stream	   = cc.digitStream2.at(0).at(id);
		LimbPartition& out = *this;

		if (limb_size > 0) {
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			{
				NTT_<false, algo, NTT_MODDOWN><<<dim3{ cc.N / (blockDimFirst.x * M * 2), limb_size }, blockDimFirst, bytesFirst, stream.ptr()>>>(
				  getGlobals(), auxLimbs.limbptr.data, PARTITION(id, 0), out.auxptr.data);
				CudaCheckErrorModNoSync;

				NTT_<true, algo, NTT_MODDOWN><<<dim3{ cc.N / (blockDimSecond.x * M * 2), limb_size }, blockDimSecond, bytesSecond, stream.ptr()>>>(
				  getGlobals(), out.auxptr.data, PARTITION(id, 0), out.limbptr.data);
				CudaCheckErrorModNoSync;
			}

			auxLimbs.s.wait(stream);
			s.wait(stream);
			CudaCheckErrorModNoSync;
		}
	}

	if constexpr (PRINT) {
		if (SELECT) {
			cudaDeviceSynchronize();
			std::cout << "GPU: " << id << "Out Moddown after submult: ";
			for (const auto& j : { &c1 }) {
				for (auto& i : j->limb) {
					SWITCH(i, printThisLimb(2));
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}

	if (cc.GPUid.size() > 1) {
		if (!PEER_ACCESS && MEMCPY_PEER) {
			for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
				if (j != static_cast<uint32_t>(id)) {
					Stream& stream_ = cc.digitStreamForMemcpyPeer.at(0).at(id).at(j);
					s.wait(stream_);
					CudaCheckErrorModNoSync;
				}
			}
		} else if (PEER_ACCESS) {
			s.record();
			openmp_synchronize();
			for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
				if (j != static_cast<uint32_t>(id)) {
					Stream& stream_ = cc.digitStream2.at(0).at(j);
					s.wait(stream_);
					CudaCheckErrorModNoSync;
				}
			}
		}
	}
}

void LimbPartition::broadcastLimb0_mgpu() {
	cudaSetDevice(device);
	static bool parity = true;
	const int limbsize = getLimbSize(*level);

	Stream& stream		  = parity ? cc.top_limb_stream[id] : cc.top_limb_stream2[id];
	uint64_t* buffer	  = parity ? cc.top_limb_buffer[id] : cc.top_limb_buffer2[id];
	VectorGPU<void*>& ptr = parity ? cc.top_limbptr[id] : cc.top_limbptr2[id];

	stream.wait(s);
	bool skip0 = meta[0].id == 0;

	if (skip0) {
		uint64_t* src_ptr = std::get<U64>(limb[0]).v.data;
		cudaMemcpyAsync(buffer, src_ptr, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream.ptr());
	}
	/*
#ifdef NCCL
	{ NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr())); }
#else
	assert(false);
#endif
*/
#ifdef NCCL
	if constexpr (0) {
		NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr()));
	} else {
		NCCLCHECK(ncclGroupStart());
		if (static_cast<uint32_t>(id) == cc.limbGPUid[0].x) {
			for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
				if (i != static_cast<uint32_t>(id))
					ncclSend(buffer, cc.N, ncclUint64, i, rank, stream.ptr());
			}
		} else {
			ncclRecv(buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr());
		}
		NCCLCHECK(ncclGroupEnd());
	}
#else
	assert(false);
#endif
	if (limbsize - skip0 > 0) {
		broadcastLimb0_mgpu_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)limbsize - skip0 }, 128, 0, stream.ptr()>>>(limbptr.data + skip0, PARTITION(id, skip0), ptr.data);
		if (MODRAISE_WITH_P0) {
			if (SPECIALmeta.size() > 0 && SPECIALmeta.at(0).id == cc.L + 1)
				broadcastLimb0_mgpu_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)1 }, 128, 0, s.ptr()>>>(SPECIALlimbptr.data, SPECIAL(id, 0), limbptr.data);
		}
	}
	s.wait(stream);
}

} // namespace FIDESlib::CKKS
