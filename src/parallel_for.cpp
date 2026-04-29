//
// Created by carlosad on 27/03/25.
//
#include "parallel_for.hpp"
#include <cassert>
#include <omp.h>

void FIDESlib::parallel_for(int init, int end, int increment, const std::function<void(int)>& f) {
#pragma omp parallel num_threads((end - init) / increment)
	{
		assert(omp_get_num_threads() == (end - init) / increment);
		int i = init + increment * omp_get_thread_num();
		// for (int i = init; i < end; i += increment) {
		f(i);
		//}
	}
}

void FIDESlib::openmp_synchronize() {
#pragma omp barrier
}
