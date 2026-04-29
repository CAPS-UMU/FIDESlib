//
// Created by carlosad on 27/03/25.
//

#ifndef FIDESLIB_PARALLEL_FOR_HPP
#define FIDESLIB_PARALLEL_FOR_HPP

#include <functional>

namespace FIDESlib {

void parallel_for(int init, int end, int increment, const std::function<void(int)>& f);

void openmp_synchronize();

} // namespace FIDESlib

#endif // FIDESLIB_PARALLEL_FOR_HPP
