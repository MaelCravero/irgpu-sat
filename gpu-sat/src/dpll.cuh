#pragma once

#include "cnf.cuh"

namespace device
{
    __global__ void dpll_solve(term_val* cnf_matrix, size_t nb_var,
                               size_t nb_clause, char* solution);

    __device__ void check_conflict(term_val* cnf_matrix, size_t nb_var,
                                   term* constants, bool* results, bool* mask);

    __device__ void simplify(term_val* cnf_matrix, size_t nb_var,
                             term* constants, bool* mask);
}
