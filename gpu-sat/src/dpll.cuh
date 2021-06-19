#pragma once

#include "cnf.cuh"

namespace host
{
    void dpll_solve(term_val* cnf_matrix, size_t nb_var, size_t nb_clause,
                    bool* solution);

}

namespace device
{
    __global__ void check_conflict(term_val* cnf_matrix, size_t nb_var,
                                   size_t nb_clause, size_t constant_pos,
                                   term_val constant_sign, bool* results,
                                   bool* mask);

    __global__ void simplify(term_val* cnf_matrix, size_t nb_var,
                             size_t nb_clause, term_val* constants, bool* mask);
} // namespace device
