#pragma once

#include "cnf.cuh"

namespace host
{
    std::optional<solution> dpll_solve(Cnf cnf);

}

namespace device
{
    __global__ void the_one_true_kernel(term_val* cnf_matrix, size_t pitch,
                                        size_t nb_var, size_t nb_clause,
                                        term_val* constants,
                                        size_t constant_pos,
                                        term_val constant_sign, bool* results);

    __global__ void check_conflict(term_val* cnf_matrix, size_t pitch,
                                   size_t nb_var, size_t nb_clause,
                                   size_t constant_pos, term_val constant_sign,
                                   bool* results, bool* mask);

    __global__ void simplify(term_val* cnf_matrix, size_t pitch, size_t nb_var,
                             size_t nb_clause, term_val* constants, bool* mask);

    __global__ void remove_terms(term_val* cnf_matrix, size_t pitch,
                                 size_t nb_var, size_t nb_clause,
                                 term_val* constants, bool* mask);
} // namespace device
