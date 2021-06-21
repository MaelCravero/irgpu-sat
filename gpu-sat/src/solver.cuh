#pragma once

#include "cnf.cuh"

namespace host
{
    std::optional<solution> solve_v1(Cnf cnf);
    std::optional<solution> solve_v2(Cnf cnf);
    std::optional<solution> solve_v2_no_pitch(Cnf cnf);
    std::optional<solution> solve_v3(Cnf cnf);
    std::optional<solution> solve_v3_no_pitch(Cnf cnf);
} // namespace host

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

    __global__ void remove_clauses(term_val* cnf_matrix, size_t pitch,
                                   size_t nb_var, size_t nb_clause,
                                   term_val* constants, bool* mask);

    __global__ void remove_terms(term_val* cnf_matrix, size_t pitch,
                                 size_t nb_var, size_t nb_clause,
                                 term_val* constants, bool* mask);
} // namespace device
