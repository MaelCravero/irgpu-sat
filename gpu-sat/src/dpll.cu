#include <optional>
#include <vector>

#include "cnf.cuh"
#include "dpll.cuh"
#include "utils.cuh"

namespace host
{
    namespace
    {
        void assign_decision_litteral(std::vector<term_val>& constants,
                                      size_t pos)
        {
            constants[pos] = 1;
        }

        void backjump(std::vector<term_val>& constants, size_t& pos)
        {
            while (pos && constants[pos] < 0)
                pos--;

            constants[pos] *= -1;
        }

        solution calculate_solution(std::vector<term_val>& constants)
        {
            solution sol;

            int i = 1;
            for (auto sign : constants)
                sol.emplace_back(i * sign);

            return sol;
        }

    } // namespace

    std::optional<solution> dpll_solve(term_val* cnf_matrix, size_t nb_var,
                                       size_t nb_clause)
    {
        // FIXME error codes should be checked (but i'm lazy)

        std::vector<term_val> constants(nb_var);

        int nb_blocks = (nb_clause / 1024) + 1;

        size_t constant_pos = 0;
        for (;;)
        {
            term_val* local_cnf;
            cudaMalloc(&local_cnf, nb_var * nb_clause * sizeof(term_val));
            cudaMemcpy(local_cnf, cnf_matrix,
                       nb_var * nb_clause * sizeof(term_val),
                       cudaMemcpyHostToDevice); // cnf seems to be on host

            size_t clause_size = nb_clause * sizeof(bool);
            bool* mask;
            cudaMalloc(&mask, clause_size);

            auto dev_constants = utils::init_from(constants);

            device::simplify<<<nb_blocks, 1024>>>(local_cnf, nb_var, nb_clause,
                                                  dev_constants, mask);

            cudaFree(dev_constants);

            bool* results;
            cudaMalloc(&results, nb_clause * sizeof(bool));

            device::check_conflict<<<nb_blocks, 1024>>>(
                local_cnf, nb_var, nb_clause, constant_pos,
                constants[constant_pos], results, mask);

            bool* host_res = (bool*)malloc(clause_size);
            cudaMemcpy(host_res, results, clause_size, cudaMemcpyDeviceToHost);

            bool conflict = false;
            for (auto i = 0; i < nb_clause && !conflict; i++)
                if (host_res[i])
                    conflict = true;

            cudaFree(local_cnf);
            cudaFree(mask);
            cudaFree(results);

            free(host_res);

            if (conflict)
            {
                backjump(constants, constant_pos);

                if (!constant_pos)
                    return {};
            }
            else if (constant_pos <= nb_var)
            {
                assign_decision_litteral(constants, constant_pos);
            }
            else
                break;
        }

        return {calculate_solution(constants)};
    }
} // namespace host

namespace device
{
    __global__ void check_conflict(term_val* cnf_matrix, size_t nb_var,
                                   size_t nb_clause, size_t constant_pos,
                                   term_val constant_sign, bool* results,
                                   bool* mask)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause || mask[x])
            return;

        bool conflict = cnf_matrix[x + constant_pos] == -constant_sign;

        if (!conflict)
            return;

        int vars_in_clause = 0;
        for (auto i = x * nb_var; i < (x + 1); i++)
        {
            if (cnf_matrix[i])
                if (!vars_in_clause)
                    vars_in_clause++;
                else
                {
                    vars_in_clause++;
                    break;
                }
        }

        if (vars_in_clause > 1)
            results[x] = true;
    }

    __global__ void simplify(term_val* cnf_matrix, size_t nb_var,
                             size_t nb_clause, term_val* constants, bool* mask)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause || mask[x])
            return;

        for (auto i = 0; i < nb_var; i++)
        {
            auto pos = x * nb_var + i;

            if (cnf_matrix[pos] && cnf_matrix[pos] == -constants[i])
                cnf_matrix[pos] = 0; // The term can be removed

            else if (cnf_matrix[pos] && cnf_matrix[pos] == constants[i])
            {
                mask[x] = true; // The clause is true
                return;
            }
        }
    }

} // namespace device
