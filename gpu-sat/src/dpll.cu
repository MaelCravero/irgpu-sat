#include <optional>
#include <vector>
#include <iostream>

#include "cnf.cuh"
#include "dpll.cuh"
#include "utils.cuh"

namespace host
{
    namespace
    {
        void backjump(std::vector<term_val>& constants, size_t& pos)
        {
            while (pos >= 1 && constants[pos - 1] > 0)
            {
                constants[pos - 1] = 0;
                pos--;
            }

            if (pos)
                constants[pos - 1] *= -1;
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
        for (;;) {

            std::cout << "\n" << "curent constants:\n";
            for (auto tv : constants)
                std::cout << (int)tv << " ";
            std::cout << "\n";

            term_val* local_cnf;
            cudaMalloc(&local_cnf, nb_var * nb_clause * sizeof(term_val));
            cudaMemcpy(local_cnf, cnf_matrix,
                       nb_var * nb_clause * sizeof(term_val),
                       cudaMemcpyHostToDevice); // cnf seems to be on host

            size_t clause_size = nb_clause * sizeof(bool);
            bool* mask;
            cudaMalloc(&mask, clause_size);

            auto cur_constant = 0;
            if (constant_pos)
            {
                cur_constant = constants[constant_pos - 1];
                constants[constant_pos - 1] = 0;
            }

            auto dev_constants = utils::init_from(constants);

            if (constant_pos)
            {
                constants[constant_pos - 1] = cur_constant;
            }

            device::simplify<<<nb_blocks, 1024>>>(local_cnf, nb_var, nb_clause,
                                                  dev_constants, mask);

            term_val* host_local_cnf = (term_val*)malloc(nb_clause * sizeof(term_val) * nb_var);
            cudaMemcpy(host_local_cnf, local_cnf, nb_clause * sizeof(term_val) * nb_var, cudaMemcpyDeviceToHost);

            bool* host_mask = (bool*)malloc(clause_size);
            cudaMemcpy(host_mask, mask, clause_size, cudaMemcpyDeviceToHost);

            std::cout << "cnf after simplify";
            for (int i = 0; i < nb_var * nb_clause; i++)
            {
                if (i % nb_var == 0)
                    std::cout << "\nMask = " << std::boolalpha << host_mask[i / nb_var] << "   ";
                std::cout << (int)host_local_cnf[i] << " ";
            }
            std::cout << "\n";

            free(host_local_cnf);
            free(host_mask);

            cudaFree(dev_constants);
            bool conflict = false;

            if (constant_pos)
            {
                bool* results;
                cudaMalloc(&results, nb_clause * sizeof(bool));

                device::check_conflict<<<nb_blocks, 1024>>>(
                    local_cnf, nb_var, nb_clause, constant_pos - 1,
                    constants[constant_pos - 1], results, mask);

                bool* host_res = (bool*)malloc(clause_size);
                cudaMemcpy(host_res, results, clause_size, cudaMemcpyDeviceToHost);

                for (auto i = 0; i < nb_clause; i++)
                {
                    std::cout << "conflict: " << std::boolalpha << host_res[i] << "\n";
                    if (host_res[i])
                        conflict = true;
                }

                free(host_res);
                cudaFree(results);
            }

            cudaFree(local_cnf);
            cudaFree(mask);

            if (conflict)
            {
                backjump(constants, constant_pos);

                if (!constant_pos)
                    return {};
            }
            else if (constant_pos < nb_var)
            {
                constants[constant_pos++] = -1;
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

        if (x >= nb_clause)
            return;

        results[x] = false;
        if (mask[x])
            return;

        bool conflict = cnf_matrix[x * nb_var + constant_pos] == -constant_sign;

        if (!conflict)
            return;

        int vars_in_clause = 0;
        for (auto i = x * nb_var; i < (x + 1) * nb_var; i++)
        {
            if (cnf_matrix[i])
            {
                if (!vars_in_clause)
                    vars_in_clause++;
                else
                {
                    vars_in_clause++;
                    break;
                }
            }
        }

        if (vars_in_clause == 1)
            results[x] = true;
    }

    __global__ void simplify(term_val* cnf_matrix, size_t nb_var,
                             size_t nb_clause, term_val* constants, bool* mask)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause)
            return;

        mask[x] = false;

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
