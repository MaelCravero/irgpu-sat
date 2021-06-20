#include <iostream>
#include <optional>
#include <vector>

#include "box.cuh"
#include "cnf.cuh"
#include "solver.cuh"
#include "utils.cuh"

namespace host
{
    namespace
    {
        void backjump(std::vector<term_val>& constants, size_t& size)
        {
            while (size >= 1 && constants[size - 1] < 0)
            {
                constants[size - 1] = 0;
                size--;
            }

            if (size)
                constants[size - 1] *= -1;
        }

        solution calculate_solution(std::vector<term_val>& constants)
        {
            solution sol;

            int i = 1;
            for (auto sign : constants)
                sol.emplace_back(i++ * sign);

            return sol;
        }

    } // namespace

    std::optional<solution> solve_v1(Cnf cnf)
    {
        std::size_t max_val = std::pow(2, cnf.nb_var_get());

        auto flat_cnf = cnf.flatten();

        auto cnf_dev = Box(utils::init_from(flat_cnf));
        auto res_host = std::vector<char>(max_val);
        auto res_dev = Box(utils::malloc(res_host));

        int block_size = 1024;
        int num_block = (res_host.size() + block_size - 1) / block_size;

        device::satisfies<<<num_block, block_size>>>(
            cnf_dev.get(), flat_cnf.size(), res_dev.get(),cnf.nb_var_get());
        utils::memcpy(res_host, res_dev.get());

        for (size_t pos = 0; pos < res_host.size(); pos++)
        {
            if (res_host[pos])
            {
                solution sol;
                for (int i = 0; i < cnf.nb_var_get(); i++)
                {
                    sol.push_back((pos >> i) % 2);
                }
                std::cout << sol;
                return sol;
            }
        }

        return {};
    }

    std::optional<solution> solve_v2(Cnf cnf)
    {
        cnf.remove_trivial_clauses();

        auto cnf_matrix = cnf.to_matrix();
        auto nb_var = cnf.nb_var_get();
        auto nb_clause = cnf.nb_clause_get();

        int nb_blocks = (nb_clause / 1024) + 1;

        int bsize = 32;
        dim3 dimGrid(nb_clause / bsize + 1, nb_var / bsize + 1);
        dim3 dimBlock(bsize, bsize);

        // This is the result vector which contains assigned variables.
        std::vector<term_val> constants(nb_var);

        // We copy the CNF on the device to lessen copy costs.
        size_t dev_cnf_pitch;
        auto dev_cnf = Box(
            utils::mallocPitch<term_val>(&dev_cnf_pitch, nb_var, nb_clause));
        utils::memcpy2D(dev_cnf.get(), dev_cnf_pitch, cnf_matrix,
                        nb_var * sizeof(term_val), nb_var, nb_clause,
                        cudaMemcpyHostToDevice);

        free(cnf_matrix);
        cnf_matrix = NULL;

        // This is the CNF used in the loop, which is recalculated.
        size_t local_cnf_pitch;
        auto local_cnf = Box(
            utils::mallocPitch<term_val>(&local_cnf_pitch, nb_var, nb_clause));

        size_t clause_size = nb_clause * sizeof(bool);

        // mask is used to determine if a clause is pruned.
        auto mask = Box(utils::malloc<bool>(nb_clause));

        // results is used to determine if a clause has a conflict.
        auto results = Box(utils::malloc<bool>(nb_clause));

        // We cannot we used a std::vector<bool> due to bitset specialization
        // which disables the .data() method...
        auto host_res = (bool*)malloc(clause_size);

        // Intermediary vector for passing constants to the device.
        auto dev_constants = Box(utils::malloc<term_val>(nb_var));

        size_t constant_size = 0;

        // Set the first constant to true
        constants[constant_size++] = 1;

        for (;;)
        {
            utils::memcpy2D(local_cnf.get(), local_cnf_pitch, dev_cnf.get(),
                            dev_cnf_pitch, nb_var, nb_clause,
                            cudaMemcpyDeviceToDevice);

            // Backup last assigned constant
            auto cur_constant = constants[constant_size - 1];
            constants[constant_size - 1] = 0;

            utils::memcpy(dev_constants.get(), constants);

            constants[constant_size - 1] = cur_constant;

            device::simplify<<<nb_blocks, 1024>>>(local_cnf, local_cnf_pitch,
                                                  nb_var, nb_clause,
                                                  dev_constants, mask);

            device::remove_terms<<<dimGrid, dimBlock>>>(
                local_cnf, local_cnf_pitch, nb_var, nb_clause, dev_constants,
                mask);

            device::check_conflict<<<nb_blocks, 1024>>>(
                local_cnf, local_cnf_pitch, nb_var, nb_clause,
                constant_size - 1, constants[constant_size - 1], results,
                mask);

            utils::memcpy(host_res, results.get(), clause_size,
                          cudaMemcpyDeviceToHost);

            bool conflict = false;
            for (auto i = 0; i < nb_clause && !conflict; i++)
                if (host_res[i])
                    conflict = true;

            if (conflict)
            {
                backjump(constants, constant_size);

                if (!constant_size)
                {
                    free(host_res);
                    return {};
                }
            }
            else if (constant_size < nb_var)
            {
                constants[constant_size++] = 1;
            }
            else
                break;
        }

        free(host_res);

        return {calculate_solution(constants)};
    }


    std::optional<solution> solve_v3(Cnf cnf)
    {
        cnf.remove_trivial_clauses();

        auto cnf_matrix = cnf.to_matrix();
        auto nb_var = cnf.nb_var_get();
        auto nb_clause = cnf.nb_clause_get();

        int nb_blocks = (nb_clause / 1024) + 1;

        int bsize = 32;
        dim3 dimGrid(nb_clause / bsize + 1, nb_var / bsize + 1);
        dim3 dimBlock(bsize, bsize);

        // This is the result vector which contains assigned variables.
        std::vector<term_val> constants(nb_var);

        // We copy the CNF on the device to lessen copy costs.
        size_t dev_cnf_pitch;
        auto dev_cnf = Box(
            utils::mallocPitch<term_val>(&dev_cnf_pitch, nb_var, nb_clause));
        utils::memcpy2D(dev_cnf.get(), dev_cnf_pitch, cnf_matrix,
                        nb_var * sizeof(term_val), nb_var, nb_clause,
                        cudaMemcpyHostToDevice);

        free(cnf_matrix);
        cnf_matrix = NULL;

        // This is the CNF used in the loop, which is recalculated.
        size_t local_cnf_pitch;
        auto local_cnf = Box(
            utils::mallocPitch<term_val>(&local_cnf_pitch, nb_var, nb_clause));

        size_t clause_size = nb_clause * sizeof(bool);

        // mask is used to determine is a clause is pruned.
        auto mask = Box(utils::malloc<bool>(nb_clause));

        // results is used to determine is a clause has a conflict.
        auto results = Box(utils::malloc<bool>(nb_clause));

        // We cannot we used a std::vector<bool> due to bitset specialization
        // which disables the .data() method...
        auto host_res = (bool*)malloc(clause_size);

        // Intermediary vector for passing constants to the device.
        auto dev_constants = Box(utils::malloc<term_val>(nb_var));

        size_t constant_size = 0;

        // Set the first constant to true
        constants[constant_size++] = 1;

        for (;;)
        {
            // Backup last assigned constant
            auto cur_constant = constants[constant_size - 1];
            constants[constant_size - 1] = 0;

            utils::memcpy(dev_constants.get(), constants);

            constants[constant_size - 1] = cur_constant;

            device::the_one_true_kernel<<<nb_blocks, 1024>>>(
                dev_cnf, dev_cnf_pitch, nb_var, nb_clause, dev_constants,
                constant_size - 1, constants[constant_size - 1], results);

            utils::memcpy(host_res, results.get(), clause_size,
                          cudaMemcpyDeviceToHost);

            bool conflict = false;
            for (auto i = 0; i < nb_clause && !conflict; i++)
                if (host_res[i])
                    conflict = true;

            if (conflict)
            {
                backjump(constants, constant_size);

                if (!constant_size)
                {
                    free(host_res);
                    return {};
                }
            }
            else if (constant_size < nb_var)
            {
                constants[constant_size++] = 1;
            }
            else
                break;
        }

        free(host_res);

        return {calculate_solution(constants)};
    }

} // namespace host

namespace device
{
    __global__ void the_one_true_kernel(term_val* cnf_matrix, size_t pitch,
                                        size_t nb_var, size_t nb_clause,
                                        term_val* constants,
                                        size_t constant_pos,
                                        term_val constant_sign, bool* results)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause)
            return;

        results[x] = false;

        // There can only be a conflict if the clause contains the negation of
        // the term.
        bool conflict = cnf_matrix[x * pitch + constant_pos] == -constant_sign;

        if (!conflict)
            return;

        // If there is a possible conflict, we have to check if the clause is
        // unitary. If it is, there is indeed a conflict.

        int vars_in_clause = 0;
        for (auto i = 0; i < nb_var; i++)
        {
            auto pos = x * pitch + i;

            if (!cnf_matrix[pos])
                continue;

            if (cnf_matrix[pos] == constants[i])
                return;

            if (constants[i])
                continue;

            if (!vars_in_clause)
                vars_in_clause++;
            else
            {
                vars_in_clause++;
                break;
            }
        }

        if (vars_in_clause == 1)
            results[x] = true;
    }

    __global__ void check_conflict(term_val* cnf_matrix, size_t pitch,
                                   size_t nb_var, size_t nb_clause,
                                   size_t constant_pos, term_val constant_sign,
                                   bool* results, bool* mask)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause)
            return;

        results[x] = false;

        if (mask[x])
            return;

        // There can only be a conflict if the clause contains the negation of
        // the term.
        bool conflict = cnf_matrix[x * pitch + constant_pos] == -constant_sign;

        if (!conflict)
            return;

        // If there is a possible conflict, we have to check if the clause is
        // unitary. If it is, there is indeed a conflict.

        int vars_in_clause = 0;
        for (auto i = 0; i < nb_var; i++)
        {
            if (cnf_matrix[x * pitch + i])
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

    __global__ void simplify(term_val* cnf_matrix, size_t pitch, size_t nb_var,
                             size_t nb_clause, term_val* constants, bool* mask)
    {
        auto x = utils::x_idx();

        if (x >= nb_clause)
            return;

        mask[x] = false;

        for (auto i = 0; i < nb_var; i++)
        {
            auto pos = x * pitch + i;

            // If the term is the same, we can remove the whole clause.
            if (cnf_matrix[pos] && cnf_matrix[pos] == constants[i])
            {
                mask[x] = true; // The clause is true
                return;
            }
        }
    }

    __global__ void remove_terms(term_val* cnf_matrix, size_t pitch,
                                 size_t nb_var, size_t nb_clause,
                                 term_val* constants, bool* mask)
    {
        auto x = utils::x_idx();
        auto y = utils::y_idx();

        if (x >= nb_clause || y >= nb_var)
            return;

        if (mask[x])
            return;

        auto pos = x * pitch + y;

        // If the term is the negation, we can remove it from the clause.
        if (cnf_matrix[pos] && cnf_matrix[pos] == -constants[y])
            cnf_matrix[pos] = 0;
    }

} // namespace device
