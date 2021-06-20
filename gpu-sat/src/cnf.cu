#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "box.cuh"
#include "cnf.cuh"
#include "utils.cuh"

namespace host
{
    Cnf::Cnf(std::istream&& input)
        : Cnf()
    {
        std::string line;
        std::getline(input, line); // skip dimacs header

        unsigned nb_clauses;
        std::sscanf(line.c_str(), "p cnf %u %u", &nb_vars_, &nb_clauses);

        while (std::getline(input, line))
        {
            clause c;
            term t;
            std::istringstream stream(line);

            while (stream >> t && t)
                c.push_back(t);

            expr_.push_back(c);
        }
    }

    void Cnf::add_clause(clause c)
    {
        expr_.push_back(c);
    }

    void Cnf::append(const Cnf& other)
    {
        for (auto clause : other.expr_)
            expr_.push_back(clause);
    }

    std::ostream& Cnf::dump(std::ostream& ostr) const
    {
        auto clauses = expr_.size();

        ostr << "p cnf " << nb_vars_ << " " << clauses << "\n";

        for (const auto& clause : expr_)
        {
            for (auto term : clause)
                ostr << term << " ";

            ostr << "0\n"; // 0 terminates a clause
        }

        return ostr;
    }

    std::vector<term> Cnf::flatten() const
    {
        std::vector<term> res;

        for (const auto& clause : expr_)
        {
            for (const auto term : clause)
                res.push_back(term);

            res.push_back(0);
        }

        return res;
    }

    void Cnf::remove_trivial_clauses()
    {
        for (term term = 0; term < nb_vars_; term++)
            for (auto it = expr_.begin(); it != expr_.end(); it++)
            {
                if (std::find(it->begin(), it->end(), term) != it->end()
                    && std::find(it->begin(), it->end(), -term) != it->end())
                    it = expr_.erase(it);
            }
    }

    term_val* Cnf::to_matrix() const
    {
        // TODO:do not handle oppose it terms

        auto matrix = new char[nb_vars_ * expr_.size()]{}; // set to 0 ?
        for (int i = 0; i < expr_.size(); i++)
        {
            for (auto t : expr_[i])
            {
                if (t > 0)
                    matrix[i * nb_vars_ + std::abs(t) - 1] = 1;
                else
                    matrix[i * nb_vars_ + std::abs(t) - 1] = -1;
            }
        }

        return matrix;
    }

    unsigned Cnf::nb_var_get() const
    {
        return nb_vars_;
    }

    unsigned Cnf::nb_clause_get() const
    {
        return expr_.size();
    }

    std::ostream& operator<<(std::ostream& o, const solution& s)
    {
        auto it = s.begin();
        o << *it++;

        for (; it != s.end(); it++)
            o << ", " << *it;

        return o << "\n";
    }
} // namespace host

namespace device
{
    namespace
    {
        __device__ bool term_matches(term t, const term* s)
        {
            auto sign = t > 0;
            auto pos = abs(t) - 1;

            return sign == (s[pos] != 0);
        }

    } // namespace

    __global__ void satisfies(term* cnf, size_t cnf_size, char* is_solution,
                              size_t nb_var)
    {
        auto idx = utils::x_idx();
        if (idx >= pow(2, nb_var))
            return;

        auto sol = compute_solution(idx, nb_var);
        size_t i = 0;
        while (i < cnf_size)
        {
            if (cnf[i] == 0)
            {
                i++;
                continue;
            }

            bool clause_sat = false;
            for (; cnf[i] != 0; i++)
                if (term_matches(cnf[i], sol))
                {
                    clause_sat = true;
                }

            if (clause_sat)
                continue;

            is_solution[idx] = 0;
            return;
        }

        is_solution[idx] = 1;
        return;
    }

    __device__ term* compute_solution(size_t index, size_t nb_var)
    {
        term* sol = (term*)malloc(nb_var * sizeof(term));

        for (int i = 0; i < nb_var; i++)
        {
            sol[i] = (index >> i) % 2;
        }

        return sol;
    }
} // namespace device
