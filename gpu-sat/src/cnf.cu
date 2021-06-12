#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

    std::optional<solution> Cnf::solve() const
    {
        std::size_t max_val = std::pow(2, nb_vars_);

        std::vector<int> vec;
        for (int i = 0; i < 1024 * 1024; i++)
            vec.push_back(i);

        /* int* dev_buffer = host::utils::malloc(vec); */
        /* utils::memcpy(dev_buffer, vec); */

        auto dev_buffer = utils::init_from(vec);

        device::inc<<<1024, 1024>>>(dev_buffer);

        utils::memcpy(vec, dev_buffer);

        for (int i = 0; i < 20; i++)
            std::cout << vec[i];

        return {};
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
    __global__ void inc(int* v)
    {
        v[x_idx()]++;
    }
} // namespace device
