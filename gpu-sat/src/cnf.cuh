#pragma once

#include <istream>
#include <optional>
#include <ostream>
#include <vector>

using term = int;
using term_val = char;

namespace host
{
    using clause = std::vector<term>;
    using expression = std::vector<clause>;
    using solution = std::vector<int>;

    class Cnf
    {
    public:
        Cnf() = default;
        Cnf(std::istream&& input);

        /// Add a clause to the expression
        void add_clause(clause c);

        /// Append a CNF
        void append(const Cnf& other);

        /// Print the CNF on \a ostr using MiniSat input format
        std::ostream& dump(std::ostream& ostr) const;

        std::vector<term> flatten() const;

        void remove_trivial_clauses();
        term_val* to_matrix() const;

        unsigned nb_var_get() const;
        unsigned nb_clause_get() const;

    private:
        /// Get the number of variables in the expression
        unsigned nb_vars_;

        /// The CNF expression
        expression expr_;
    };

    std::ostream& operator<<(std::ostream& o, const solution& s);

} // namespace host

namespace device
{
    __global__ void satisfies(term* cnf, size_t cnf_size, char* is_solution,
                              size_t nb_var);

    __device__ term* compute_solution(size_t index, size_t nb_var);

} // namespace device
