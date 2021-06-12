#pragma once

#include <istream>
#include <optional>
#include <ostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using term = int;

namespace host
{
    using clause = thrust::host_vector<term>;
    using expression = thrust::host_vector<clause>;
    using solution = thrust::host_vector<bool>;

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

        std::optional<solution> solve() const;

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
    using clause = thrust::device_vector<term>;
    using expression = thrust::device_vector<clause>;
    using solution = thrust::device_vector<bool>;

    __global__ void inc(int* v);
} // namespace device
