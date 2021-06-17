#pragma once

#include <istream>
#include <optional>
#include <ostream>
#include <set>
#include <vector>

class Dpll;

class Cnf
{
public:
    using term = int;
    using clause = std::vector<term>;
    using expression = std::vector<clause>;
    using solution = std::vector<bool>;

    Cnf() = default;
    Cnf(std::istream&& input);

    /// Add a clause to the expression
    void add_clause(clause c);

    /// Append a CNF
    void append(const Cnf& other);

    bool unit_propagation(term t);

    /// Print the CNF on \a ostr using MiniSat input format
    std::ostream& dump(std::ostream& ostr) const;

    bool satisfies(const solution& s) const;

    std::optional<solution> solve(bool trace = false) const;

    friend Dpll;

private:
    /// Get the number of variables in the expression
    unsigned nb_vars_;

    /// The CNF expression
    expression expr_;
};

std::ostream& operator<<(std::ostream& o, const Cnf::solution& s);
