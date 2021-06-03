#include "cnf.hh"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>

Cnf::Cnf(std::istream&& input)
    : Cnf()
{
    std::string line;
    std::getline(input, line); // skip dimacs header

    unsigned nb_clauses;
    std::sscanf(line.c_str(), "p cnf %u %u", &nb_vars_, &nb_clauses);

    std::size_t pos; // Track the positions for recovering specific clauses
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
    expr_.emplace_back(c);
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

namespace
{
    bool term_matches(Cnf::term t, const Cnf::solution& s)
    {
        auto sign = t > 0;
        auto pos = std::abs(t) - 1;

        return sign == s[pos];
    }

} // namespace

bool Cnf::satisfies(const solution& s) const
{
    for (const auto& clause : expr_)
    {
        if (clause.empty())
            continue;

        bool clause_sat = false;
        for (const auto& term : clause)
            if (term_matches(term, s))
            {
                clause_sat = true;
                break;
            }

        if (clause_sat)
            continue;

        return false;
    }

    return true;
}

std::optional<Cnf::solution> Cnf::solve(bool trace) const
{
    using std::views::iota;

    for (auto n : iota(0u, std::pow(2, nb_vars_)))
    {
        solution s;
        for (auto i : iota(0u, nb_vars_))
        {
            s.push_back((n >> i) % 2);
        }

        if (trace)
            std::cerr << "testing " << s << std::endl;

        if (satisfies(s))
            return s;
    }

    return {};
}

std::ostream& operator<<(std::ostream& o, const Cnf::solution& s)
{
    auto it = s.begin();
    o << *it++;

    for (; it != s.end(); it++)
        o << ", " << *it;

    return o << "\n";
}
