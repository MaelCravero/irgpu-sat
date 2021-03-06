#include "cnf.hh"

#include <algorithm>
#include <bitset>
#include <iostream>
#include <sstream>
#include <string>

#include "utils.hh"

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
    expr_.emplace_back(c);
}

void Cnf::append(const Cnf& other)
{
    for (auto clause : other.expr_)
        expr_.push_back(clause);
}

bool Cnf::unit_propagation(std::set<term> terms)
{
    for (auto clause : expr_)
        if (clause.size() == 1 && terms.contains(-clause[0]))
            return true;

    utils::erase_if(expr_, [&](auto clause) {
        for (auto t : clause)
        {
            if (terms.contains(t))
                return true;
        }
        return false;
    });

    for (auto& clause : expr_)
        utils::erase_if(clause, [&](term elt) { return terms.contains(-elt);});

    return false;
}

bool Cnf::unit_propagation(term t)
{
    for (auto clause : expr_)
        if (clause.size() == 1 && clause[0] == -t)
            return true;

    utils::erase_if(expr_, [&](auto clause) {
        return std::find(clause.begin(), clause.end(), t) != clause.end();
    });

    for (auto& clause : expr_)
        utils::erase_if(clause, [&](term elt) { return elt == -t; });

    return false;
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


std::ostream& operator<<(std::ostream& o, const Cnf::solution& s)
{
    int i = 0;
    o << (s[i++] ? 1 : -1);

    for (; i != s.size(); i++)
        o << ", " << (i + 1) * (s[i] ? 1 : -1);

    return o << "\n";
}
