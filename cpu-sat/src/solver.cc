#include "solver.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>

#include "utils.hh"

namespace
{
    void assign_decision_litteral(std::vector<Cnf::term>& assigned)
    {
        assigned.push_back(assigned.size() + 1);
    }

    void backjump(std::vector<Cnf::term>& assigned)
    {
        while (!assigned.empty() && assigned.back() < 0)
            assigned.pop_back();

        if (!assigned.empty())
            assigned.back() *= -1;
    }

    Cnf::solution calculate_solution(std::vector<Cnf::term>& assigned)
    {
        Cnf::solution sol(assigned.size());

        for (auto term : assigned)
        {
            auto sign = term > 0 ? 1 : -1;
            sol[term * sign - 1] = sign == 1;
        }

        return sol;
    }

} // namespace

bool Solver::check_conflict(const Cnf& cnf,
                          const std::vector<Cnf::term>& assigned,
                          Cnf::term last)
{
    for (auto clause : cnf.expr_)
    {
        if (!utils::contains(clause, -last))
            continue;

        int vars_in_clause = 0;
        bool clause_is_true = false;
        for (auto term : clause)
        {
            if (utils::contains(assigned, term))
            {
                clause_is_true = true;
                break; // The clause is true
            }

            if (!utils::contains(assigned, -term))
                vars_in_clause++;
        }

        if (vars_in_clause == 1 && !clause_is_true)
            return true;
    }

    return false;
}

std::optional<Cnf::solution> Solver::solve_v1(const Cnf& cnf)
{
    using std::views::iota;

    for (auto n : iota(0u, std::pow(2, cnf.nb_vars_)))
    {
        Cnf::solution s;
        for (auto i : iota(0u, cnf.nb_vars_))
        {
            s.push_back((n >> i) % 2);
        }

        if (cnf.satisfies(s))
            return s;
    }

    return {};
}

std::optional<Cnf::solution> Solver::solve_v2(const Cnf& cnf)
{
    std::vector<Cnf::term> assigned;

    for (;;)
    {
        auto local = cnf;
        auto conflict = false;

        if (!assigned.empty())
        {
            conflict = local.unit_propagation(
                    std::set<Cnf::term>{assigned.begin(),
                    assigned.end() - 1});

            conflict = conflict || local.unit_propagation(assigned.back());
        }

        if (conflict)
        {
            backjump(assigned);

            if (assigned.empty())
                return {};
        }
        else if (assigned.size() < cnf.nb_vars_)
        {
            assign_decision_litteral(assigned);
        }
        else
            break;
    }

    return {calculate_solution(assigned)};
}

std::optional<Cnf::solution> Solver::solve_v3(const Cnf& cnf)
{
    std::vector<Cnf::term> assigned;

    for (;;)
    {
        auto conflict = false;

        if (!assigned.empty())
        {
            auto last = assigned.back();
            assigned.pop_back();
            conflict = check_conflict(cnf, assigned, last);
            assigned.emplace_back(last);
        }

        if (conflict)
        {
            backjump(assigned);

            if (assigned.empty())
                return {};
        }
        else if (assigned.size() < cnf.nb_vars_)
        {
            assign_decision_litteral(assigned);
        }
        else
            break;
    }

    return {calculate_solution(assigned)};
}
