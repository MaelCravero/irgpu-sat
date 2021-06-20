#include "dpll.hh"

#include <algorithm>
#include <iostream>

#include "utils.hh"

/* namespace */
/* { */
/*     std::optional<Cnf::solution> aux_solve(Cnf cnf, Cnf::solution
 * current_sol, */
/*                                            int idx, bool val) */
/*     { */
/*         if (idx >= current_sol.size()) */
/*             return {}; */
/*  */
/*         current_sol[idx] = val; */
/*  */
/*         // propage value for idx and val */
/*         auto res = cnf.unit_propagation((idx + 1) * (val ? 1 : -1)); */
/*  */
/*         // pure litterals */
/*         if (!res) */
/*             return {}; */
/*  */
/*         auto sat = cnf.satisfies(current_sol); */
/*         if (sat) */
/*             return {current_sol}; */
/*  */
/*         // We want to increment idx every two val values, so we add val which
 * is */
/*         // either true or false to it */
/*         auto first = aux_solve(cnf, current_sol, idx + 1, true); */
/*  */
/*         if (first == std::nullopt) */
/*             return aux_solve(cnf, current_sol, idx + 1, false); */
/*         else */
/*             return first; */
/*     } */
/*  */
/* } // namespace */
/*  */
/* std::optional<Cnf::solution> Dpll::solve(const Cnf& cnf) */
/* { */
/*     Cnf::solution sol(cnf.nb_vars_); */
/*  */
/*     auto first = aux_solve(cnf, sol, 0, true); */
/*  */
/*     if (first == std::nullopt) */
/*         return aux_solve(cnf, sol, 0, false); */
/*     else */
/*         return first; */
/* } */

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

bool Dpll::check_conflict(const Cnf& cnf,
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

std::optional<Cnf::solution> Dpll::solve(const Cnf& cnf)
{
    std::vector<Cnf::term> assigned;

    /*     auto local = cnf; */
    /*     auto conflict = local.unit_propagation(); */
    /*  */
    /*     if (conflict) */
    /*         return {}; */

    for (;;)
    {
        /* auto local = cnf; */
        auto conflict = false;

        if (!assigned.empty())
        {
            /*             conflict = local.unit_propagation( */
            /*                 std::set<Cnf::term>{assigned.begin(),
             * assigned.end() - 1}); */
            /*  */
            /*             conflict = conflict ||
             * local.unit_propagation(assigned.back()); */

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
