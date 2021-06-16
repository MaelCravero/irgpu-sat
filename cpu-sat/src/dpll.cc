#include "dpll.hh"
#include <iostream>

namespace
{
    std::optional<Cnf::solution> aux_solve(Cnf cnf, Cnf::solution current_sol,
                                           int idx, bool val)
    {
        if (idx >= current_sol.size())
            return {};

        current_sol[idx] = val;

        // propage value for idx and val
        auto res = cnf.unit_propagation((idx + 1) * (val ? 1 : -1));

        // pure litterals
        if (!res)
            return {};

        auto sat = cnf.satisfies(current_sol);
        if (sat)
            return {current_sol};

        // We want to increment idx every two val values, so we add val which is
        // either true or false to it
        auto first = aux_solve(cnf, current_sol, idx + 1, true);

        if (first == std::nullopt)
            return aux_solve(cnf, current_sol, idx + 1, false);
        else
            return first;
    }

} // namespace

std::optional<Cnf::solution> Dpll::solve(const Cnf& cnf)
{
    Cnf::solution sol(cnf.nb_vars_);

    auto first = aux_solve(cnf, sol, 0, true);

    if (first == std::nullopt)
        return aux_solve(cnf, sol, 0, false);
    else
        return first;
}
