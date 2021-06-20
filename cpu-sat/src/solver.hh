#pragma once

#include <optional>

#include "cnf.hh"

struct Solver
{
    // Brut force version
    static std::optional<Cnf::solution> solve_v1(const Cnf& cnf);

    // First dpll version, using copies
    static std::optional<Cnf::solution> solve_v2(const Cnf& cnf);

    // Second dpll version without copies
    static std::optional<Cnf::solution> solve_v3(const Cnf& cnf);

    static bool check_conflict(const Cnf& cnf,
                               const std::vector<Cnf::term>& assigned,
                               Cnf::term last);
};
