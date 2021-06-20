#pragma once

#include <optional>

#include "cnf.hh"

struct Dpll
{
    static std::optional<Cnf::solution> solve(const Cnf& cnf);
    static bool check_conflict(const Cnf& cnf,
                               const std::vector<Cnf::term>& assigned,
                               Cnf::term last);
};
