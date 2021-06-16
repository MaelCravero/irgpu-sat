#pragma once

#include <optional>

#include "cnf.hh"

struct Dpll
{
    static std::optional<Cnf::solution> solve(const Cnf& cnf);
};
