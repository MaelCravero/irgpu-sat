#include <fstream>
#include <iostream>

#include "cnf.hh"
#include "solver.hh"

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: cpu-sat [-v1|-v2|-v3] file.cnf\n";
        return 1;
    }

    std::string path = argv[argc == 2 ? 1 : 2];

    auto cnf = Cnf(std::ifstream(path));

    std::optional<Cnf::solution> res;

    if (argc == 2 || std::string(argv[1]) == "-v3")
        res = Solver::solve_v3(cnf);
    else if (std::string(argv[1]) == "-v1")
        res = Solver::solve_v1(cnf);
    else
        res = Solver::solve_v2(cnf);

    if (res.has_value())
        std::cout << "sat\n" << *res << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
