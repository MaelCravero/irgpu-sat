#include <fstream>
#include <iostream>

#include "cnf.hh"
#include "solver.hh"

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: cpu-sat file.cnf [-v1|-v2|-v3]\n";
        return 1;
    }

    std::string path = argv[1];

    auto cnf = Cnf(std::ifstream(path));

    std::optional<Cnf::solution> res;

    if (argc == 2)
        res = Solver::solve_v3(cnf);
    else if (std::string(argv[2]) == "-v1")
        res = Solver::solve_v1(cnf);
    else if (std::string(argv[2]) == "-v2")
        res = Solver::solve_v2(cnf);
    else
        res = Solver::solve_v3(cnf);

    if (res.has_value())
        std::cout << "sat\n" << *res << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
