#include <fstream>
#include <iostream>

#include "cnf.cuh"
#include "solver.cuh"

int main(int argc, char** argv)
{
    using namespace host;
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: cpu-sat file.cnf [-v1|-v2|-v3]\n";
        return 1;
    }

    std::string path = argv[1];

    auto cnf = Cnf(std::ifstream(path));

    std::optional<solution> solution;

    if (argc == 2)
        solution = solve_v3(cnf);

    else if (std::string(argv[2]) == "-v1")
        solution = solve_v1(cnf);
    else if (std::string(argv[2]) == "-v2")
        solution = solve_v2(cnf);
    else
        solution = solve_v3(cnf);

    if (solution.has_value())
        std::cout << "sat\n" << *solution << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
