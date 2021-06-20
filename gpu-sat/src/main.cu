#include <fstream>
#include <iostream>

#include "cnf.cuh"
#include "dpll.cuh"

int main(int argc, char** argv)
{
    using namespace host;
    if (argc != 2)
        return 1;

    std::string path = argv[1];

    auto cnf = Cnf(std::ifstream(path));

    auto solution = dpll_solve(cnf);

    if (solution.has_value())
        std::cout << "sat\n" << *solution << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
