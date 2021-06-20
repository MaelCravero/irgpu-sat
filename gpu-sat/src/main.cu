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
    /* cnf.dump(std::cout); */

    auto cnf_matrix = cnf.to_matrix();
    auto nb_var = cnf.nb_var_get();
    auto nb_clause = cnf.nb_clause_get();

    auto solution = dpll_solve(cnf_matrix, nb_var, nb_clause);

    if (solution.has_value())
        std::cout << "sat\n" << *solution << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
