#include <fstream>
#include <iostream>

#include "cnf.hh"
#include "dpll.hh"

int main(int argc, char** argv)
{
    if (argc != 2)
        return 1;

    std::string path = argv[1];

    auto cnf = Cnf(std::ifstream(path));

    // cnf.dump(std::cout);

    // std::cout << cnf.satisfies({true, true, true}) << std::endl;
    // std::cout << cnf.satisfies({false, true, true}) << std::endl;

    /*     cnf.unit_propagation(); */
    /*  */
    /*     cnf.dump(std::cout); */

    auto res = Dpll::solve(cnf);

    std::cout << (res.has_value() ? "sat\n" : "unsat\n");

    return 0;
}
