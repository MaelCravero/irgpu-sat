#include <fstream>
#include <iostream>

#include "cnf.cuh"

int main(int argc, char** argv)
{
    using namespace host;
    if (argc != 2)
        return 1;

    std::string path = argv[1];

    auto cnf = Cnf(std::ifstream(path));

    cnf.dump(std::cout);

    auto res = cnf.solve();

    std::cout << (res.has_value() ? "sat\n" : "unsat\n");

    return 0;
}
