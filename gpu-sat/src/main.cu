#include <fstream>
#include <iostream>

#include "cnf.cuh"
#include "solver.cuh"

int main(int argc, char** argv)
{
    using namespace host;
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: cpu-sat [-v1|-v2|-v3] file.cnf\n";
        return 1;
    }

    std::string path = argv[argc == 2 ? 1 : 2];

    auto cnf = Cnf(std::ifstream(path));

    std::optional<solution> res;

    if (argc == 2 || std::string(argv[1]) == "-v3")
        res = solve_v3(cnf);
    else if (std::string(argv[1]) == "-v2-no-pitch")
        res = solve_v2_no_pitch(cnf);
    else if (std::string(argv[1]) == "-v3-no-pitch")
        res = solve_v3_no_pitch(cnf);
    else if (std::string(argv[1]) == "-v1")
        res = solve_v1(cnf);
    else
        res = solve_v2(cnf);

    if (res.has_value())
        std::cout << "sat\n" << *res << "\n";
    else
        std::cout << "unsat\n";

    return 0;
}
