# IRGPU SAT Solver

C++/CUDA SAT Solver implemented for the IRGPU class.

To build:
```
cmake -B build
make -C build
```

To run (CPU):
```
build/cpu-sat/cpu-sat -v1 <test.cnf> # brute force
build/cpu-sat/cpu-sat -v2 <test.cnf> # dpll
build/cpu-sat/cpu-sat -v3 <test.cnf> # dpll with read only algorithm, no CNF copy
```

To run (GPU):
```
build/gpu-sat/gpu-sat -v1          <test.cnf> # brute force
build/gpu-sat/gpu-sat -v2          <test.cnf> # dpll
build/gpu-sat/gpu-sat -v2-no-pitch <test.cnf> # dpll with 1D arrays for CNF
build/gpu-sat/gpu-sat -v3          <test.cnf> # dpll with read only algorithm, no CNF copy
build/gpu-sat/gpu-sat -v3-no-pitch <test.cnf> # dpll with 1D arrays for CNF and read only algorithm, no CNF copy
```
