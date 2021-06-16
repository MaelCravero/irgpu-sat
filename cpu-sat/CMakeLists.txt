set(SRC "src/main.cc" "src/cnf.cc" "src/dpll.cc")

set(INCLUDE_DIR "src")

add_executable(cpu-sat)
target_sources(cpu-sat PRIVATE "src/main.cc" ${SRC})
target_include_directories(cpu-sat PRIVATE ${INCLUDE_DIR})

target_link_libraries(cpu-sat PRIVATE ${LIBRARIES})
