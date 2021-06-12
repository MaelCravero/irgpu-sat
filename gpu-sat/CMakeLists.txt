set(OPT_FLAGS "-O3")

set(SRC "src/main.cu" "src/cnf.cu")

set(INCLUDE_DIR "src")

set(CMAKE_CXX "clang++ -x cuda")
add_executable(gpu-sat)
target_sources(gpu-sat PRIVATE "src/main.cu" ${SRC})
target_include_directories(gpu-sat PRIVATE ${INCLUDE_DIR})

target_link_libraries(gpu-sat PRIVATE ${LIBRARIES})
