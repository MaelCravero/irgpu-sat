set(SRC "src/main.cu" "src/cnf.cu" "src/dpll.cu")

set(INCLUDE_DIR "src")

add_executable(gpu-sat)
target_sources(gpu-sat PRIVATE "src/main.cu" ${SRC})
target_include_directories(gpu-sat PRIVATE ${INCLUDE_DIR})

target_link_libraries(gpu-sat PRIVATE ${LIBRARIES})
