set(OPT_FLAGS "-O3")

set(SRC "src/main.cc" "src/cnf.cc")

set(INCLUDE_DIR "src")

cmake_minimum_required(VERSION 3.12)
project("Simple Sat")

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -fno-inline")

add_executable(simple-sat)
target_sources(simple-sat PRIVATE "src/main.cc" ${SRC})
target_include_directories(simple-sat PRIVATE ${INCLUDE_DIR})

target_link_libraries(simple-sat PRIVATE ${LIBRARIES})
