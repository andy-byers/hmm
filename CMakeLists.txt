cmake_minimum_required(VERSION 3.22)
project(hmm)

set(CMAKE_CXX_STANDARD 20)

set(HMM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(HMM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hmm")
set(HMM_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(HMM_SOURCES src/hmm.cpp include/hmm/hmm.h)
set(HMM_WARNINGS -Wall -Wconversion -Wsign-conversion -Werror -Wextra -Wpedantic -Wpointer-arith)

add_library(hmm
    STATIC "${HMM_SOURCE_DIR}/hmm.cpp"
           "${HMM_INCLUDE_DIR}/hmm.h")
target_include_directories(hmm
    PUBLIC  $<BUILD_INTERFACE:${HMM_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(hmm
    PRIVATE "${HMM_WARNINGS}")

add_executable(test
    "${HMM_TEST_DIR}/test.cpp")
target_compile_options(test
    PRIVATE
    "${HMM_WARNINGS}")
target_link_libraries(test
    PRIVATE hmm)
