cmake_minimum_required(VERSION 3.7)
project(simple_cnn)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)

add_executable(simple "examples/simple.cpp")
add_executable(alexnet "examples/alexnet.cpp")
