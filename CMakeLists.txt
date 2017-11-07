cmake_minimum_required(VERSION 3.7)
project(simple_cnn)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)

add_executable(example1 "Example MNIST/example1.cpp")
add_executable(example2 "Example MNIST/example2.cpp")
