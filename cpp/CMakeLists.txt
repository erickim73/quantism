cmake_minimum_required(VERSION 3.15)
project(quantsim_matching_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(quantsim_matching_engine src/order_book.cpp)
install(TARGETS quantsim_matching_engine LIBRARY DESTINATION .)
