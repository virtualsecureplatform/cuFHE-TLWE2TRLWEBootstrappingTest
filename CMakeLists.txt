cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

project(tlwe2trlwegb LANGUAGES CXX)

add_subdirectory(TFHEpp)
enable_language(CUDA)
find_package(CUDA)

add_subdirectory(cuFHE/cufhe)

#add_executable(tlwe2trlwe testcuFHETRLWElvl.cpp)
#add_executable(tlwe2trlwe gbtest.cpp)
add_executable(tlwe2trlwe SEandKStest.cpp)

target_link_libraries(tlwe2trlwe tfhe++ cufhe_gpu)
target_include_directories(tlwe2trlwe PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

