cmake_minimum_required(VERSION 3.11)

project(
    hpc 
    VERSION 1.0.0
    DESCRIPTION "A small parallel computing project using MPI"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    
add_executable(
    ${PROJECT_NAME}
    include/constants.hpp
    include/linalg.hpp
    include/Matrix.hpp
    include/MpiRoutines.hpp
    include/MpiWrapper.hpp
    include/Params.hpp
    include/utils.hpp
    src/HPC.cpp
    )

target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
target_include_directories(${PROJECT_NAME} PUBLIC include)
