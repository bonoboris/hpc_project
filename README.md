Description
====

This repo contains the final project for the High performance computing module.
The goal of the project is to implements 2 algorithms to solve symmetric, definite, positive linear equation systems locally and on a distributed system with MPI protocol.
The 2 algorithms implemented are: the Lancszos method and the conjugate gradient method.

Prerequisite / Compatibility
====
- C++ compiler that support MPI
- an MPI implementation 
- a recentish version of CMake

The solution has been tested on PopOS/Ubuntu (20.04) and using:
- g++ 9.3.0
- cmake 3.16.3
with default installed openMPI implementation of MPI

Install
====
1. Clone this repo:
`git clone https://github.com/bonoboris/hpc_project.git [REPO_DIR]`
2. Compile
`source [REPO_DIR]/install.sh [BUILD_DIR]`\
`[BUILD_DIR]` is optional, by default the build files and executable can be found in `[REPO_DIR]/build`.
3. Execute \
With 4 processes: \
`mpirun -n 4 [BUILD_DIR]/hpc` \
With 9 processes: \
`mpirun -n 9 [BUILD_DIR]/hpc`

Output
====

The ouput contains the residual errors for each step of each implementation of the methods.

Notes
====
This project is built as a high level header lib for MPI with a single source file (hpc.cpp) solving the problem.

The project grading grid doesn't focus on code cleaness nor extended documetation as a result the code documentation is minimal and there are no test in any way.