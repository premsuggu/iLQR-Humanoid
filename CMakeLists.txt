cmake_minimum_required(VERSION 3.10)
project(pendulum_mpc)           # mpc for pendulum for now

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
pkg_check_modules(CASADI REQUIRED casadi)
# pkg_check_modules(PINOCCHIO REQUIRED pinocchio)


include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${CASADI_INCLUDE_DIRS}
    # ${PINOCCHIO_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)

link_directories(
    ${CASADI_LIBRARY_DIRS}
    # ${PINOCCHIO_LIBRARY_DIRS}
)

add_executable(mpc
    # main.cpp
    # robot_model.cpp
    # mpc_utils.cpp
    mpc.cpp
    ilqr.cpp
)

target_link_libraries(mpc
    ${CASADI_LIBRARIES}
    # ${PINOCCHIO_LIBRARIES}
)
