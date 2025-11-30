# CMakeLists.txt - Kalman Filter Foundations
# Builds implementations of various Kalman Filters from different mathematical perspectives
cmake_minimum_required(VERSION 3.14)
project(kalman_foundations VERSION 1.0.0 LANGUAGES CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Common compile options
function(add_kalman_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target_name} PRIVATE /W4)
    endif()
endfunction()

# =============================================================================
# Demo Executables
# =============================================================================

# Linear Kalman Filter Demo (4 mathematical perspectives)
add_kalman_executable(kalman_demo src/main.cpp)

# EKF vs UKF Demo (Nonlinear pendulum tracking)
add_kalman_executable(ekf_ukf_demo src/ekf_ukf_demo.cpp)

# Quaternion EKF/UKF Demo (3D orientation estimation)
add_kalman_executable(quaternion_demo src/quaternion_demo.cpp)
