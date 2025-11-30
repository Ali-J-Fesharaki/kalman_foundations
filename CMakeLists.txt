# CMakeLists.txt - Kalman Filter Foundations
# Builds four implementations of the Kalman Filter from different mathematical perspectives
cmake_minimum_required(VERSION 3.14)
project(kalman_foundations VERSION 1.0.0 LANGUAGES CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Qt automoc/autorcc/autouic
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Qt6 components for pure Qt GUI
find_package(Qt6 REQUIRED COMPONENTS Widgets Charts)

# Create the Qt GUI executable
add_executable(kalman_demo
    src/main.cpp
)

# Link Eigen3 and Qt6
target_link_libraries(kalman_demo PRIVATE
    Eigen3::Eigen
    Qt6::Widgets
    Qt6::Charts
)

# Include directories for our header-only implementations
target_include_directories(kalman_demo PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Enable warnings for better code quality
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(kalman_demo PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(kalman_demo PRIVATE /W4)
endif()
