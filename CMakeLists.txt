# CMakeLists.txt - Kalman Filter Foundations
# Builds four implementations of the Kalman Filter from different mathematical perspectives
cmake_minimum_required(VERSION 3.14)
project(kalman_foundations VERSION 1.0.0 LANGUAGES CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build GUI
option(BUILD_GUI "Build the educational GUI application" ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Create the console demo executable
add_executable(kalman_demo
    src/main.cpp
)

# Link Eigen3
target_link_libraries(kalman_demo PRIVATE Eigen3::Eigen)

# Include directories for our header-only implementations
target_include_directories(kalman_demo PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Enable warnings for better code quality
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(kalman_demo PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(kalman_demo PRIVATE /W4)
endif()

# ==============================================================================
# GUI Application (Optional)
# ==============================================================================
if(BUILD_GUI)
    # Find OpenGL and GLFW
    find_package(OpenGL REQUIRED)
    find_package(glfw3 3.3 QUIET)
    
    if(NOT glfw3_FOUND)
        # Try finding GLFW via pkg-config
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(GLFW3 QUIET glfw3)
        endif()
    endif()
    
    if(glfw3_FOUND OR GLFW3_FOUND)
        message(STATUS "Building GUI application")
        
        # ImGui source files
        set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/extern/imgui)
        set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/extern/implot)
        
        set(IMGUI_SOURCES
            ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_draw.cpp
            ${IMGUI_DIR}/imgui_widgets.cpp
            ${IMGUI_DIR}/imgui_tables.cpp
            ${IMGUI_DIR}/imgui_impl_glfw.cpp
            ${IMGUI_DIR}/imgui_impl_opengl3.cpp
        )
        
        set(IMPLOT_SOURCES
            ${IMPLOT_DIR}/implot.cpp
            ${IMPLOT_DIR}/implot_items.cpp
        )
        
        # Create ImGui library
        add_library(imgui STATIC ${IMGUI_SOURCES})
        target_include_directories(imgui PUBLIC ${IMGUI_DIR})
        target_link_libraries(imgui PUBLIC OpenGL::GL)
        if(glfw3_FOUND)
            target_link_libraries(imgui PUBLIC glfw)
        else()
            target_include_directories(imgui PUBLIC ${GLFW3_INCLUDE_DIRS})
            target_link_libraries(imgui PUBLIC ${GLFW3_LIBRARIES})
        endif()
        
        # Create ImPlot library
        add_library(implot STATIC ${IMPLOT_SOURCES})
        target_include_directories(implot PUBLIC ${IMPLOT_DIR})
        target_link_libraries(implot PUBLIC imgui)
        
        # Create the GUI executable
        add_executable(kalman_gui
            src/gui/main_gui.cpp
        )
        
        target_include_directories(kalman_gui PRIVATE 
            ${CMAKE_SOURCE_DIR}/src
            ${IMGUI_DIR}
            ${IMPLOT_DIR}
        )
        
        target_link_libraries(kalman_gui PRIVATE 
            Eigen3::Eigen
            imgui
            implot
        )
        
        # Enable warnings for GUI
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(kalman_gui PRIVATE -Wall -Wextra)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(kalman_gui PRIVATE /W4)
        endif()
        
        message(STATUS "GUI application will be built as 'kalman_gui'")
    else()
        message(WARNING "GLFW not found. GUI application will not be built.")
        message(STATUS "Install GLFW with: sudo apt-get install libglfw3-dev")
    endif()
endif()
