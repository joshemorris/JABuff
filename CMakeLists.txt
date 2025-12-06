# CMake minimum version
cmake_minimum_required(VERSION 3.12)

# Project Name and Version
project(JABuff VERSION 1.0.0)

# Set C++ Standard (C++17 is good for <atomic>)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- JABuff Library Definition ---
# We define JABuff as an INTERFACE library since it's header-only.
# This means it doesn't get compiled on its own, but consumers
# who link to it will automatically get its include paths.
add_library(JABuff INTERFACE)

# Add the 'include' directory to JABuff's public interface.
# Any target linking against JABuff will inherit this include directory.
target_include_directories(JABuff INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> # Path when installed
)

# --- Example Executable ---
# We add the 'src' directory, which contains our example executable.
# This is a good way to test and demonstrate the library.
add_subdirectory(src)


# --- Testing ---
enable_testing()
add_subdirectory(tests)
