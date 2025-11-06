# Add the executable
add_executable(jabuff_example main.cpp)

# Link the executable against our header-only JABuff library.
# This automatically handles include directories.
target_link_libraries(jabuff_example PRIVATE JABuff)
