#pragma once

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

// Simple assertion macro
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "[\033[1;31mFAIL\033[0m] " << message << "\n" \
                      << "       File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// Approximate float comparison
#define ASSERT_NEAR(val1, val2, tol, message) \
    do { \
        if (std::abs((val1) - (val2)) > (tol)) { \
            std::cerr << "[\033[1;31mFAIL\033[0m] " << message << "\n" \
                      << "       Expected: " << (val2) << ", Got: " << (val1) << "\n" \
                      << "       File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

inline void print_header(const std::string& name) {
    std::cout << "[\033[1;32mRUN \033[0m] " << name << std::endl;
}

inline void print_pass() {
    std::cout << "[\033[1;32mPASS\033[0m] All checks passed." << std::endl;
}
