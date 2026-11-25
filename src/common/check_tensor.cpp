#include "project.h"
#include <cmath>
#include <iostream>


// Returns 1 (pass) if the max abs diff between actual/expected is within
// tol, else 0 (fail). Always prints a one-line summary -- use this from
// every kernel test harness so output format is consistent across the
// project.
int check_tensor(const float* actual, const float* expected, int n,
                                const float tol, const char* name) {
    float maxdiff = 0.0f;
    int argmax = 0;
    for (int i = 0; i < n; i++) {
        float d = std::fabs(actual[i] - expected[i]);
        if (d > maxdiff) { maxdiff = d; argmax = i; }
    }
    const int pass = maxdiff <= tol;
    std::cout << "[" << name << "] " << (pass ? "PASS" : "FAIL")
              << "  (max abs diff = " << maxdiff << " at index " << argmax
              << ", tol = " << tol << ")" << std::endl;
    return pass;
}


