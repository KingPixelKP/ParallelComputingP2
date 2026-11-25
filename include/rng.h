#ifndef COMPAR2627_TRANSFORMER_RNG_H
#define COMPAR2627_TRANSFORMER_RNG_H

// Fills n floats with pseudo-random values in [lo, hi]. Uses libc rand(),
// so call srand() once at program start for reproducibility within a run.
// Exact bit-for-bit reproducibility across machines/compilers is NOT
// guaranteed and isn't needed here: validate/verify_reference.py
// compares against dumped values, not against a re-generated sequence,
// so the two sides never need to agree on how the random numbers were
// produced -- only on what the correct output is for whatever numbers
// were dumped.
void fill_random(float* a, int n, float lo, float hi);

#endif //COMPAR2627_TRANSFORMER_RNG_H
