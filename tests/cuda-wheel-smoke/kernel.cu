// This device-only function makes nvcc compile a CUDA translation unit.
__device__ int square(int value) {
  return value * value;
}

#include "smoke.h"

int kernel_value() {
  return 7 + SMOKE_OFFSET + CCACHE_SMOKE_OPTION;
}
