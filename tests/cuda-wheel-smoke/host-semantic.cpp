#include <iostream>

#include "smoke.h"

int main() {
  std::cout << "cuda-wheel-smoke: " << host_value() + kernel_value() << std::endl;
  return 0;
}

int host_value() {
  return 5 + SMOKE_OFFSET + CCACHE_SMOKE_OPTION;
}
