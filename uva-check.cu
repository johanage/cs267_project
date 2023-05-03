#include <cuda.h>
#include <iostream>
void print_ptr_dev (int max_depth, int current_depth = 0)
{
  cudaSetDevice (current_depth % 2);

  void *ptr {};

  cudaMalloc (&ptr, 1024 * 1024 * 4);

  cudaPointerAttributes pointer_attributes {};
  cudaPointerGetAttributes (&pointer_attributes, ptr);

  std::cout << (size_t)ptr << ", " << pointer_attributes.device << "\n";

  if (max_depth > current_depth)
      print_ptr_dev (max_depth, current_depth + 1);

  cudaFree (ptr);
}

int main ()
{
  print_ptr_dev (40);
  return 0;
}
