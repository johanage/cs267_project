double measure_allocation_time (unsigned int max_allocations)
{
  std::vector<char *> pointers (max_allocations, nullptr);

  const auto begin = std::chrono::high_resolution_clock::now ();
  for (unsigned int alloc_id = 0; alloc_id < max_allocations; alloc_id++)
    cudaMalloc (&pointers[alloc_id], 2 * 1024 * 1024);
  const auto end = std::chrono::high_resolution_clock::now ();

  for (auto &ptr: pointers)
    cudaFree (ptr);

  return std::chrono::duration<double> (end - begin).count ();
}

void p2p_alloc_bench ()
{
  const unsigned int allocations_count = 1000;
  const double basic_time = measure_allocation_time (allocations_count);
  cudaDeviceEnablePeerAccess (1, 0);
  const double p2p_time = measure_allocation_time (allocations_count);

  std::cout << basic_time << " s => " << p2p_time << std::endl;
}
