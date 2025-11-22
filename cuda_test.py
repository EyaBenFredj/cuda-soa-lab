#!/usr/bin/env python3
print("🔍 Starting CUDA test...")

try:
    from numba import cuda
    import numpy as np
    
    print("✅ CUDA imports successful")
    
    # Check if CUDA is available
    if cuda.is_available():
        print("✅ CUDA is available")
        print(f"✅ Number of GPUs: {cuda.gpus.len()}")
    else:
        print("❌ CUDA not available")
        exit(1)
        
    # Simple test - just check we can use CUDA
    @cuda.jit
    def add_kernel(a, result):
        idx = cuda.grid(1)
        if idx < a.size:
            result[idx] = a[idx] + 1.0
    
    # Small test
    size = 10
    a = np.ones(size, dtype=np.float32)
    result = np.zeros(size, dtype=np.float32)
    
    d_a = cuda.to_device(a)
    d_result = cuda.device_array_like(result)
    
    # Launch kernel
    threads_per_block = 32
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_result)
    
    # Copy back
    host_result = d_result.copy_to_host()
    print(f"✅ CUDA test passed! Result: {host_result}")
    
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
    exit(1)