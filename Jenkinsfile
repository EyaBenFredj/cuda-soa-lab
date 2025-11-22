#!/usr/bin/env python3
"""
Simple CUDA test script for Jenkins pipeline
"""

try:
    import numpy as np
    from numba import cuda
    import sys

    print("🔍 Testing CUDA availability...")
    
    # Check if CUDA is available
    if not cuda.is_available():
        print("❌ CUDA is not available")
        sys.exit(1)
    
    print(f"✅ CUDA is available")
    print(f"✅ Detected {cuda.gpus.len()} GPU(s)")
    
    # Simple vector addition test
    @cuda.jit
    def add_kernel(a, b, result):
        idx = cuda.grid(1)
        if idx < a.size:
            result[idx] = a[idx] + b[idx]
    
    # Test data
    size = 100
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32)
    result = np.zeros(size, dtype=np.float32)
    
    # Copy to device
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(result)
    
    # Launch kernel
    threads_per_block = 32
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_result)
    
    # Copy back and verify
    host_result = d_result.copy_to_host()
    expected = a + b
    
    if np.allclose(host_result, expected):
        print("✅ CUDA kernel test passed!")
        print(f"   Input: {a[:5]}... + {b[:5]}...")
        print(f"   Result: {host_result[:5]}...")
    else:
        print("❌ CUDA kernel test failed!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
    sys.exit(1)