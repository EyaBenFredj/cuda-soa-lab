#!/usr/bin/env python3
"""
Simple dependency test for Jenkins
"""

print("🔍 Testing dependencies...")

try:
    import numpy as np
    print('✅ NumPy installed successfully')
    
    import numba
    print('✅ Numba installed successfully')
    
    # Create simple test data
    matrix_a = np.random.rand(10, 10).astype(np.float32)
    matrix_b = np.random.rand(10, 10).astype(np.float32)
    np.savez('test_matrix_a.npz', matrix_a)
    np.savez('test_matrix_b.npz', matrix_b)
    print('✅ Test matrices created successfully')
    
except Exception as e:
    print(f'❌ Dependency test failed: {e}')
    exit(1)

print("✅ All dependency tests passed!")