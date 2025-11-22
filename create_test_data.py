#!/usr/bin/env python3
"""
Create proper test NPZ files for the pipeline
"""
import numpy as np
import os

print("📁 Creating test NPZ files...")

# Create test matrices
matrix1 = np.random.rand(100, 100).astype(np.float32)
matrix2 = np.random.rand(100, 100).astype(np.float32)

# Save as NPZ files
np.savez('matrix1.npz', matrix1)
np.savez('matrix2.npz', matrix2)

print("✅ Test matrices created:")
print(f"   matrix1.npz: {matrix1.shape}, {os.path.getsize('matrix1.npz')} bytes")
print(f"   matrix2.npz: {matrix2.shape}, {os.path.getsize('matrix2.npz')} bytes")