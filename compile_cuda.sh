#!/bin/bash

# CUDA Compilation Script for SOA Lab
# This script compiles the CUDA code and sets up the environment

echo "🔧 Starting CUDA Compilation Setup..."
echo "======================================"

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "✅ nvcc found: $(nvcc --version | head -n1)"

# Check if we're in the right directory
if [ ! -f "gpu_service.cu" ]; then
    echo "❌ ERROR: gpu_service.cu not found in current directory"
    echo "💡 Make sure you're in the project root directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📄 Found gpu_service.cu"

# Create lib directory if it doesn't exist
mkdir -p lib

echo "🔨 Compiling CUDA code..."
echo "--------------------------------------"

# Compile CUDA code with optimization flags
nvcc -Xcompiler -fPIC -shared -o lib/libgpuadd.so gpu_service.cu \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -O3

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: libgpuadd.so compiled successfully!"
    echo "📁 Output: lib/libgpuadd.so"
    
    # Display file information
    echo "📊 File info:"
    file lib/libgpuadd.so
    echo "💾 File size: $(du -h lib/libgpuadd.so | cut -f1)"
    
else
    echo "❌ ERROR: CUDA compilation failed!"
    exit 1
fi

echo ""
echo "🧪 Verification Steps:"
echo "--------------------------------------"

# Verify the shared library was created
if [ -f "lib/libgpuadd.so" ]; then
    echo "✅ Shared library exists and is accessible"
else
    echo "❌ Shared library not found"
    exit 1
fi

# Test if the library can be loaded (optional)
echo "🔍 Testing library loading..."
python3 -c "
try:
    from ctypes import cdll
    lib = cdll.LoadLibrary('./lib/libgpuadd.so')
    print('✅ Library can be loaded successfully')
except Exception as e:
    print(f'❌ Error loading library: {e}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo "======================================"
echo "Next steps:"
echo "1. Run: python main.py"
echo "2. Test: curl http://localhost:8000/health"
echo "3. Deploy with Jenkins"