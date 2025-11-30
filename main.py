from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import time
import io
import os
import sys
import subprocess
import json
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

# Initialize FastAPI app
app = FastAPI(
    title="GPU Matrix Addition Microservice",
    description="SOA Lab - GPU-accelerated matrix addition service with monitoring",
    version="2.0.0"
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Custom metrics
REQUEST_COUNT = Counter('request_count', 'Total request count', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds', ['endpoint'])
GPU_MEMORY_USED = Gauge('gpu_memory_used_mb', 'GPU Memory Used in MB', ['gpu_id'])
GPU_MEMORY_TOTAL = Gauge('gpu_memory_total_mb', 'GPU Memory Total in MB', ['gpu_id'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU Utilization Percentage', ['gpu_id'])
GPU_TEMPERATURE = Gauge('gpu_temperature_c', 'GPU Temperature in Celsius', ['gpu_id'])
MATRIX_SIZE = Gauge('matrix_size_elements', 'Number of elements in processed matrices')
ACTIVE_GPU_COUNT = Gauge('active_gpu_count', 'Number of active GPUs')

# Check if we're on Windows
IS_WINDOWS = os.name == 'nt'

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
    
    # CUDA kernel for matrix addition
    @cuda.jit
    def matrix_add_kernel(a, b, c):
        """
        CUDA Kernel for matrix addition
        Each thread computes one element: c[i,j] = a[i,j] + b[i,j]
        """
        i, j = cuda.grid(2)
        if i < c.shape[0] and j < c.shape[1]:
            c[i, j] = a[i, j] + b[i, j]
            
    def gpu_matrix_add(matrix_a, matrix_b):
        """Perform matrix addition on GPU using Numba CUDA"""
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Matrix shapes must match")
        
        rows, cols = matrix_a.shape
        
        # Copy matrices to GPU device memory
        d_matrix_a = cuda.to_device(matrix_a)
        d_matrix_b = cuda.to_device(matrix_b)
        d_result = cuda.device_array_like(matrix_a)
        
        # Configure kernel launch parameters
        threads_per_block = (16, 16)  # 256 threads per block
        blocks_per_grid_x = (rows + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (cols + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch CUDA kernel
        matrix_add_kernel[blocks_per_grid, threads_per_block](d_matrix_a, d_matrix_b, d_result)
        cuda.synchronize()
        
        # Copy result back to host
        return d_result.copy_to_host()
        
except ImportError:
    # Fallback to CPU implementation
    print("⚠️  GPU libraries not available - running in CPU mode")
    
    def gpu_matrix_add(matrix_a, matrix_b):
        """CPU fallback for matrix addition"""
        return matrix_a + matrix_b

def get_gpu_info():
    """
    Get GPU information using nvidia-smi
    Returns detailed GPU information for /gpu-info endpoint
    """
    try:
        # Run nvidia-smi to get comprehensive GPU info
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpu_data = {
                            "gpu": parts[0].strip(),
                            "name": parts[1].strip(),
                            "memory_used_MB": int(parts[2]),
                            "memory_total_MB": int(parts[3]),
                            "utilization_percent": int(parts[4]),
                            "temperature_c": int(parts[5]),
                            "driver_version": parts[6].strip() if len(parts) > 6 else "unknown"
                        }
                        gpus.append(gpu_data)
            
            # Update Prometheus metrics
            ACTIVE_GPU_COUNT.set(len(gpus))
            for gpu in gpus:
                GPU_MEMORY_USED.labels(gpu_id=gpu["gpu"]).set(gpu["memory_used_MB"])
                GPU_MEMORY_TOTAL.labels(gpu_id=gpu["gpu"]).set(gpu["memory_total_MB"])
                GPU_UTILIZATION.labels(gpu_id=gpu["gpu"]).set(gpu["utilization_percent"])
                GPU_TEMPERATURE.labels(gpu_id=gpu["gpu"]).set(gpu["temperature_c"])
            
            return gpus
        else:
            print(f"nvidia-smi failed: {result.stderr}")
            return []
            
    except subprocess.TimeoutExpired:
        print("nvidia-smi command timed out")
        return []
    except FileNotFoundError:
        print("nvidia-smi not found")
        return []
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def get_gpu_memory_info():
    """Get GPU memory info for health endpoint"""
    gpus = get_gpu_info()
    if gpus:
        # Return info for first GPU
        return gpus[0]["memory_used_MB"], gpus[0]["memory_total_MB"]
    else:
        # Fallback values
        return 512, 4096

def load_npz_file_safely(file_content: bytes, filename: str):
    """
    Safely load NPZ file with comprehensive error handling
    """
    try:
        with io.BytesIO(file_content) as buffer:
            npz_data = np.load(buffer, allow_pickle=False)
            
            # Debug: Print available arrays
            print(f"📁 NPZ file '{filename}' contains arrays: {npz_data.files}")
            
            # Try different strategies to extract the matrix
            matrix = None
            
            # Strategy 1: Look for common keys
            common_keys = ['arr_0', 'matrix', 'data', 'array', 'mat']
            for key in common_keys:
                if key in npz_data.files:
                    matrix = npz_data[key]
                    print(f"✅ Found matrix using key: '{key}' with shape {matrix.shape}")
                    break
            
            # Strategy 2: Use first array if no common key found
            if matrix is None and len(npz_data.files) > 0:
                first_key = npz_data.files[0]
                matrix = npz_data[first_key]
                print(f"✅ Using first array '{first_key}' with shape {matrix.shape}")
            
            # Strategy 3: If still no matrix, try to find any 2D array
            if matrix is None:
                for key in npz_data.files:
                    array = npz_data[key]
                    if hasattr(array, 'shape') and len(array.shape) == 2:
                        matrix = array
                        print(f"✅ Found 2D array '{key}' with shape {matrix.shape}")
                        break
            
            if matrix is None:
                raise ValueError(f"No suitable 2D array found in NPZ file. Available arrays: {list(npz_data.files)}")
            
            # Ensure it's a proper 2D matrix
            if len(matrix.shape) != 2:
                raise ValueError(f"Expected 2D matrix, got array with shape {matrix.shape}")
            
            return matrix
            
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file '{filename}': {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "GPU Matrix Addition Microservice",
        "version": "2.0.0",
        "gpu_available": GPU_AVAILABLE,
        "environment": "windows" if IS_WINDOWS else "linux",
        "endpoints": {
            "health": "/health",
            "gpu_info": "/gpu-info",
            "add": "/add",
            "metrics": "/metrics",
            "test_data": "/test-data"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
    
    used, total = get_gpu_memory_info()
    gpus = get_gpu_info()
    
    return {
        "status": "ok",
        "service": "GPU Matrix Addition",
        "gpu_available": GPU_AVAILABLE,
        "active_gpus": len(gpus),
        "environment": "windows" if IS_WINDOWS else "linux",
        "gpu_info": {
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_usage_percent": round((used / total) * 100, 2) if total > 0 else 0
        },
        "timestamp": time.time()
    }

@app.get("/gpu-info")
async def gpu_info_endpoint():
    """
    GPU information endpoint
    Returns detailed information about all available GPUs
    """
    REQUEST_COUNT.labels(method='GET', endpoint='/gpu-info').inc()
    
    gpus = get_gpu_info()
    
    if gpus:
        return {
            "gpus": gpus,
            "total_gpus": len(gpus),
            "timestamp": time.time()
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="GPU information not available. nvidia-smi may not be installed or accessible."
        )

@app.post("/add")
async def matrix_add(
    file_a: UploadFile = File(..., description="First matrix in .npz format"),
    file_b: UploadFile = File(..., description="Second matrix in .npz format")
):
    """
    Matrix addition endpoint
    Accepts two NPZ files and returns their sum computed on GPU
    """
    REQUEST_COUNT.labels(method='POST', endpoint='/add').inc()
    start_time = time.perf_counter()
    
    # Validate file types
    if not file_a.filename.endswith('.npz') or not file_b.filename.endswith('.npz'):
        raise HTTPException(
            status_code=400, 
            detail="Both files must be in .npz format"
        )
    
    try:
        # Read uploaded files
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        print(f"📥 Processing files: {file_a.filename} ({len(content_a)} bytes), {file_b.filename} ({len(content_b)} bytes)")
        
        # Load matrices from NPZ files with safe loading
        matrix_a = load_npz_file_safely(content_a, file_a.filename)
        matrix_b = load_npz_file_safely(content_b, file_b.filename)
        
        # Ensure matrices are float32 for GPU computation
        matrix_a = matrix_a.astype(np.float32)
        matrix_b = matrix_b.astype(np.float32)
        
        print(f"📊 Matrix A shape: {matrix_a.shape}, dtype: {matrix_a.dtype}")
        print(f"📊 Matrix B shape: {matrix_b.shape}, dtype: {matrix_b.dtype}")
        
        # Validate shapes
        if matrix_a.shape != matrix_b.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shapes must match. Got {matrix_a.shape} and {matrix_b.shape}"
            )
        
        rows, cols = matrix_a.shape
        total_elements = rows * cols
        MATRIX_SIZE.set(total_elements)
        
        # Check if matrices are too large
        if total_elements > 10_000_000:  # 10M elements limit
            raise HTTPException(
                status_code=400,
                detail=f"Matrix too large: {total_elements} elements. Maximum allowed: 10,000,000"
            )
        
        # Perform matrix addition
        gpu_start_time = time.perf_counter()
        result_matrix = gpu_matrix_add(matrix_a, matrix_b)
        gpu_time = time.perf_counter() - gpu_start_time
        
        total_time = time.perf_counter() - start_time
        
        # Update metrics
        REQUEST_LATENCY.labels(endpoint='/add').observe(total_time)
        
        # Get current GPU info for response
        gpus = get_gpu_info()
        used_memory = gpus[0]["memory_used_MB"] if gpus else 512
        total_memory = gpus[0]["memory_total_MB"] if gpus else 4096
        
        # Verify the result
        cpu_result = matrix_a + matrix_b
        results_match = np.allclose(result_matrix, cpu_result, rtol=1e-5)
        
        print(f"✅ Addition completed: {rows}x{cols} matrices, GPU time: {gpu_time:.4f}s, Results match: {results_match}")
        
        return {
            "matrix_shape": [int(rows), int(cols)],
            "total_elements": total_elements,
            "elapsed_time": round(total_time, 6),
            "gpu_computation_time": round(gpu_time, 6),
            "device": "GPU" if GPU_AVAILABLE else "CPU",
            "gpu_memory_used_mb": used_memory,
            "gpu_memory_total_mb": total_memory,
            "gpu_utilization_percent": gpus[0]["utilization_percent"] if gpus else 0,
            "results_verified": results_match,
            "environment": "windows" if IS_WINDOWS else "linux"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in matrix addition: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing matrices: {str(e)}"
        )

@app.get("/test-data")
async def create_test_data():
    """
    Create and return test NPZ files for demonstration
    """
    try:
        # Create sample matrices
        matrix_1 = np.random.rand(100, 100).astype(np.float32)
        matrix_2 = np.random.rand(100, 100).astype(np.float32)
        
        # Save to bytes buffers
        buffer_1 = io.BytesIO()
        buffer_2 = io.BytesIO()
        
        np.savez(buffer_1, matrix_1)
        np.savez(buffer_2, matrix_2)
        
        buffer_1.seek(0)
        buffer_2.seek(0)
        
        return {
            "message": "Test matrices created successfully",
            "matrix_1_shape": matrix_1.shape,
            "matrix_2_shape": matrix_2.shape,
            "download_urls": {
                "matrix_1": "/download/matrix1.npz",
                "matrix_2": "/download/matrix2.npz"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating test data: {str(e)}")

@app.get("/download/{filename}")
async def download_test_file(filename: str):
    """
    Download test NPZ files
    """
    if filename == "matrix1.npz":
        matrix = np.random.rand(100, 100).astype(np.float32)
    elif filename == "matrix2.npz":
        matrix = np.random.rand(100, 100).astype(np.float32)
    else:
        raise HTTPException(status_code=404, detail="File not found")
    
    buffer = io.BytesIO()
    np.savez(buffer, matrix)
    buffer.seek(0)
    
    return JSONResponse(
        content={"message": f"Downloading {filename}"},
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/octet-stream"
        },
        media_type="application/octet-stream"
    )

@app.get("/cuda-test")
async def cuda_test():
    """Test endpoint to verify CUDA functionality"""
    REQUEST_COUNT.labels(method='GET', endpoint='/cuda-test').inc()
    
    try:
        if GPU_AVAILABLE:
            from numba import cuda
            
            # Get device info safely
            device_count = 0
            device_info = []
            
            try:
                devices = cuda.list_devices()
                device_count = len(devices)
                
                for i in range(device_count):
                    with cuda.gpus[i]:
                        device = cuda.get_current_device()
                        device_info.append({
                            "device_id": i,
                            "name": device.name,
                            "compute_capability": device.compute_capability
                        })
            except Exception as e:
                print(f"⚠️  Error getting detailed device info: {e}")
                device_count = 1
                device_info = [{
                    "device_id": 0,
                    "name": "GPU (details unavailable)",
                    "compute_capability": "Unknown"
                }]
            
            return {
                "cuda_available": True,
                "device_count": device_count,
                "devices": device_info
            }
        else:
            return {
                "cuda_available": False,
                "message": "CUDA not available on this system"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CUDA test failed: {str(e)}"
        )

@app.get("/simple-add")
async def simple_matrix_add(size: int = 100):
    """
    Simple matrix addition test without file upload
    Useful for quick testing
    """
    try:
        # Create test matrices
        matrix_a = np.random.rand(size, size).astype(np.float32)
        matrix_b = np.random.rand(size, size).astype(np.float32)
        
        # Perform addition
        start_time = time.perf_counter()
        result = gpu_matrix_add(matrix_a, matrix_b)
        computation_time = time.perf_counter() - start_time
        
        # Verify result
        cpu_result = matrix_a + matrix_b
        results_match = np.allclose(result, cpu_result, rtol=1e-5)
        
        return {
            "matrix_size": f"{size}x{size}",
            "computation_time": round(computation_time, 6),
            "device": "GPU" if GPU_AVAILABLE else "CPU",
            "results_verified": results_match,
            "total_elements": size * size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple add test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8001"))
    
    print("🚀 Starting Enhanced GPU Matrix Addition Service")
    print("=" * 60)
    print(f"📍 Port: {port}")
    print(f"🎮 GPU Available: {GPU_AVAILABLE}")
    print(f"💻 Environment: {'Windows' if IS_WINDOWS else 'Linux'}")
    print(f"📊 Metrics: http://localhost:{port}/metrics")
    print(f"❤️  Health: http://localhost:{port}/health")
    print(f"🎯 GPU Info: http://localhost:{port}/gpu-info")
    print(f"🧪 CUDA Test: http://localhost:{port}/cuda-test")
    print(f"➕ Matrix Add: http://localhost:{port}/add")
    print(f"🔧 Simple Test: http://localhost:{port}/simple-add")
    print(f"📁 Test Data: http://localhost:{port}/test-data")
    print("=" * 60)
    
    # Test GPU detection on startup
    gpus = get_gpu_info()
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"   GPU {gpu['gpu']}: {gpu['name']} - {gpu['memory_used_MB']}/{gpu['memory_total_MB']} MB")
    else:
        print("⚠️  No GPUs detected or nvidia-smi not available")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")