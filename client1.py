from fastapi import FastAPI
import numpy as np
from numba import cuda
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model parameters
w_local = np.random.randn()
b_local = np.random.randn()

@app.post("/train")
async def train():
    """Train linear regression model on local data and return weights"""
    global w_local, b_local
    
    logger.info("Starting local training on client1...")
    
    # Generate synthetic data for client1
    np.random.seed(42)
    X = np.random.rand(1000, 1) * 10
    y_true = 3.5 * X + 2.0 + np.random.randn(1000, 1) * 0.5
    
    # Train the model
    w_local, b_local = train_linear_regression(X.flatten(), y_true.flatten())
    
    logger.info(f"Client1 training completed: w={w_local:.4f}, b={b_local:.4f}")
    
    return {"w": float(w_local), "b": float(b_local)}

@app.get("/health")
async def health():
    return {"status": "healthy", "client": "client1"}

def train_linear_regression(X, y, learning_rate=0.01, epochs=100):
    """Train linear regression using GPU-accelerated gradient computation"""
    w = np.random.randn()
    b = np.random.randn()
    
    # Check if CUDA is available
    if not cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return train_linear_regression_cpu(X, y, learning_rate, epochs)
    
    # Copy data to GPU
    X_gpu = cuda.to_device(X.astype(np.float32))
    y_gpu = cuda.to_device(y.astype(np.float32))
    
    for epoch in range(epochs):
        # Compute gradients on GPU
        grad_w, grad_b = compute_gradients_gpu(X_gpu, y_gpu, w, b)
        
        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    
    return w, b

def train_linear_regression_cpu(X, y, learning_rate=0.01, epochs=100):
    """CPU fallback implementation"""
    w = np.random.randn()
    b = np.random.randn()
    N = len(X)
    
    for epoch in range(epochs):
        # Compute predictions
        y_pred = w * X + b
        # Compute errors
        errors = y_pred - y
        # Compute gradients
        grad_w = np.dot(errors, X) / N
        grad_b = np.sum(errors) / N
        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    
    return w, b

@cuda.jit
def gradient_kernel(X, y, w, b, grad_w, grad_b, N):
    """CUDA kernel to compute gradients for linear regression"""
    idx = cuda.grid(1)
    
    if idx < N:
        # Prediction
        y_pred = w * X[idx] + b
        # Error
        error = y_pred - y[idx]
        # Thread-local gradients
        local_grad_w = error * X[idx]
        local_grad_b = error
        
        # Atomic add to global gradient sums
        cuda.atomic.add(grad_w, 0, local_grad_w)
        cuda.atomic.add(grad_b, 0, local_grad_b)

def compute_gradients_gpu(X_gpu, y_gpu, w, b):
    """Compute gradients using GPU kernel"""
    N = len(X_gpu)
    
    # Initialize gradient arrays on GPU
    grad_w = cuda.to_device(np.zeros(1, dtype=np.float32))
    grad_b = cuda.to_device(np.zeros(1, dtype=np.float32))
    
    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    gradient_kernel[blocks_per_grid, threads_per_block](
        X_gpu, y_gpu, w, b, grad_w, grad_b, N
    )
    
    # Copy results back to CPU
    grad_w_cpu = grad_w.copy_to_host()[0] / N
    grad_b_cpu = grad_b.copy_to_host()[0] / N
    
    return grad_w_cpu, grad_b_cpu

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")