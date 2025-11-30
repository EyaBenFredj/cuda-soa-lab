
## 📸 Implementation Evidence

### Service Deployment

| Service | Status | Port |
|---------|--------|------|
| Client1 | ✅ Running | 8001 |
| Client2 | ✅ Running | 8002 | 
| Client3 | ✅ Running | 8003 | 
| Aggregator | ✅ Running | 9000 |

### Federated Learning Results

*Successful federated learning round showing all services healthy and global model aggregation*

## 🔧 Technical Implementation

### Core Components

#### 1. Client Services (`client1.py`, `client2.py`, `client3.py`)
- **Framework**: FastAPI microservices
- **Model**: Linear regression (y = wx + b)
- **Training**: Local SGD with manual gradient computation
- **GPU Support**: Numba CUDA kernels (with CPU fallback)
- **Endpoints**: 
  - `POST /train` - Perform local training
  - `GET /health` - Service health check

#### 2. Aggregator Service (`aggregator.py`)
- **Algorithm**: Federated Averaging
- **Function**: Compute global model parameters
- **Endpoints**:
  - `POST /aggregate` - Combine client weights
  - `GET /health` - Service status

### 🧠 Federated Learning Process

1. **Local Training**: Each client trains on its private synthetic data
2. **Weight Sharing**: Clients send only model parameters (w, b) to aggregator
3. **Aggregation**: Federated averaging: `w_global = average(w_clients)`
4. **Privacy Preservation**: Raw data never leaves the clients

## 📊 Results Analysis

### Model Performance
- **Theoretical Values**: w=3.5, b=2.0
- **Achieved Global Model**: w=3.6362, b=1.1040
- **Model Error**: w_error=0.1362, b_error=0.8960

### Individual Client Results
| Client | Weight (w) | Bias (b) |
|--------|------------|----------|
| Client1 | 3.5576 | 1.6570 |
| Client2 | 3.8281 | -0.1798 |
| Client3 | 3.5228 | 1.8348 |
| **Global Model** | **3.6362** | **1.1040** |

## 🚀 How to Run the System

### Prerequisites
- Python 3.8+
- Docker (optional)
- Required packages: `fastapi`, `uvicorn`, `numba`, `numpy`, `requests`, `pydantic`

### Method 1: Manual Service Start
bash
# Terminal 1 - Client1
python client1.py
python test_services.py
<img width="911" height="170" alt="o1" src="https://github.com/user-attachments/assets/f7c86e5a-37ac-4cb8-b550-f0f150001718" />

# Terminal 2 - Client2  
python client2.py
<img width="931" height="146" alt="o2" src="https://github.com/user-attachments/assets/5c02ccca-4c04-4f96-b658-1d88f311c366" />


# Terminal 3 - Client3
python client3.py
<img width="885" height="132" alt="o3" src="https://github.com/user-attachments/assets/a29189ae-564a-4ed1-b3bd-4f8fd85b83e0" />

# Terminal 4 - Aggregator
python aggregator.py
<img width="906" height="143" alt="o4" src="https://github.com/user-attachments/assets/8ea7dd84-cc05-492c-bb9c-3a3f0365a71f" />

# Terminal 5 - Testing
python test_services.py
<img width="1462" height="635" alt="o5" src="https://github.com/user-attachments/assets/40cbaada-0918-4b28-a677-c982485aee0d" />
