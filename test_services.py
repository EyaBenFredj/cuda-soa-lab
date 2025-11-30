import requests
import json
import time
import sys

def test_health_checks():
    """Test health endpoints of all services"""
    services = {
        "client1": "http://localhost:8001/health",
        "client2": "http://localhost:8002/health", 
        "client3": "http://localhost:8003/health",
        "aggregator": "http://localhost:9000/health"
    }
    
    print("🧪 Testing health checks...")
    all_healthy = True
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: {response.json()}")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            all_healthy = False
    
    return all_healthy

def simulate_federated_round():
    """Simulate one round of federated learning"""
    print("\n🚀 Starting Federated Learning Round...")
    
    # Train on all clients
    clients = [
        ("client1", "http://localhost:8001/train"),
        ("client2", "http://localhost:8002/train"), 
        ("client3", "http://localhost:8003/train")
    ]
    
    all_weights = []
    
    # Get weights from each client
    for client_name, client_url in clients:
        try:
            print(f"📡 Training {client_name}...")
            response = requests.post(client_url, timeout=30)
            weights = response.json()
            all_weights.append(weights)
            print(f"   {client_name} weights: w={weights['w']:.4f}, b={weights['b']:.4f}")
        except Exception as e:
            print(f"❌ Error contacting {client_name}: {e}")
    
    # Aggregate weights
    if all_weights:
        print("\n🔄 Aggregating weights...")
        aggregator_url = "http://localhost:9000/aggregate"
        aggregation_request = {"weights_list": all_weights}
        
        try:
            response = requests.post(aggregator_url, json=aggregation_request, timeout=10)
            global_weights = response.json()
            
            print(f"✅ Global model: w={global_weights['w_global']:.4f}, b={global_weights['b_global']:.4f}")
            print(f"📊 Theoretical values: w=3.5, b=2.0")
            print(f"👥 Clients participated: {global_weights['clients_count']}")
            
            # Calculate accuracy
            w_error = abs(global_weights['w_global'] - 3.5)
            b_error = abs(global_weights['b_global'] - 2.0)
            print(f"📈 Model error: w_error={w_error:.4f}, b_error={b_error:.4f}")
            
            return global_weights
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
    
    return None

if __name__ == "__main__":
    print("🏁 Federated Learning Test Suite")
    print("=" * 40)
    
    # Test health first
    if not test_health_checks():
        print("\n⚠️  Some services are not healthy. Please start all services first.")
        print("   Run: docker compose up --build")
        sys.exit(1)
    
    # Wait a moment for services to be ready
    print("\n⏳ Waiting for services to be ready...")
    time.sleep(3)
    
    # Run federated learning round
    results = simulate_federated_round()
    
    if results:
        print(f"\n🎉 Federated Learning completed successfully!")
        print(f"📋 Final Results: {results}")
    else:
        print("\n💥 Federated Learning failed!")
    
    print("\nPress Enter to exit...")
    input()