import requests
import numpy as np
import time
import os

class GPUServiceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """Test health endpoint"""
        print("🧪 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"✅ Health check: {response.status_code}")
            print(f"📊 Response: {response.json()}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_gpu_info(self):
        """Test GPU info endpoint"""
        print("\n🧪 Testing GPU info endpoint...")
        try:
            response = requests.get(f"{self.base_url}/gpu-info")
            print(f"✅ GPU info: {response.status_code}")
            data = response.json()
            print(f"📊 CUDA available: {data.get('cuda_available', False)}")
            print(f"📊 GPU devices: {data.get('gpu_devices', [])}")
            return True
        except Exception as e:
            print(f"❌ GPU info failed: {e}")
            return False
    
    def create_test_matrix(self, shape, filename):
        """Create a test matrix and save as NPZ"""
        matrix = np.random.rand(*shape).astype(np.float32)
        np.savez(filename, matrix)
        return matrix
    
    def test_matrix_addition(self, shape=(512, 512)):
        """Test matrix addition endpoint"""
        print(f"\n🧪 Testing matrix addition {shape}...")
        
        # Create test matrices
        matrix_a_file = "test_matrix_a.npz"
        matrix_b_file = "test_matrix_b.npz"
        
        matrix_a = self.create_test_matrix(shape, matrix_a_file)
        matrix_b = self.create_test_matrix(shape, matrix_b_file)
        
        try:
            # Upload files
            with open(matrix_a_file, 'rb') as f1, open(matrix_b_file, 'rb') as f2:
                files = {
                    'file_a': (matrix_a_file, f1, 'application/octet-stream'),
                    'file_b': (matrix_b_file, f2, 'application/octet-stream')
                }
                
                start_time = time.time()
                response = requests.post(f"{self.base_url}/add", files=files)
                elapsed_time = time.time() - start_time
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Matrix addition successful!")
                print(f"📊 Shape: {result['matrix_shape']}")
                print(f"⏱️  Total time: {result['elapsed_time']:.4f}s")
                print(f"🎯 GPU time: {result.get('gpu_computation_time', 0):.4f}s")
                print(f"💾 GPU memory: {result.get('gpu_memory_used_mb', 0)}MB")
                print(f"🌐 Request time: {elapsed_time:.4f}s")
                return True
            else:
                print(f"❌ Matrix addition failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Matrix addition error: {e}")
            return False
        finally:
            # Cleanup
            if os.path.exists(matrix_a_file):
                os.remove(matrix_a_file)
            if os.path.exists(matrix_b_file):
                os.remove(matrix_b_file)
    
    def test_metrics(self):
        """Test Prometheus metrics endpoint"""
        print("\n🧪 Testing metrics endpoint...")
        try:
            response = requests.get(f"{self.base_url}/metrics")
            print(f"✅ Metrics: {response.status_code}")
            # Count the number of metrics
            lines = response.text.strip().split('\n')
            metric_count = len([line for line in lines if line and not line.startswith('#')])
            print(f"📊 Found {metric_count} metrics")
            return True
        except Exception as e:
            print(f"❌ Metrics failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive GPU Service Tests")
        print("=" * 50)
        
        tests = [
            self.test_health,
            self.test_gpu_info,
            lambda: self.test_matrix_addition((256, 256)),
            lambda: self.test_matrix_addition((512, 512)),
            self.test_metrics
        ]
        
        results = []
        for test in tests:
            results.append(test())
            time.sleep(1)  # Small delay between tests
        
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 50)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Service is working correctly.")
        else:
            print("❌ Some tests failed. Check the service configuration.")

if __name__ == "__main__":
    tester = GPUServiceTester()
    tester.run_all_tests()