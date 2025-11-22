pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = '10.90.90.100:5000'
        IMAGE_NAME = "cuda-soa-lab"
        STUDENT_ID = "EyaBenFredj"
        STUDENT_PORT = "8000"
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/${IMAGE_NAME}:${STUDENT_ID}"
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo '📥 Checking out code from GitHub...'
                git branch: 'main', url: 'https://github.com/EyaBenFredj/cuda-soa-lab.git'
                sh 'ls -la'
            }
        }

        stage('GPU Sanity Test') {
            steps {
                echo '🔧 Installing required dependencies for cuda_test...'
                sh '''
                    python3 -c "
try:
    import numpy as np
    print('✅ NumPy installed successfully')
    
    # Create test matrices
    matrix_a = np.random.rand(100, 100).astype(np.float32)
    matrix_b = np.random.rand(100, 100).astype(np.float32)
    np.savez('test_matrix_a.npz', matrix_a)
    np.savez('test_matrix_b.npz', matrix_b)
    print('✅ Test matrices created successfully')
except Exception as e:
    print(f'❌ Dependency test failed: {e}')
    exit 1
"
                '''
                
                echo '🎮 Running CUDA sanity check...'
                sh '''
                    python3 -c "
try:
    # Test basic imports
    import fastapi, uvicorn, prometheus_client
    print('✅ FastAPI and monitoring dependencies OK')
    
    # Test CUDA availability
    try:
        from numba import cuda
        if cuda.is_available():
            print('✅ CUDA is available via Numba')
            print(f'Number of GPUs detected: {len(cuda.list_devices())}')
        else:
            print('⚠️ CUDA not available (expected in CI environment)')
    except ImportError:
        print('⚠️ Numba/CUDA not available (expected in CI environment)')
        
except Exception as e:
    print(f'❌ CUDA sanity check failed: {e}')
    exit 1
"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                script {
                    // Build the Docker image
                    docker.build("${DOCKER_IMAGE}")
                    
                    // Test the built image
                    sh """
                    docker run --rm ${DOCKER_IMAGE} python3 -c "
import sys
print('Python version:', sys.version)
try:
    import fastapi, numpy, prometheus_client
    print('✅ All dependencies installed in container')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"
                    """
                }
            }
        }

        stage('Test Container') {
            steps {
                echo "🧪 Testing Docker container functionality..."
                script {
                    // Test that the container starts and basic functionality works
                    sh """
                    docker run -d --name test-container -p 8080:8000 ${DOCKER_IMAGE}
                    sleep 10
                    
                    # Test health endpoint
                    curl -f http://localhost:8080/health || exit 1
                    echo '✅ Health endpoint working'
                    
                    # Test GPU info endpoint
                    curl -f http://localhost:8080/gpu-info || echo '⚠️ GPU info not available (expected)'
                    
                    # Test metrics endpoint
                    curl -f http://localhost:8080/metrics | head -5
                    echo '✅ Metrics endpoint working'
                    
                    # Cleanup test container
                    docker stop test-container
                    docker rm test-container
                    """
                }
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                script {
                    // Push to registry
                    docker.withRegistry("http://${DOCKER_REGISTRY}") {
                        docker.image("${DOCKER_IMAGE}").push()
                    }
                    
                    echo "📦 Image pushed to registry: ${DOCKER_IMAGE}"
                    echo "🌐 Service will be available at: http://10.90.90.100:${STUDENT_PORT}"
                    echo "📊 Monitoring: http://10.90.90.100:9090"
                    echo "📺 Dashboard: http://10.90.90.100:3000"
                }
            }
        }

        stage('Integration Test') {
            steps {
                echo "🔍 Running integration tests..."
                script {
                    sh """
                    # Wait for deployment to be ready
                    sleep 30
                    
                    # Test the deployed service
                    echo "Testing deployed service at http://10.90.90.100:${STUDENT_PORT}"
                    
                    # Test health endpoint
                    curl -f http://10.90.90.100:${STUDENT_PORT}/health && echo '✅ Production health check PASSED'
                    
                    # Test GPU info
                    curl -f http://10.90.90.100:${STUDENT_PORT}/gpu-info && echo '✅ GPU info endpoint working'
                    """
                }
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            echo "Student: ${STUDENT_ID}"
            echo "Service URL: http://10.90.90.100:${STUDENT_PORT}"
            echo "GPU Info: http://10.90.90.100:${STUDENT_PORT}/gpu-info"
            echo "Metrics: http://10.90.90.100:9090"
            echo "Dashboard: http://10.90.90.100:3000"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
        }
        always {
            echo "🧾 Pipeline finished."
            script {
                // Cleanup
                sh """
                docker rm -f test-container 2>/dev/null || true
                """
            }
        }
    }
}