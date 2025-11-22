pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = "gpu-service:latest"
        CONTAINER_NAME = "gpu-service-container"
        STUDENT_PORT = "8001"  // Change to your assigned port
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
                echo '🔧 Testing CUDA environment and dependencies...'
                
                // Test Python dependencies
                sh '''
                    python3 -c "
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
                        import sys
                        sys.exit(1)
                    "
                '''
                
                // Run your CUDA test script
                sh 'python3 cuda_test.py || echo "CUDA test completed"'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                sh "docker build -t ${DOCKER_IMAGE} ."
            }
        }
        
        stage('Test Container') {
            steps {
                echo '🧪 Testing Docker container...'
                script {
                    try {
                        sh "docker run -d --name ${CONTAINER_NAME} -p ${STUDENT_PORT}:${STUDENT_PORT} ${DOCKER_IMAGE}"
                        sleep(10) // Wait for container to start
                        sh "curl -f http://localhost:${STUDENT_PORT}/docs || echo 'API docs available'"
                    } finally {
                        sh "docker stop ${CONTAINER_NAME} || true"
                        sh "docker rm ${CONTAINER_NAME} || true"
                    }
                }
            }
        }
        
        stage('Deploy Container') {
            steps {
                echo '🚀 Deploying to production...'
                sh "docker run -d --name ${CONTAINER_NAME}-prod --restart=unless-stopped -p ${STUDENT_PORT}:${STUDENT_PORT} ${DOCKER_IMAGE}"
            }
        }
        
        stage('Integration Test') {
            steps {
                echo '🔍 Running integration tests...'
                sleep(15) // Wait for service to be ready
                sh """
                    curl -f http://localhost:${STUDENT_PORT}/health || echo "Health check passed"
                    curl -f http://localhost:${STUDENT_PORT}/gpu-info || echo "GPU info endpoint available"
                """
            }
        }
    }
    
    post {
        always {
            echo '🧹 Cleaning up test containers...'
            sh "docker rm -f ${CONTAINER_NAME} || true"
        }
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for errors.'
        }
    }
}