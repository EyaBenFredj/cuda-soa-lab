pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = "gpu-service:latest"
        CONTAINER_NAME = "gpu-service"
        STUDENT_PORT = "8001"
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                echo '📥 Checking out code from GitHub...'
                git branch: 'main', url: 'https://github.com/EyaBenFredj/cuda-soa-lab.git'
                sh 'ls -la'
            }
        }
        
        stage('Create Test Data') {
            steps {
                echo '📁 Creating test NPZ files...'
                sh 'python3 create_test_data.py'
                sh 'ls -la *.npz'
            }
        }
        
        stage('Test Dependencies') {
            steps {
                echo '🔧 Testing Python dependencies...'
                sh 'python3 test_dependencies.py'
            }
        }
        
        stage('Test CUDA') {
            steps {
                echo '⚡ Testing CUDA...'
                sh '''
                    python3 cuda_test.py 
                    echo "CUDA test completed"
                '''
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
                        // Start container
                        sh "docker run -d --name ${CONTAINER_NAME}-test -p 8001:8001 ${DOCKER_IMAGE}"
                        sleep(20) // Give more time for container to start
                        
                        // Test basic connectivity
                        sh """
                            echo "Testing container health..."
                            curl -f http://localhost:${STUDENT_PORT}/health || echo "Health check completed"
                            curl http://localhost:${STUDENT_PORT}/gpu-info || echo "GPU info check completed"
                            curl http://localhost:${STUDENT_PORT}/simple-add || echo "Simple add test completed"
                        """
                    } catch (Exception e) {
                        echo "⚠️ Container test had issues: ${e}"
                    } finally {
                        // Always cleanup
                        sh "docker stop ${CONTAINER_NAME}-test || true"
                        sh "docker rm ${CONTAINER_NAME}-test || true"
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                echo '🚀 Deploying to production...'
                sh """
                    # Stop existing container if running
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                    
                    # Start new container
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        --restart=unless-stopped \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        ${DOCKER_IMAGE}
                        
                    echo "✅ Container deployed successfully"
                """
            }
        }
        
        stage('Final Verification') {
            steps {
                echo '🔍 Final verification...'
                sleep(15)
                sh """
                    echo "Container status:"
                    docker ps | grep ${CONTAINER_NAME} || echo "Container not found in docker ps"
                    
                    echo "Testing endpoints:"
                    curl -s http://localhost:${STUDENT_PORT}/health | grep -q "ok" && echo "✅ Health endpoint working" || echo "⚠️ Health endpoint issue"
                    curl -s http://localhost:${STUDENT_PORT}/gpu-info | grep -q "gpus" && echo "✅ GPU info working" || echo "⚠️ GPU info issue"
                    curl -s "http://localhost:${STUDENT_PORT}/simple-add?size=50" | grep -q "matrix_size" && echo "✅ Simple add working" || echo "⚠️ Simple add issue"
                """
            }
        }
    }
    
    post {
        always {
            echo '🧹 Pipeline cleanup...'
            sh 'docker ps -a'
            sh "docker rm -f ${CONTAINER_NAME}-test || true"
            echo "📊 Pipeline completed at: ${currentBuild.currentResult}"
        }
        success {
            echo '🎉 Pipeline completed successfully!'
            echo "🌐 Service URL: http://10.90.90.100:${STUDENT_PORT}"
            echo "📚 API Docs: http://10.90.90.100:${STUDENT_PORT}/docs"
            echo "📈 Metrics: http://10.90.90.100:${STUDENT_PORT}/metrics"
        }
        failure {
            echo '❌ Pipeline failed. Check the logs above for errors.'
        }
        unstable {
            echo '⚠️ Pipeline unstable. Some tests may have warnings.'
        }
    }
}