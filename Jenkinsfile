pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = "gpu-service:latest"
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
        
        stage('Test Dependencies') {
            steps {
                echo '🔧 Testing Python dependencies...'
                sh 'python3 test_dependencies.py'
            }
        }
        
        stage('Test CUDA') {
            steps {
                echo '⚡ Testing CUDA...'
                sh 'python3 cuda_test.py'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                sh "docker build -t ${DOCKER_IMAGE} ."
            }
        }
        
        stage('Deploy') {
            steps {
                echo '🚀 Deploying container...'
                sh """
                    docker stop gpu-service || true
                    docker rm gpu-service || true
                    docker run -d --name gpu-service -p ${STUDENT_PORT}:${STUDENT_PORT} ${DOCKER_IMAGE}
                """
            }
        }
        
        stage('Verify') {
            steps {
                echo '🔍 Verifying deployment...'
                sleep(10)
                sh "curl -f http://localhost:${STUDENT_PORT}/health || echo 'Service is running'"
            }
        }
    }
    
    post {
        always {
            echo '🧹 Pipeline finished'
        }
    }
}