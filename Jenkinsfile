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
            }
        }

        stage('Basic Dependency Test') {
            steps {
                echo '🔧 Testing basic dependencies...'
                sh '''
                    python3 -c "
                    try:
                        import numpy as np
                        import fastapi
                        print('✅ Basic dependencies OK')
                    except Exception as e:
                        print(f'❌ Dependencies failed: {e}')
                        exit 1
                    "
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image..."
                script {
                    docker.build("${DOCKER_IMAGE}")
                }
            }
        }

        stage('Simple Container Test') {
            steps {
                echo "🧪 Testing container basics..."
                script {
                    sh """
                    # Test that container can start
                    docker run --rm ${DOCKER_IMAGE} python3 -c "
                    print('Container test passed')
                    "
                    """
                }
            }
        }

        stage('Deploy to Registry') {
            steps {
                echo "🚀 Deploying to registry..."
                script {
                    docker.withRegistry("http://${DOCKER_REGISTRY}") {
                        docker.image("${DOCKER_IMAGE}").push()
                    }
                }
            }
        }
    }

    post {
        success {
            echo "🎉 Pipeline completed successfully!"
            echo "Service URL: http://10.90.90.100:${STUDENT_PORT}"
        }
        failure {
            echo "💥 Pipeline failed."
        }
        always {
            echo "🧹 Cleaning up..."
            sh 'docker rm -f test-container 2>/dev/null || true'
        }
    }
}