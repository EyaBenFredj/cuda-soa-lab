@echo off
echo ========================================
echo    Starting Federated Learning Services
echo ========================================
echo.

echo 🚀 Building and starting services with Docker...
docker compose up --build

if %errorlevel% neq 0 (
    echo.
    echo ❌ Docker failed! Starting services manually...
    echo.
    echo Starting Client1 on port 8001...
    start "Client1" cmd /k "python client1.py"
    timeout /t 3 /nobreak >nul
    
    echo Starting Client2 on port 8002...
    start "Client2" cmd /k "python client2.py"
    timeout /t 3 /nobreak >nul
    
    echo Starting Client3 on port 8003...
    start "Client3" cmd /k "python client3.py"
    timeout /t 3 /nobreak >nul
    
    echo Starting Aggregator on port 9000...
    start "Aggregator" cmd /k "python aggregator.py"
    
    echo.
    echo ✅ All services started manually!
    echo 🧪 Open a NEW Command Prompt and run: python test_services.py
)

pause