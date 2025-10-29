#!/bin/bash

if [ ! -f "backend/main.py" ]; then
    echo "run this script from the project root directory"
    exit 1
fi

check_venv() {
    if [ ! -d "backend/venv" ]; then
        echo "Creating Python virtual environment..."
        if ! command -v python3 &> /dev/null; then
            echo "Error: python3 is not installed. Please install Python 3 to continue."
            exit 1
        fi
        cd backend
        python3 -m venv venv
        cd ..
        echo "Virtual environment created successfully."
    fi
}

setup_backend() {
    echo "Setting up backend dependencies..."
    check_venv
    cd backend
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    echo "Backend dependencies installed"
}

setup_frontend() {
    echo "Setting up frontend dependencies..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    cd ..
    echo "Frontend dependencies installed"
}

run_app() {
    echo " Starting the application..."
    echo ""
    echo " Instructions:"
    echo "1. Backend will run on: http://localhost:8000"
    echo "2. Frontend will run on: http://localhost:5173"
    echo "3. Open http://localhost:5173 in your browser"
    echo "4. Press Ctrl+C to stop both servers"
    echo ""
    
    cd backend
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    cd ..
    
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    echo "Press Ctrl+C to stop the servers..."
    trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
    wait
}

main() {
    setup_backend
    setup_frontend
    
    echo ""
    echo "Setup complete"

    run_app
}

main
