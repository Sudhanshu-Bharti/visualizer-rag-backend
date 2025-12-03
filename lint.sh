#!/bin/bash
# Backend linting script
# Usage: ./lint.sh or bash lint.sh

echo "Backend Linting Script"
echo "======================"

# Check if we're in the backend directory
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: Please run this script from the backend directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    
    # Handle different activation scripts for different platforms
    if [ -f "venv/Scripts/activate" ]; then
        # Windows
        source venv/Scripts/activate
    elif [ -f "venv/bin/activate" ]; then
        # Linux/Mac
        source venv/bin/activate
    else
        echo "WARNING: Virtual environment found but activation script not found"
    fi
else
    echo "WARNING: No virtual environment found (venv directory missing)"
fi

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    echo "ERROR: ruff not found. Installing..."
    pip install ruff
fi

# Run ruff check
echo "Running ruff check..."
ruff check .

if [ $? -eq 0 ]; then
    echo "✅ All linting checks passed!"
    exit 0
else
    echo "❌ Linting failed!"
    exit 1
fi