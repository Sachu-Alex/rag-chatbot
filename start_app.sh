#!/bin/bash
#
# RAG ChatBot Application Launcher
# 
# This script provides a convenient way to start the RAG ChatBot system with proper
# environment setup and error handling. It automatically:
# 1. Locates and activates the Python virtual environment
# 2. Sets up the Python module path for imports
# 3. Launches the Streamlit web application
# 4. Provides helpful user guidance
#
# Usage: ./start_app.sh
# Requirements: Python virtual environment with dependencies installed
#

set -e  # Exit on any error

echo "🤖 Starting RAG ChatBot System..."
echo "📋 Initializing environment..."

# Change to the project directory (handles script execution from any location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "📁 Working directory: $SCRIPT_DIR"

# Activate virtual environment with fallback options
# Priority order: .venv -> venv -> rag_env (common naming conventions)
VENV_ACTIVATED=false

if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment (.venv)..."
    source .venv/bin/activate
    VENV_ACTIVATED=true
elif [ -d "venv" ]; then
    echo "📦 Activating virtual environment (venv)..."
    source venv/bin/activate
    VENV_ACTIVATED=true
elif [ -d "rag_env" ]; then
    echo "📦 Activating virtual environment (rag_env)..."
    source rag_env/bin/activate
    VENV_ACTIVATED=true
else
    echo "❌ No virtual environment found!"
    echo "📝 Please create a virtual environment first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements_advanced.txt"
    exit 1
fi

# Verify virtual environment activation
if [ "$VENV_ACTIVATED" = true ]; then
    echo "✅ Virtual environment activated successfully"
    echo "🐍 Python: $(which python)"
    echo "📦 Pip: $(which pip)"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Set Python module path to include current directory for local imports
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "🔧 Python path configured: $PYTHONPATH"

# Verify required dependencies
echo "🔍 Checking dependencies..."
if ! python -c "import streamlit, langchain, chromadb" 2>/dev/null; then
    echo "❌ Missing required dependencies!"
    echo "📦 Please install them with: pip install -r requirements_advanced.txt"
    exit 1
fi
echo "✅ Dependencies verified"

# Start the Streamlit application
echo ""
echo "🚀 Launching Streamlit application..."
echo "🌐 Application will be available at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the application"
echo "📊 Loading RAG ChatBot interface..."
echo ""

# Run the main application with error handling
if ! streamlit run ui/streamlit_app.py; then
    echo ""
    echo "❌ Application failed to start!"
    echo "🔍 Check the error messages above for troubleshooting"
    exit 1
fi