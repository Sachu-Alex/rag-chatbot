#!/bin/bash
# Launch script for RAG ChatBot Streamlit application

echo "🤖 Starting RAG ChatBot System..."

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment (try different names)
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "rag_env" ]; then
    echo "📦 Activating virtual environment (rag_env)..."
    source rag_env/bin/activate
elif [ -d "venv" ]; then
    echo "📦 Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Please create one first."
    exit 1
fi

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "🚀 Starting Streamlit application..."
echo "📍 Open your browser to: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the application"
echo ""

# Run the application
streamlit run ui/streamlit_app.py