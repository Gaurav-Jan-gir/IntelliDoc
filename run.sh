#!/usr/bin/env bash
# ==============================================================================
# IntelliDoc Run Script
# Launches both the FastAPI backend and Streamlit frontend applications.
# Gracefully terminates all background processes upon exit (Ctrl+C / SIGINT / SIGTERM).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================================="
echo "           Starting IntelliDoc Platform                   "
echo "=========================================================="

# 1. Environment Verification
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[INFO] .env not found. Copying .env.example to .env..."
        cp .env.example .env
        echo "[WARNING] Please edit .env to set your GOOGLE_API_KEY for Gemini LLM generation."
    else
        echo "[WARNING] No .env file found. Using default configurations."
    fi
fi

# 2. Check Python executable
PYTHON_CMD="python3"
if [ -d ".venv" ]; then
    echo "[INFO] Using virtual environment in .venv/"
    PYTHON_CMD=".venv/bin/python"
elif [ -d "venv" ]; then
    echo "[INFO] Using virtual environment in venv/"
    PYTHON_CMD="venv/bin/python"
fi

# 3. Create required directories
mkdir -p data vectorstore

# 4. Cleanup Trap
cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping IntelliDoc backend and frontend services..."
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "  - Stopping FastAPI Backend (PID: ${BACKEND_PID})..."
        kill -TERM "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "  - Stopping Streamlit Frontend (PID: ${FRONTEND_PID})...."
        kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "[SHUTDOWN] All services stopped. Goodbye!"
    exit 0
}

trap cleanup INT TERM EXIT

# 5. Start FastAPI Backend (Uvicorn)
echo "[STARTUP] Launching FastAPI Backend on http://127.0.0.1:8000..."
${PYTHON_CMD} -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait briefly for backend to initialize
sleep 2

# 6. Start Streamlit Frontend
echo "[STARTUP] Launching Streamlit Frontend on http://127.0.0.1:8501..."
${PYTHON_CMD} -m streamlit run frontend/app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo " IntelliDoc is now running!"
echo "   - API Documentation: http://127.0.0.1:8000/docs"
echo "   - Web Interface:     http://127.0.0.1:8501"
echo " Press Ctrl+C to terminate both servers."
echo "=========================================================="
echo ""

# Keep script running and wait for background jobs
wait "${BACKEND_PID}" "${FRONTEND_PID}"

