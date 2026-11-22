#!/usr/bin/env bash
set -e

# ClaimClarity AI Startup Script
echo "============================================================"
echo " Starting ClaimClarity AI - Insurance Photo Review Assistant"
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check if .venv exists, otherwise create it
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment in .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Ensure sample images exist
echo "Checking synthetic sample images..."
python generate_samples.py

# Launch server
PORT="${PORT:-5001}"
echo "------------------------------------------------------------"
echo "ClaimClarity AI is starting on: http://127.0.0.1:${PORT}"
echo "Open your browser to test photo upload, preview, and comparison."
echo "Press Ctrl+C to stop the server."
echo "------------------------------------------------------------"
exec python app.py
