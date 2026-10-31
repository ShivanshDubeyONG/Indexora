#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

echo "🐍 Installing Python dependencies..."
pip install -e ".[dev,benchmark,api]"

echo "⚛️ Building Frontend with Vite..."
if [ -d "frontend" ]; then
  cd frontend
  npm install
  npm run build
  cd ..
fi

echo "✅ Indexora build complete and ready for production deployment!"
