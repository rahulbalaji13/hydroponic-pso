#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Hydroponic Agriculture ML app locally..."
echo "Backend API + frontend dashboard will be available at http://127.0.0.1:5000"

cd "$ROOT_DIR/backend"
python flask_app.py
