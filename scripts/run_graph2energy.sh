#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT_DIR/main.py" --task graph2energy --config "$ROOT_DIR/configs/graph2energy.minimal.json" "$@"
