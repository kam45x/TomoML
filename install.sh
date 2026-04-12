#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Install pip packages
cd "$SCRIPT_DIR"
pip install -r requirements.txt

# 2. Install custom dival library
cd "$SCRIPT_DIR/.."
if [ ! -d dival ]; then
    git clone https://github.com/kam45x/dival.git
fi
cd dival
pip install -e .
