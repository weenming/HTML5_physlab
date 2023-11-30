#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py

