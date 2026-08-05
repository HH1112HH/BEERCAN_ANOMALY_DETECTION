#!/bin/bash
echo "========================================"
echo "  Industrial Defect Detection System"
echo "========================================"
echo ""

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

export PATH="$HOME/.local/bin:$PATH"

if ! python3 -m pip --version &>/dev/null; then
    echo "Installing pip..."
    python3 /tmp/get-pip.py --user --break-system-packages &>/dev/null
fi

echo "Checking dependencies..."
python3 -m pip install --user --break-system-packages -q -r requirements.txt 2>/dev/null

echo ""
echo "--- GPU Status ---"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} | VRAM: {props.total_memory/1024**3:.1f}GB | CUDA: {torch.version.cuda}')
else:
    print('  CPU mode (no CUDA GPU detected)')
"

echo ""
echo "========================================"
echo "  Starting server on http://0.0.0.0:5004"
echo "  Dashboard: http://localhost:5004"
echo "  Video:     http://localhost:5004/video"
echo "  Status:    http://localhost:5004/status"
echo "========================================"
echo ""

python3 main.py
