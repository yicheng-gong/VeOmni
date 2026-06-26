#!/bin/bash
set -e

echo "=========================================="
echo "馃殌 TorchCodec One-Click Installation for Ascend NPU (Enhanced Version)"
echo "=========================================="
echo "馃搷 Working Directory: $(pwd)"
echo "馃悕 Python: $(python --version)"
echo "=========================================="

# ==========================
# 馃攳 銆怤ew銆慒orce check if Python is a shared library (must be .so)
# ==========================
check_python_shared() {
    echo "[Pre-check] Checking if Python is a shared library version..."

    PYTHON_BIN=$(which python)
    PYTHON_LIB=$(python -c "
import sysconfig
import os
lib = sysconfig.get_config_var('LIBDIR')
ver = sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION')
print(os.path.join(lib, f'libpython{ver}.so'))
")

    if [ -f "$PYTHON_LIB" ]; then
        echo "   鉁?Python is a shared library version: $PYTHON_LIB"
        echo "   Can compile TorchCodec extensions normally~"
    else
        echo "=========================================================="
        echo "   鉂?Error: Current Python is a [static library version], cannot compile C++ extensions!"
        echo "   馃攳 Dynamic library not found: $PYTHON_LIB"
        echo "=========================================================="
        echo "   馃挕 Solutions:"
        echo "      1. Recompile Python with parameter: ./configure --enable-shared"
        echo "      2. Must execute after installation: ldconfig"
        echo "      3. Confirm the existence of file: libpython3.11.so"
        echo "   馃挰 All deep learning / PyTorch / Ascend environments require dynamic library Python!"
        echo "=========================================================="
        exit 1
    fi
}

# Execute check
check_python_shared

# --- Accept CANN environment script path from command line ---
if [ $# -lt 1 ]; then
    echo "鉂?Usage: $0 <CANN set_env.sh path>"
    echo "Example: $0 /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi
ASCEND_ENV="$1"

# --- Self-check ---
if [ ! -f "pyproject.toml" ] || [ ! -d "src/torchcodec" ]; then
    echo "鉂?Error: Please run this script in the torchcodec source root directory!"
    exit 1
fi

# ==========================================
# 馃敡 Fully automatic installation of complete FFmpeg (complement all dependencies)
# ==========================================
echo "[0/4] Checking and installing FFmpeg development dependencies..."
install_ffmpeg() {
    if command -v yum &> /dev/null; then
        echo "   馃敡 Installing with yum..."
        yum install -y ffmpeg ffmpeg-devel --nogpgcheck
    elif command -v apt &> /dev/null; then
        echo "   馃敡 Installing with apt..."
        apt update -y
        apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev
    else
        echo "   鉂?Error: yum / apt package manager not found!"
        exit 1
    fi
}

# Check if key libraries are missing
if ! pkg-config --exists libavdevice libavfilter 2>/dev/null; then
    echo "   鈿狅笍 Missing libavdevice / libavfilter, automatically installing..."
    install_ffmpeg
else
    echo "   鉁?Complete FFmpeg development packages already exist"
fi

# --- 1. Python dependencies (force root without warnings) ---
echo "[1/4] Installing Python build tools..."
pip install --quiet --upgrade pip --root-user-action=ignore
pip install --quiet pybind11 wheel setuptools cmake ninja --root-user-action=ignore

# --- 2. Load Ascend environment ---
echo "[2/4] Loading Ascend CANN environment..."
if [ -f "$ASCEND_ENV" ]; then
    source "$ASCEND_ENV"
    echo "   鉁?Loaded: $ASCEND_ENV"
else
    echo "鉂?CANN environment not found: $ASCEND_ENV"
    exit 1
fi

# --- 3. Auto search FFmpeg path ---
echo "[3/4] Configuring FFmpeg and compilation environment variables..."
FFMPEG_PC_PATH=$(find /usr /usr/local /opt -name "libavcodec.pc" 2>/dev/null | head -n 1)
if [ -z "$FFMPEG_PC_PATH" ]; then
    echo "   鉂?Error: FFmpeg development files not found!"
    exit 1
fi

PC_DIR=$(dirname "$FFMPEG_PC_PATH")
export PKG_CONFIG_PATH="$PC_DIR:$PKG_CONFIG_PATH"
echo "   鉁?FFmpeg pkg-config path: $PC_DIR"

# Verify all libraries
FFMPEG_VER=$(pkg-config --modversion libavcodec)
echo "   鉁?FFmpeg identified successfully (version: $FFMPEG_VER)"

# Compilation environment variables
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())"):$CMAKE_PREFIX_PATH
export LIBRARY_PATH=/usr/local/lib:/usr/lib/$(uname -m)-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64:$LD_LIBRARY_PATH

# --- 4. Compile and install ---
echo "[4/4] Cleaning and compiling installation..."
rm -rf build/ dist/ *.egg-info src/torchcodec.egg-info

pip install -e . --no-build-isolation --root-user-action=ignore

echo "=========================================="
echo "馃帀 Installation successful!"
echo "=========================================="
echo "Verification commands:"
echo "source $ASCEND_ENV"
echo "python -c \"from torchcodec.decoders import VideoDecoder; print('install success')\""
