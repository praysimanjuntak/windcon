# Environment for the spiral-fit experiment on the GPU machine. The scripts in this folder source it
# from $HOME/Documents/postel/vesuvius_challenge/env.sh; keep every cache inside that folder.
export POSTEL="$HOME/Documents/postel/vesuvius_challenge"
export PATH="$POSTEL/bin:$PATH"
export UV_CACHE_DIR="$POSTEL/cache/uv"
export UV_PYTHON_INSTALL_DIR="$POSTEL/cache/uv-python"
export UV_TOOL_DIR="$POSTEL/cache/uv-tools"
export UV_TOOL_BIN_DIR="$POSTEL/bin"
export XDG_CACHE_HOME="$POSTEL/cache/xdg"
export TRITON_CACHE_DIR="$POSTEL/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$POSTEL/cache/inductor"
export CUPY_CACHE_DIR="$POSTEL/cache/cupy"
export NUMBA_CACHE_DIR="$POSTEL/cache/numba"
export HF_HOME="$POSTEL/cache/hf"
export MPLCONFIGDIR="$POSTEL/cache/mpl"
export WANDB_MODE=disabled
export WANDB_DIR="$POSTEL/cache/wandb"
export FIT_SPIRAL_CACHE_DIR="$POSTEL/cache/spiral"
