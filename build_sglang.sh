#!/bin/bash
# =============================================================================
# SGLang 编译脚本
# =============================================================================
# 功能: 编译安装 sglang 和 sgl-kernel
# 用法: ./build_sglang.sh [--clean] [--reinstall]
# =============================================================================

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SGLANG_PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_CMD="${PYTHON_BIN:-$(which python3)}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 解析参数
CLEAN=false
REINSTALL=false
FORCE_CUDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --reinstall)
            REINSTALL=true
            shift
            ;;
        --force-cuda)
            FORCE_CUDA=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --clean       清理旧构建文件"
            echo "  --reinstall   强制重新安装 (清除 build 缓存)"
            echo "  --force-cuda  强制使用 CUDA 扩展"
            echo "  -h, --help    显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "SGLang 编译安装脚本"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "Python: $PYTHON_CMD"
echo "Python 版本: $($PYTHON_CMD --version)"
echo ""

# ============================================================================
# 环境检查
# ============================================================================

log_info "检查环境..."

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU 检测:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  - $line"
    done
else
    log_warn "未检测到 GPU (将安装 CPU 版本)"
fi

# 检查 CUDA
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
log_info "CUDA 版本: $CUDA_VERSION"

# 加载环境变量
if [ -f "${PROJECT_ROOT}/.env" ]; then
    log_info "加载 .env 文件..."
    source "${PROJECT_ROOT}/.env"
fi

# ============================================================================
# 清理旧构建
# ============================================================================

if [ "$CLEAN" = true ]; then
    log_info "清理旧构建文件..."
    cd "$PROJECT_ROOT"

    # 清理 sglang
    if [ -d "python/build" ]; then
        rm -rf python/build python/dist python/*.egg-info
        log_info "  - 清理 sglang build/"
    fi

    # 清理 sgl-kernel
    if [ -d "sgl-kernel/build" ]; then
        rm -rf sgl-kernel/build sgl-kernel/dist sgl-kernel/*.egg-info
        log_info "  - 清理 sgl-kernel build/"
    fi
fi

# ============================================================================
# 编译安装 sgl-kernel
# ============================================================================

cd "$PROJECT_ROOT"

if [ -d "sgl-kernel" ]; then
    log_info "编译安装 sgl-kernel..."
    cd sgl-kernel

    # 检查是否有预编译的 .so 文件
    if [ -f "python/sgl_kernel/sm100/common_ops.abi3.so" ]; then
        log_info "  检测到预编译 sgl-kernel，跳过编译..."
    else
        log_info "  开始编译 (可能需要 10-30 分钟)..."
        $PYTHON_CMD -m pip install . --no-build-isolation 2>&1 | tail -20
    fi

    # 验证安装
    $PYTHON_CMD -c "from sgl_kernel import common_ops; print('  sgl-kernel 加载:', common_ops.__file__)" || log_error "  sgl-kernel 安装失败"

    cd "$PROJECT_ROOT"
else
    log_warn "sgl-kernel 目录不存在，跳过"
fi

# ============================================================================
# 编译安装 sglang
# ============================================================================

if [ -d "python" ]; then
    log_info "编译安装 sglang..."
    cd python

    # 清理
    if [ "$REINSTALL" = true ]; then
        rm -rf build/ dist/ *.egg-info
        log_info "  已清理旧构建"
    fi

    # 安装 (开发模式)
    log_info "  安装 sglang (开发模式)..."
    $PYTHON_CMD -m pip install -e . --no-build-isolation 2>&1 | tail -20

    # 验证安装
    log_info "  验证安装..."
    $PYTHON_CMD -c "import sglang; print('  sglang 版本:', sglang.__version__)" 2>/dev/null || \
    $PYTHON_CMD -c "import sglang; print('  sglang 路径:', sglang.__file__)" || \
    log_error "  sglang 安装失败"

    cd "$PROJECT_ROOT"
else
    log_error "sglang 目录不存在"
    exit 1
fi

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "=========================================="
log_info "编译完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "  source ${PROJECT_ROOT}/.env      # 加载环境变量"
echo "  python tools/sglang-simulator/test_standalone.py  # 运行测试"
echo ""
