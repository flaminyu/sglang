#!/bin/bash
# =============================================================================
# SGLang Continuum 环境配置脚本
# =============================================================================
# 功能: 自动检测项目根目录并生成 .env 文件
# 用法: ./setup_env.sh
#       source .env  # 加载环境变量

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

echo "=========================================="
echo "SGLang Continuum 环境配置"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 生成 .env 文件
cat > "$ENV_FILE" << 'EOF'
# SGLang Continuum 环境变量配置
# 由 setup_env.sh 自动生成
# 生成时间: GENERATION_TIME_PLACEHOLDER

# ============================================================================
# 项目配置
# ============================================================================
export SGLANG_PROJECT_ROOT="PROJECT_ROOT_PLACEHOLDER"

# ============================================================================
# Simulator 配置
# ============================================================================
export SGLANG_SIMULATOR_CONFIG_PATH="${SGLANG_PROJECT_ROOT}/tools/sglang-simulator/test/assets/config.json"
export SGLANG_SIMULATOR_OUTPUT_DIR="${SGLANG_PROJECT_ROOT}/output/simulator"
export SGLANG_SIMULATOR_OUTPUT_MODE="OFFLINE"
export SGLANG_SIMULATOR_NUM_WARMUP=0

# ============================================================================
# Python 路径配置
# ============================================================================
export PYTHONPATH="${SGLANG_PROJECT_ROOT}/tools/sglang-simulator/src:${SGLANG_PROJECT_ROOT}/python:${PYTHONPATH:-}"

# ============================================================================
# Continuum TTL 配置
# ============================================================================
export SGLANG_CONTINUUM_TTL_DEFAULT_SEC=0
export SGLANG_CONTINUUM_TTL_MIN_SEC=0
export SGLANG_CONTINUUM_TTL_MAX_SEC=90
export SGLANG_CONTINUUM_TTL_HISTORY_MAXLEN=4096
export SGLANG_CONTINUUM_TTL_HISTORY_THRESHOLD=20
export SGLANG_CONTINUUM_TTL_TOOL_DELAY_SEC=0
export SGLANG_CONTINUUM_TTL_TOOL_DELAY_RATIO=1.5
export SGLANG_CONTINUUM_MEMORY_PRESSURE_PENALTY=0.3

# ============================================================================
# SGLang 服务器配置
# ============================================================================
export SGLANG_PORT=30000
export SGLANG_HOST_IP=127.0.0.1

# ============================================================================
# GPU 配置
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
EOF

# 替换占位符
sed -i "s|GENERATION_TIME_PLACEHOLDER|$(date '+%Y-%m-%d %H:%M:%S')|g" "$ENV_FILE"
sed -i "s|PROJECT_ROOT_PLACEHOLDER|${PROJECT_ROOT}|g" "$ENV_FILE"

echo "[OK] 环境配置文件已生成: $ENV_FILE"
echo ""
echo "下一步:"
echo "  1. source .env              # 加载环境变量"
echo "  2. ./build_sglang.sh       # 编译 sglang (可选)"
echo "  3. python test_standalone.py  # 运行测试"
echo ""
echo "=========================================="

# 自动加载环境变量
source "$ENV_FILE"
echo "[OK] 环境变量已加载"
echo "PYTHONPATH: ${PYTHONPATH:0:80}..."
echo "SGLANG_PROJECT_ROOT: $SGLANG_PROJECT_ROOT"
