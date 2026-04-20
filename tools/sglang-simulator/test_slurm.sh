#!/bin/bash
# =============================================================================
# SGLang Simulator + TTL 测试脚本
# =============================================================================

#SBATCH --job-name=simulator-ttl-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=${SLURM_LOG_DIR:-/tmp/slurm}/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR:-/tmp/slurm}/%x_%j.err

# 自动检测项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SGLANG_PROJECT_ROOT:-$SCRIPT_DIR}"
LOG_DIR="${SGLANG_LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "$LOG_DIR" "$PROJECT_DIR"
cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}/tools/sglang-simulator/src:${PROJECT_DIR}/python:${PYTHONPATH:-}"

echo "==================================================================="
echo "SGLang Simulator + TTL Test $(date)"
echo "==================================================================="
echo "Project: $PROJECT_DIR"
echo "Python Path: ${PYTHONPATH:0:80}..."

# 检测 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda Environment: $CONDA_DEFAULT_ENV"
else
    echo "[INFO] No conda environment detected, using system Python"
fi

echo ""
echo "[1] 检查环境..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "[2] 测试 simulator 模块..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/tools/sglang-simulator/src')
from sglang_simulator import simulation
from sglang_simulator.hook import install_class_hooks
from sglang_simulator.simulation.sglang import C_SchedulerHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
print('All simulator modules imported successfully!')
"
TEST2_RESULT=$?

echo ""
echo "[3] 验证 TTL 机制..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/python')
from sglang.srt.managers.scheduler import Scheduler
ttl_methods = ['pin_request_kv', 'unpin_request_kv', 'cleanup_expired_pins', 'init_continuum_pin']
for m in ttl_methods:
    assert hasattr(Scheduler, m), 'Missing TTL method: ' + m
print('All TTL methods verified in Scheduler!')
"
TEST3_RESULT=$?

echo ""
echo "[4] 测试 simulator hooks 安装..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/tools/sglang-simulator/src')
sys.path.insert(0, '${PROJECT_DIR}/python')
from sglang_simulator.hook import install_class_hooks
from sglang_simulator.simulation.sglang import scheduler, model_runner, cache_controller, hiradix_cache

install_class_hooks([
    scheduler.C_SchedulerHook,
    model_runner.C_ModelRunnerHook,
    cache_controller.C_HiCacheController,
    hiradix_cache.C_HiRadixCacheHook,
])
print('All simulator hooks installed successfully!')
"
TEST4_RESULT=$?

echo ""
echo "[5] 测试 launch_server --help..."
python3 -m sglang_simulator.simulation.sglang.launch_server --help 2>&1 | head -5
TEST5_RESULT=$?

echo ""
echo "==================================================================="
echo "测试结果汇总:"
echo "  [2] Simulator 模块: $([ $TEST2_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  [3] TTL 机制: $([ $TEST3_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  [4] Hooks 安装: $([ $TEST4_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  [5] Launch Server: $([ $TEST5_RESULT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "==================================================================="

if [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ] && [ $TEST4_RESULT -eq 0 ]; then
    echo "Simulator + TTL 基本测试完成!"
    exit 0
else
    echo "部分测试失败，请检查日志"
    exit 1
fi
