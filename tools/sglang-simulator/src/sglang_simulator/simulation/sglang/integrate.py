#!/usr/bin/env python3
"""
SGLang Real Integration Module - 真实 SGLang 集成

提供与真实 SGLang 服务器的集成，支持:
1. HTTP API 调用真实 SGLang 服务器
2. 本地模拟器模式 (CPU-based)
3. Continuum TTL 策略集成
4. 官方 benchmark 兼容

Usage:
    # 方式1: 直接运行模拟器
    python -m sglang_simulator.simulation.sglang.integrate \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --simulation-mode offline \
        --dataset-name agent_traces \
        --dataset-path ./agent_traces.json

    # 方式2: 调用真实服务器
    python -m sglang_simulator.simulation.sglang.integrate \
        --backend-url http://localhost:30000 \
        --simulation-mode http \
        --benchmark-mode

    # 方式3: 运行官方 benchmark
    python -m sglang_simulator.simulation.sglang.integrate \
        --benchmark-mode \
        --backend sglang \
        --num-prompts 1000
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# Add simulator to path
SGLANG_SIM_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(SGLANG_SIM_PATH))

import aiohttp
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SimulationConfig:
    """仿真配置"""
    # 模式
    simulation_mode: str = "offline"  # offline, blocking, http
    backend_url: str = "http://localhost:30000"

    # 模型配置
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_path: Optional[str] = None
    dtype: str = "half"

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 30000
    max_running_requests: int = 100
    mem_fraction_static: float = 0.9

    # Continuum TTL 配置
    continuum_enabled: bool = True
    continuum_ttl_default: float = 3.0
    continuum_ttl_min: float = 0.1
    continuum_ttl_max: float = 10.0
    continuum_ttl_history: int = 10
    continuum_ttl_penalty: float = 0.5

    # 缓存配置
    max_total_tokens: int = 100000
    stream_interval: int = 1

    # 输出
    output_dir: str = "/tmp/sglang_simulator/output"
    enable_profile: bool = True

    def to_env(self) -> Dict[str, str]:
        """转换为环境变量"""
        return {
            "SGLANG_CONTINUUM_TTL_ENABLED": "1" if self.continuum_enabled else "0",
            "SGLANG_CONTINUUM_TTL_DEFAULT_SEC": str(self.continuum_ttl_default),
            "SGLANG_CONTINUUM_TTL_MIN_SEC": str(self.continuum_ttl_min),
            "SGLANG_CONTINUUM_TTL_MAX_SEC": str(self.continuum_ttl_max),
            "SGLANG_CONTINUUM_TTL_HISTORY_THRESHOLD": str(self.continuum_ttl_history),
            "SGLANG_CONTINUUM_MEMORY_PRESSURE_PENALTY": str(self.continuum_ttl_penalty),
            "SGLANG_SIMULATOR_OUTPUT_DIR": self.output_dir,
            "HISIM_SIMULATION_MODE": self.simulation_mode.upper(),
        }

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        """从环境变量创建配置"""
        return cls(
            continuum_enabled=os.getenv("SGLANG_CONTINUUM_TTL_ENABLED", "1") == "1",
            continuum_ttl_default=float(os.getenv("SGLANG_CONTINUUM_TTL_DEFAULT_SEC", "3.0")),
            continuum_ttl_min=float(os.getenv("SGLANG_CONTINUUM_TTL_MIN_SEC", "0.1")),
            continuum_ttl_max=float(os.getenv("SGLANG_CONTINUUM_TTL_MAX_SEC", "10.0")),
            continuum_ttl_history=int(os.getenv("SGLANG_CONTINUUM_TTL_HISTORY_THRESHOLD", "10")),
            continuum_ttl_penalty=float(os.getenv("SGLANG_CONTINUUM_MEMORY_PRESSURE_PENALTY", "0.5")),
            output_dir=os.getenv("SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator/output"),
            simulation_mode=os.getenv("HISIM_SIMULATION_MODE", "OFFLINE").lower(),
        )


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""
    # 请求配置
    num_prompts: int = 1000
    request_rate: float = float("inf")  # infinite = all at once
    random_input: int = 1024
    random_output: int = 1024
    random_range_ratio: float = 0.5

    # 数据集
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    ignore_timestamp: bool = False

    # 预热
    num_warmup: int = 10

    # 输出
    save_result: bool = True
    result_dir: str = "/tmp/benchmark_results"


# ============================================================================
# Agent Trace Data Structures
# ============================================================================

@dataclass
class AgentTurn:
    """Agent 轮次"""
    turn_index: int
    input_tokens: int
    output_tokens: int
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_duration: float = 0.0


@dataclass
class AgentProgram:
    """Agent 程序"""
    program_id: str
    arrival_time: float
    turns: List[AgentTurn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "arrival_time": self.arrival_time,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "is_tool_call": t.is_tool_call,
                    "tool_name": t.tool_name,
                    "tool_duration": t.tool_duration,
                }
                for t in self.turns
            ],
        }


# ============================================================================
# Request Generator
# ============================================================================

class AgentTraceRequestGenerator:
    """
    Agent Trace 请求生成器

    从 agent traces 生成模拟请求
    """

    def __init__(
        self,
        config: SimulationConfig,
        benchmark_config: BenchmarkConfig,
    ):
        self.config = config
        self.benchmark_config = benchmark_config
        self.programs: List[AgentProgram] = []

    def generate_programs(
        self,
        num_programs: int = 20,
        turns_per_program: int = 7,
        target_tokens: int = 20000,
        arrival_rate: float = 0.3,
        tool_ratio: float = 0.7,
        tool_duration_mean: float = 1.0,
        tool_duration_std: float = 0.5,
        seed: int = 42,
    ) -> List[AgentProgram]:
        """生成模拟程序"""
        np.random.seed(seed)
        self.programs = []

        arrival_time = 0.0
        for p in range(num_programs):
            # 生成到达时间 (Poisson 过程)
            if p > 0:
                arrival_time += np.random.exponential(1.0 / arrival_rate)

            program = AgentProgram(
                program_id=f"prog_{p}",
                arrival_time=arrival_time,
                turns=[],
            )

            input_tokens = 500  # 初始输入
            for t in range(turns_per_program):
                is_tool_call = t > 0 and np.random.random() < tool_ratio
                tool_name = f"tool_{t % 3}" if is_tool_call else None
                tool_duration = (
                    max(0.1, np.random.normal(tool_duration_mean, tool_duration_std))
                    if is_tool_call else 0.0
                )

                program.turns.append(AgentTurn(
                    turn_index=t,
                    input_tokens=input_tokens,
                    output_tokens=50 + np.random.randint(0, 50),
                    is_tool_call=is_tool_call,
                    tool_name=tool_name,
                    tool_duration=tool_duration,
                ))

                input_tokens += 100  # 每轮输入增长

            self.programs.append(program)

        return self.programs

    def generate_requests(self) -> List[Dict[str, Any]]:
        """生成请求列表"""
        requests = []
        request_id = 0

        for program in self.programs:
            for turn in program.turns:
                requests.append({
                    "request_id": f"req_{request_id}",
                    "program_id": program.program_id,
                    "turn_index": turn.turn_index,
                    "created_time": program.arrival_time + turn.turn_index * 1.0,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "is_tool_call": turn.is_tool_call,
                    "tool_name": turn.tool_name,
                    "tool_duration": turn.tool_duration,
                    "prompt": f"[Program {program.program_id}] Turn {turn.turn_index}",  # Placeholder
                })
                request_id += 1

        return requests

    def to_simulation_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 simulation 参数"""
        return {
            "program_id": request["program_id"],
            "turn_index": request["turn_index"],
            "created_time": request["created_time"],
            "total_request": len(self.generate_requests()),
            "is_tool_call": request["is_tool_call"],
            "tool_name": request["tool_name"],
            "tool_duration": request["tool_duration"],
        }


# ============================================================================
# HTTP Client
# ============================================================================

class SGLangHTTPClient:
    """
    SGLang HTTP 客户端

    用于与真实 SGLang 服务器交互
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def generate(
        self,
        prompt: str,
        sampling_params: Dict[str, Any],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送生成请求"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            **sampling_params,
        }

        if extra_body:
            payload["extra_body"] = extra_body

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Request failed: {resp.status} - {error_text}")
                return await resp.json()

    async def benchmark(
        self,
        requests: List[Dict[str, Any]],
        request_rate: float = float("inf"),
        ignore_timestamp: bool = False,
    ) -> List[Dict[str, Any]]:
        """运行 benchmark"""
        results = []
        start_time = time.time()
        pending_tasks = []

        for i, req in enumerate(requests):
            # 计算请求到达时间
            if ignore_timestamp:
                delay = i / request_rate if request_rate != float("inf") else 0
            else:
                delay = req.get("created_time", 0) - (time.time() - start_time)
                delay = max(0, delay)

            if delay > 0:
                await asyncio.sleep(delay)

            # 发送请求
            sampling_params = {
                "max_tokens": req.get("output_tokens", 100),
                "temperature": 0.0,
            }

            extra_body = {
                "conversation_id": req.get("program_id", "unknown"),
                "is_tool_call": req.get("is_tool_call", False),
                "tool_name": req.get("tool_name"),
                "tool_duration": req.get("tool_duration", 0),
            }

            task = asyncio.create_task(
                self.generate(req["prompt"], sampling_params, extra_body)
            )
            pending_tasks.append((req, task))

            # 如果请求率有限制，等待
            if request_rate != float("inf") and i > 0:
                await asyncio.sleep(1.0 / request_rate)

        # 收集结果
        for req, task in pending_tasks:
            try:
                result = await task
                results.append({
                    "request": req,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "request": req,
                    "success": False,
                    "error": str(e),
                })

        return results


# ============================================================================
# Simulator Launcher
# ============================================================================

class SimulatorLauncher:
    """
    模拟器启动器

    启动 SGLang 模拟器并运行 benchmark
    """

    def __init__(
        self,
        config: SimulationConfig,
        benchmark_config: BenchmarkConfig,
    ):
        self.config = config
        self.benchmark_config = benchmark_config

        # 设置环境变量
        for key, value in config.to_env().items():
            os.environ[key] = value

        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)

    def run_offline_simulation(self) -> Dict[str, Any]:
        """运行离线模拟"""
        print("=" * 70)
        print("Running offline simulation")
        print("=" * 70)
        print(f"Model: {self.config.model_path}")
        print(f"Continuum TTL: {'enabled' if self.config.continuum_enabled else 'disabled'}")
        print(f"Output: {self.config.output_dir}")

        # 生成请求
        generator = AgentTraceRequestGenerator(self.config, self.benchmark_config)
        programs = generator.generate_programs(
            num_programs=20,
            turns_per_program=7,
            target_tokens=20000,
            arrival_rate=0.3,
            tool_ratio=0.7,
        )
        requests = generator.generate_requests()

        print(f"\nGenerated {len(programs)} programs with {len(requests)} total requests")

        # 构建模拟器参数
        simulation_args = {
            "simulation_mode": "offline",
            "total_requests": len(requests),
            "programs": [p.to_dict() for p in programs],
        }

        # 运行模拟器
        try:
            # 导入模拟器模块
            from sglang_simulator.simulation.sglang import bench_runner

            # 创建 benchmark runner
            server_args = self._create_server_args()
            runner = bench_runner.SGLangBenchmarkRunner(server_args)

            # 创建数据集
            dataset = self._create_dataset(requests)

            # 运行 benchmark
            metrics = runner.benchmark(
                self._create_benchmark_config(),
                dataset,
            )

            print("\n" + "=" * 70)
            print("Simulation Results")
            print("=" * 70)
            print(json.dumps(metrics, indent=2))

            return metrics

        except Exception as e:
            print(f"Error running simulation: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _create_server_args(self):
        """创建服务器参数"""
        from sglang.srt.server_args import ServerArgs

        return ServerArgs(
            model_path=self.config.model_path,
            tokenizer_path=self.config.tokenizer_path,
            host=self.config.host,
            port=self.config.port,
            max_running_requests=self.config.max_running_requests,
            mem_fraction_static=self.config.mem_fraction_static,
            disable_cuda_graph=True,
            disable_disk_cache=True,
        )

    def _create_benchmark_config(self):
        """创建 benchmark 配置"""
        from sglang_simulator.simulation.benchmark import BenchmarkConfig

        return BenchmarkConfig(
            request_rate=self.benchmark_config.request_rate,
            ignore_request_timestamp=self.benchmark_config.ignore_timestamp,
            num_warmup=self.benchmark_config.num_warmup,
        )

    def _create_dataset(self, requests: List[Dict[str, Any]]):
        """创建数据集"""
        from sglang_simulator.dataset import GenericRequest

        dataset = []
        for req in requests:
            dataset.append(GenericRequest(
                prompt=req.get("prompt", "placeholder"),
                token_ids=list(range(req["input_tokens"])),
                output_length=req["output_tokens"],
                custom_params=req,
            ))
        return dataset

    def run_http_benchmark(self) -> Dict[str, Any]:
        """运行 HTTP benchmark"""
        print("=" * 70)
        print(f"Running HTTP benchmark against {self.config.backend_url}")
        print("=" * 70)

        # 生成请求
        generator = AgentTraceRequestGenerator(self.config, self.benchmark_config)
        programs = generator.generate_programs(
            num_programs=20,
            turns_per_program=7,
        )
        requests = generator.generate_requests()

        # 创建客户端并运行
        client = SGLangHTTPClient(self.config.backend_url)

        results = asyncio.run(client.benchmark(
            requests,
            request_rate=self.benchmark_config.request_rate,
            ignore_timestamp=self.benchmark_config.ignore_timestamp,
        ))

        # 统计结果
        success = sum(1 for r in results if r["success"])
        failed = len(results) - success

        metrics = {
            "total_requests": len(results),
            "successful": success,
            "failed": failed,
            "success_rate": success / max(1, len(results)),
            "results": results,
        }

        print("\n" + "=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print(f"Total: {len(results)}")
        print(f"Success: {success}")
        print(f"Failed: {failed}")

        return metrics


# ============================================================================
# CLI Interface
# ============================================================================

def add_cli_args(parser: argparse.ArgumentParser):
    """添加命令行参数"""
    # Simulation 模式
    parser.add_argument(
        "--simulation-mode",
        type=str,
        default="offline",
        choices=["offline", "blocking", "http"],
        help="Simulation mode",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang backend URL (for http mode)",
    )

    # 模型配置
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        choices=["fp32", "fp16", "half", "bf16", "int8", "fp8"],
        help="Model dtype",
    )

    # Continuum TTL 配置
    parser.add_argument(
        "--continuum",
        action="store_true",
        default=True,
        help="Enable Continuum TTL (default: enabled)",
    )
    parser.add_argument(
        "--no-continuum",
        action="store_true",
        help="Disable Continuum TTL",
    )
    parser.add_argument(
        "--ttl-default",
        type=float,
        default=3.0,
        help="Default TTL in seconds",
    )
    parser.add_argument(
        "--ttl-min",
        type=float,
        default=0.1,
        help="Minimum TTL in seconds",
    )
    parser.add_argument(
        "--ttl-max",
        type=float,
        default=10.0,
        help="Maximum TTL in seconds",
    )

    # Benchmark 配置
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Run in benchmark mode",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts for benchmark",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (requests per second)",
    )
    parser.add_argument(
        "--random-input",
        type=int,
        default=1024,
        help="Random input length",
    )
    parser.add_argument(
        "--random-output",
        type=int,
        default=1024,
        help="Random output length",
    )

    # 数据集配置
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (for bench_serving)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset path",
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/sglang_simulator/output",
        help="Output directory",
    )


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Real Integration - Run simulation or benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run offline simulation with agent traces
    python -m sglang_simulator.simulation.sglang.integrate \\
        --simulation-mode offline \\
        --model-path meta-llama/Llama-3.1-8B-Instruct \\
        --num-prompts 1000

    # Run HTTP benchmark against real server
    python -m sglang_simulator.simulation.sglang.integrate \\
        --simulation-mode http \\
        --backend-url http://localhost:30000 \\
        --num-prompts 100

    # Run with Continuum TTL disabled
    python -m sglang_simulator.simulation.sglang.integrate \\
        --no-continuum \\
        --num-prompts 500
        """,
    )

    add_cli_args(parser)
    args = parser.parse_args()

    # 创建配置
    config = SimulationConfig(
        simulation_mode=args.simulation_mode,
        backend_url=args.backend_url,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        continuum_enabled=not args.no_continuum,
        continuum_ttl_default=args.ttl_default,
        continuum_ttl_min=args.ttl_min,
        continuum_ttl_max=args.ttl_max,
        output_dir=args.output_dir,
    )

    benchmark_config = BenchmarkConfig(
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        random_input=args.random_input,
        random_output=args.random_output,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    # 创建启动器
    launcher = SimulatorLauncher(config, benchmark_config)

    # 运行
    if args.simulation_mode == "http":
        metrics = launcher.run_http_benchmark()
    else:
        metrics = launcher.run_offline_simulation()

    # 保存结果
    if args.output_dir:
        output_path = Path(args.output_dir) / "simulation_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
