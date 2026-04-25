import heapq
import importlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict

from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args
from sglang_simulator.simulation.manager import ConfigManager, Envs, StateManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.types import (
    RequestStats,
    SimulationMode,
)
from sglang_simulator.simulation.utils import (
    calc_metrics,
)
from sglang_simulator.time_predictor import InferTimePredictor
from sglang_simulator.time_predictor import ScheduleBatch as SimulationScheduleBatch
from sglang_simulator.time_predictor import ScheduleRequest
from sglang_simulator.utils import get_logger
from sglang_simulator.utils.json import CustomJsonEncoder

logger = get_logger("sgl_simulator")


# ========== Visualization Data Collection Helpers ==========

def _extract_program_id(req) -> tuple[str, int]:
    """从请求中提取 program_id 和 turn_index"""
    try:
        sampling_params = getattr(req, 'sampling_params', None)
        custom_params = getattr(sampling_params, 'custom_params', None) or {}
        sim_params = custom_params.get("simulation", {}) if isinstance(custom_params, dict) else {}
        program_id = sim_params.get("program_id", f"req_{req.rid[:8]}")
        turn_index = sim_params.get("turn_index", 0)
    except (AttributeError, TypeError):
        # 处理 None 或无效的 sampling_params
        program_id = f"req_{req.rid[:8]}"
        turn_index = 0
    return program_id, turn_index


def _get_cache_state_snapshot(gpu) -> dict:
    """获取 GPU 缓存状态快照"""
    # L0 (GPU) 缓存条目
    l0_entries = []
    if hasattr(gpu, 'radix_cache'):
        cache = gpu.radix_cache
        if hasattr(cache, 'cache'):
            for key, node in cache.cache.items():
                if node and node.kv_cache:
                    tokens = getattr(node, 'used_tokens', 0)
                    pinned = getattr(node, 'is_pinned', False)
                    if tokens > 0:
                        l0_entries.append({
                            "key": str(key),
                            "tokens": tokens,
                            "pinned": pinned,
                        })

    # L1 (Host) 缓存条目（如果有）
    l1_entries = []
    if hasattr(gpu, 'hicache') and gpu.hicache:
        storage = gpu.hicache
        if hasattr(storage, 'host_storage') and storage.host_storage:
            for key, data in storage.host_storage.items():
                tokens = getattr(data, 'size', 0) if data else 0
                if tokens > 0:
                    l1_entries.append({
                        "key": str(key),
                        "tokens": tokens,
                    })

    return {
        "l0_entries": l0_entries,
        "l1_entries": l1_entries,
    }


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    SIMULATION_BATCH: SimulationScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = SimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[tuple[float, int, RequestStats]] = (
        []
    )  # tuple(created time, salt, request)

    # ========== Continuum TTL Integration ==========
    TTL_SIMULATOR = None  # ContinuumTTLSimulator instance
    TTL_ENABLED = False
    PINNED_PROGRAMS: dict[str, float] = {}  # program_id -> expires_at
    TTL_CONFIG: dict = {
        "default_ttl": 3.0,
        "min_ttl": 0.1,
        "max_ttl": 10.0,
        "history_threshold": 10,
        "memory_pressure_penalty": 0.5,
    }

    # ========== Visualization Data Collection ==========
    # 每个时间步的完整缓存状态快照
    CACHE_SNAPSHOTS: list[dict] = []
    # 每个事件的详细命中信息
    EVENT_RECORDS: list[dict] = []
    # 程序元信息
    PROGRAM_META: dict[str, dict] = {}
    # 缓存配置
    CACHE_CONFIG: dict = {
        "max_tokens": 0,
        "max_host_tokens": 0,
        "l1_bandwidth_gb": 50.0,
        "l2_bandwidth_gb": 5.0,
    }

    @classmethod
    def _init_visualization_data(cls):
        """初始化可视化数据收集器"""
        cls.CACHE_SNAPSHOTS.clear()
        cls.EVENT_RECORDS.clear()
        cls.PROGRAM_META.clear()
        cls.CACHE_CONFIG = {
            "max_tokens": 0,
            "max_host_tokens": 0,
            "l1_bandwidth_gb": 50.0,
            "l2_bandwidth_gb": 5.0,
        }

    @classmethod
    def _init_cache_config(cls, server_args):
        """从服务器参数初始化缓存配置"""
        # 从 scheduler_config 获取 max_tokens
        sched_config = ConfigManager.get_scheduler_config()
        if sched_config:
            cls.CACHE_CONFIG["max_tokens"] = getattr(
                sched_config, "max_total_num_tokens", 100000
            )
            cls.CACHE_CONFIG["max_host_tokens"] = getattr(
                sched_config, "max_total_num_tokens", 100000
            ) * 10  # Host 通常是 GPU 的 10 倍

    @classmethod
    def _init_ttl_simulator(cls, server_args):
        """Initialize Continuum TTL Simulator from config"""
        from sglang_simulator.simulation.agent.continuum_ttl import ContinuumTTLSimulator

        # Load TTL config from environment or use defaults
        ttl_default = float(os.getenv("CONTINUUM_TTL_DEFAULT", "3.0"))
        ttl_min = float(os.getenv("CONTINUUM_TTL_MIN", "0.1"))
        ttl_max = float(os.getenv("CONTINUUM_TTL_MAX", "10.0"))
        ttl_history = int(os.getenv("CONTINUUM_TTL_HISTORY", "10"))
        ttl_penalty = float(os.getenv("CONTINUUM_TTL_PENALTY", "0.5"))
        cls.TTL_ENABLED = os.getenv("CONTINUUM_TTL_ENABLED", "false").lower() == "true"

        cls.TTL_CONFIG = {
            "default_ttl": ttl_default,
            "min_ttl": ttl_min,
            "max_ttl": ttl_max,
            "history_threshold": ttl_history,
            "memory_pressure_penalty": ttl_penalty,
        }

        if cls.TTL_ENABLED:
            cls.TTL_SIMULATOR = ContinuumTTLSimulator(
                default_ttl=ttl_default,
                min_ttl=ttl_min,
                max_ttl=ttl_max,
                history_threshold=ttl_history,
                memory_pressure_penalty=ttl_penalty,
                enable_adaptive_ttl=True,
            )
            logger.info(f"[Continuum TTL] Simulator enabled with config: {cls.TTL_CONFIG}")
        else:
            cls.TTL_SIMULATOR = None
            logger.info("[Continuum TTL] Simulator disabled (baseline mode)")

    @classmethod
    def _record_tool_execution(cls, program_id: str, tool_name: str, duration: float, idle_gap: float = 0):
        """Record tool execution for TTL calculation"""
        if cls.TTL_SIMULATOR is not None:
            cls.TTL_SIMULATOR.record_tool_execution(program_id, tool_name, duration, idle_gap)

    @classmethod
    def _select_ttl(cls, program_id: str, tool_name: str = None, queue_time: float = 0) -> tuple[float, str]:
        """Select dynamic TTL for a program"""
        if cls.TTL_SIMULATOR is not None:
            return cls.TTL_SIMULATOR.select_dynamic_ttl(program_id, tool_name, queue_time)
        return cls.TTL_CONFIG["default_ttl"], "default"

    @classmethod
    def _expire_pins(cls, current_time: float):
        """Expire old PINs and return expired program IDs"""
        expired = []
        for pid, expires_at in list(cls.PINNED_PROGRAMS.items()):
            if current_time > expires_at:
                expired.append(pid)
                del cls.PINNED_PROGRAMS[pid]
        return expired

    @classmethod
    def _is_pinned(cls, program_id: str, current_time: float) -> bool:
        """Check if a program is currently pinned"""
        if program_id not in cls.PINNED_PROGRAMS:
            return False
        return current_time <= cls.PINNED_PROGRAMS[program_id]

    @classmethod
    def _pin_program(cls, program_id: str, ttl_sec: float, current_time: float):
        """Pin a program for the specified TTL"""
        cls.PINNED_PROGRAMS[program_id] = current_time + ttl_sec

    @classmethod
    def _get_priority(cls, program_id: str, arrival_time: float, current_time: float) -> tuple:
        """Get scheduling priority: PIN programs > FCFS"""
        if cls._is_pinned(program_id, current_time):
            return (0, arrival_time)  # PIN programs get highest priority
        return (1, arrival_time)  # FCFS for others

    @classmethod
    def get_ttl_stats(cls) -> dict:
        """Get TTL statistics"""
        if cls.TTL_SIMULATOR is not None:
            stats = cls.TTL_SIMULATOR.get_stats()
            return {
                **stats,
                "ttl_enabled": cls.TTL_ENABLED,
                "active_pins": len(cls.PINNED_PROGRAMS),
                "ttl_config": cls.TTL_CONFIG,
            }
        return {
            "ttl_enabled": False,
            "active_pins": len(cls.PINNED_PROGRAMS),
            "ttl_config": cls.TTL_CONFIG,
        }

    @classmethod
    def record_cache_snapshot(cls, current_time: float, gpu_states: list):
        """记录当前时间步的缓存状态快照"""
        if not gpu_states:
            return

        # 汇总所有 GPU 的缓存状态
        total_l0_tokens = 0
        total_l1_tokens = 0
        l0_entries = []
        l1_entries = []

        for gpu in gpu_states:
            # 统计 L0 缓存
            if hasattr(gpu, 'radix_cache'):
                cache = gpu.radix_cache
                if hasattr(cache, 'used_tokens'):
                    total_l0_tokens += cache.used_tokens

            # 收集 L0 条目信息
            if hasattr(gpu, 'running_batch') and gpu.running_batch:
                for req in gpu.running_batch.reqs:
                    tokens = getattr(req, 'cached_tokens', 0)
                    if tokens > 0:
                        program_id, turn_index = _extract_program_id(req)
                        pinned = getattr(req, 'is_pinned', False)
                        l0_entries.append({
                            "program_id": program_id,
                            "turn": turn_index,
                            "tokens": tokens,
                            "pinned": pinned,
                        })

            # 统计 L1 缓存
            if hasattr(gpu, 'hicache') and gpu.hicache:
                storage = gpu.hicache
                if hasattr(storage, 'host_tokens'):
                    total_l1_tokens += storage.host_tokens

        snapshot = {
            "time": current_time,
            "l0_tokens": total_l0_tokens,
            "l1_tokens": total_l1_tokens,
            "l0_entries": l0_entries,
            "l1_entries": l1_entries,
            "running_count": sum(len(gpu.running_batch.reqs) if hasattr(gpu, 'running_batch') else 0 for gpu in gpu_states),
            "pending_count": sum(len(gpu.waiting_queue) if hasattr(gpu, 'waiting_queue') else 0 for gpu in gpu_states),
        }
        cls.CACHE_SNAPSHOTS.append(snapshot)

    @classmethod
    def record_event(cls, event_data: dict):
        """记录一个事件"""
        cls.EVENT_RECORDS.append(event_data)

    @classmethod
    def get_visualization_data(cls) -> dict:
        """获取完整的可视化数据"""
        # 统计汇总
        total_events = len(cls.EVENT_RECORDS)
        l0_hits = sum(1 for e in cls.EVENT_RECORDS if e.get("hit_type") == "L0")
        l1_hits = sum(1 for e in cls.EVENT_RECORDS if e.get("hit_type") == "L1")
        misses = sum(1 for e in cls.EVENT_RECORDS if e.get("hit_type") == "Miss")
        pinned = sum(1 for e in cls.EVENT_RECORDS if e.get("was_pinned", False))
        evicted = sum(e.get("evicted_tokens", 0) for e in cls.EVENT_RECORDS)

        # Time breakdown statistics
        total_prefill_time = sum(e.get("prefill_time", 0) for e in cls.EVENT_RECORDS)
        total_decode_time = sum(e.get("decode_time", 0) for e in cls.EVENT_RECORDS)
        total_tool_time = sum(e.get("tool_time", 0) for e in cls.EVENT_RECORDS)
        total_queue_time = sum(e.get("queue_time", 0) for e in cls.EVENT_RECORDS)

        total_time = 0
        if cls.CACHE_SNAPSHOTS:
            total_time = cls.CACHE_SNAPSHOTS[-1].get("time", 0)

        # Get TTL stats
        ttl_stats = cls.get_ttl_stats()

        return {
            "config": cls.CACHE_CONFIG,
            "ttl_config": cls.TTL_CONFIG,
            "events": cls.EVENT_RECORDS,
            "cache_snapshots": cls.CACHE_SNAPSHOTS,
            "program_meta": cls.PROGRAM_META,
            "summary": {
                "total_events": total_events,
                "total_time": total_time,
                "l0_hits": l0_hits,
                "l1_hits": l1_hits,
                "misses": misses,
                "pinned_count": pinned,
                "evicted_tokens": evicted,
                "l0_hit_rate": l0_hits / max(1, total_events),
                "l1_hit_rate": l1_hits / max(1, total_events),
                "overall_hit_rate": (l0_hits + l1_hits) / max(1, total_events),
            },
            "time_breakdown": {
                "total_prefill_time": total_prefill_time,
                "total_decode_time": total_decode_time,
                "total_tool_time": total_tool_time,
                "total_queue_time": total_queue_time,
                "prefill_pct": total_prefill_time / max(1, total_time) * 100,
                "decode_pct": total_decode_time / max(1, total_time) * 100,
                "tool_pct": total_tool_time / max(1, total_time) * 100,
                "queue_pct": total_queue_time / max(1, total_time) * 100,
            },
            "stats": {
                "l0_hits": l0_hits,
                "l1_hits": l1_hits,
                "misses": misses,
                "evicted": evicted,
                "hit_rate": (l0_hits + l1_hits) / max(1, total_events),
            },
            "ttl_stats": ttl_stats,
        }

    @classmethod
    def reset_visualization_data(cls):
        """重置可视化数据"""
        cls.CACHE_SNAPSHOTS.clear()
        cls.EVENT_RECORDS.clear()
        cls.PROGRAM_META.clear()
        cls.PINNED_PROGRAMS.clear()
        if cls.TTL_SIMULATOR:
            cls.TTL_SIMULATOR.reset()

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = target.recv_requests
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            # Note: TTL Continuum Pinning (init_continuum_pin) is already called
            # by the original __init__ method in scheduler.py.
            # The simulator preserves all TTL-related functionality.

            # ========== Initialize Continuum TTL Simulator ==========
            C_SchedulerHook._init_ttl_simulator(server_args)

            # ========== Initialize Visualization Data Collection ==========
            C_SchedulerHook._init_visualization_data()
            C_SchedulerHook._init_cache_config(server_args)

            try:
                if ConfigManager.get_model_info() is None:
                    model = resolve_model_info(self.model_config)
                    ConfigManager.set_model_info(model)

                model = ConfigManager.get_model_info()

                hw = ConfigManager.get_accelerator_info()

                if ConfigManager.get_scheduler_config() is None:
                    sched_config = resolve_scheduler_config(
                        server_args=self.server_args,
                    )
                    ConfigManager.set_scheduler_config(sched_config)
                sched_config = ConfigManager.get_scheduler_config()

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        # 安全获取 total_request
                        sampling_params = getattr(gen_req, 'sampling_params', None)
                        custom_params = getattr(sampling_params, 'custom_params', None) if sampling_params else None
                        simulation = (custom_params or {}).get("simulation") if custom_params else None
                        total_request = (simulation or {}).get("total_request", 1) if simulation else 1

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    recv_reqs.append(req)
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)

                    # 安全获取 max_new_tokens
                    sampling_params = getattr(req, 'sampling_params', None)
                    if sampling_params:
                        req_stats.output_length = getattr(sampling_params, 'max_new_tokens', 0) or 0
                        custom_params = getattr(sampling_params, 'custom_params', None)
                        simulation_args = (custom_params or {}).get("simulation") if custom_params else None
                    else:
                        req_stats.output_length = 0
                        simulation_args = None

                    if simulation_args is None:
                        # 跳过非模拟请求
                        continue
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                simulation_batch = SimulationScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=req.extend_input_len,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not simulation_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            simulation_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.SIMULATION_BATCH = simulation_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock() + hicache_l2_backup_dur
                    )
                else:
                    StateManager.step_global_clock(
                        hicache_l2_load_dur
                        + current_inference_dur
                        + hicache_l2_backup_dur
                    )
                    request_response_time = StateManager.get_global_clock()
                # Request statistics
                for req in batch.reqs:
                    if req.is_chunked == 0:
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.SIMULATION_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                    }
                )

                # ========== Visualization Data Collection ==========
                # 获取所有 GPU 状态
                gpu_states = getattr(self, 'gpu_states', [self])
                current_time = StateManager.get_global_clock()

                # 记录缓存快照
                C_SchedulerHook.record_cache_snapshot(current_time, gpu_states)

                # 为每个请求记录事件
                for req in batch.reqs:
                    if req.is_chunked == 0:
                        program_id, turn_index = _extract_program_id(req)
                        req_stats = C_SchedulerHook.REQUEST_STATS.get(req.rid)

                        # 判断缓存命中类型
                        hit_type = "Miss"
                        hit_tokens = 0
                        miss_tokens = req_stats.input_length if req_stats else 0
                        was_pinned = getattr(req, 'is_pinned', False)

                        if hasattr(req, 'cached_tokens') and req.cached_tokens > 0:
                            hit_type = "L0"
                            hit_tokens = req.cached_tokens
                            miss_tokens = max(0, (req_stats.input_length if req_stats else 0) - hit_tokens)

                        # 记录程序元信息
                        if program_id not in C_SchedulerHook.PROGRAM_META:
                            simulation_args = getattr(req, 'sampling_params', None)
                            simulation_args = (getattr(simulation_args, 'custom_params', None) or {}).get("simulation", {}) if simulation_args else {}
                            if isinstance(simulation_args, dict):
                                C_SchedulerHook.PROGRAM_META[program_id] = {
                                    "program_id": program_id,
                                    "is_tool_call": simulation_args.get("is_tool_call", False),
                                    "tool_name": simulation_args.get("tool_name"),
                                }
                            else:
                                simulation_args = {}
                                C_SchedulerHook.PROGRAM_META[program_id] = {
                                    "program_id": program_id,
                                    "is_tool_call": False,
                                    "tool_name": None,
                                }

                        # 获取 simulation_args 用于事件记录
                        if program_id in C_SchedulerHook.PROGRAM_META:
                            meta = C_SchedulerHook.PROGRAM_META[program_id]
                            is_tool_call = meta.get("is_tool_call", False)
                            tool_name = meta.get("tool_name")
                        else:
                            is_tool_call = False
                            tool_name = None

                        # Calculate time breakdown
                        current_inference_dur = StateManager.get_current_inference_dur()
                        hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                        hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()

                        # Queue time: time from last event to now
                        queue_time = 0
                        if req_stats:
                            queue_time = current_time - req_stats.last_event_time

                        # Prefill time vs decode time based on hit type
                        if hit_type == "Miss" or hit_type == "L0":
                            # Miss or L0 hit requires prefill
                            prefill_time = current_inference_dur * 0.7  # Approximate: 70% prefill
                            decode_time = current_inference_dur * 0.3   # 30% decode
                        else:
                            # L1 hit or other - mostly decode
                            prefill_time = 0
                            decode_time = current_inference_dur

                        # Tool time (if this is a tool call, add the tool duration)
                        tool_time = 0
                        if is_tool_call and tool_name:
                            # Get tool duration from simulation params
                            tool_duration_meta = meta.get("tool_duration", 0)
                            tool_time = tool_duration_meta

                        # PIN logic for TTL
                        ttl_sec = 0
                        ttl_strategy = ""
                        if is_tool_call and C_SchedulerHook.TTL_ENABLED:
                            # Select TTL based on tool execution history
                            ttl_sec, ttl_strategy = C_SchedulerHook._select_ttl(
                                program_id, tool_name, queue_time
                            )
                            # Pin the program
                            C_SchedulerHook._pin_program(program_id, ttl_sec, current_time)

                        # 记录事件
                        event = {
                            "time": current_time,
                            "program_id": program_id,
                            "turn_index": turn_index,
                            "input_tokens": req_stats.input_length if req_stats else 0,
                            "output_tokens": req_stats.output_length if req_stats else 0,
                            "hit_type": hit_type,
                            "hit_tokens": hit_tokens,
                            "miss_tokens": miss_tokens,
                            "was_pinned": was_pinned,
                            "evicted_tokens": 0,
                            "is_tool_call": is_tool_call,
                            "tool_name": tool_name,
                            # Time breakdown
                            "prefill_time": prefill_time,
                            "decode_time": decode_time,
                            "tool_time": tool_time,
                            "queue_time": queue_time,
                            "h2d_time": hicache_l2_load_dur,
                            "l2_backup_time": hicache_l2_backup_dur,
                            # TTL info
                            "ttl_sec": ttl_sec,
                            "ttl_strategy": ttl_strategy,
                        }
                        C_SchedulerHook.record_event(event)
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            stats: list[RequestStats] = []
            for item in C_SchedulerHook.REQUEST_STATS.values():
                if item.rid is not None and item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    # ========== Export Visualization Data ==========
                    viz_data = C_SchedulerHook.get_visualization_data()
                    # 对齐时间戳
                    if viz_data["events"]:
                        min_time = min(e["time"] for e in viz_data["events"])
                        for event in viz_data["events"]:
                            event["time"] -= min_time
                    if viz_data["cache_snapshots"]:
                        min_time = min(s["time"] for s in viz_data["cache_snapshots"])
                        for snapshot in viz_data["cache_snapshots"]:
                            snapshot["time"] -= min_time
                    viz_data["summary"]["total_time"] = max(
                        (e["time"] for e in viz_data["events"]), default=0
                    )

                    with open(f"{output_dir}/visualization_data.json", "w") as f:
                        f.write(json.dumps(viz_data, cls=CustomJsonEncoder, indent=2))

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.reset_visualization_data()  # Reset visualization data
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target.profile = wrapped_profile
