import json
import time
from typing import Optional

from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.types import PlatformConfig, SchedulerConfig
from sglang_simulator.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from sglang_simulator.spec import AcceleratorInfo, DataType, ModelInfo
from sglang_simulator.utils import get_logger

logger = get_logger()


class SimulatedTimePredictor:
    """A simple time predictor that uses analytical models when aiconfigurator is not available."""

    # Pre-computed latency factors (in seconds per token)
    PREFILL_LATENCY_PER_TOKEN = 0.0001  # 0.1ms per input token
    DECODE_LATENCY_PER_TOKEN = 0.00005   # 0.05ms per output token

    def __init__(self, model_info: ModelInfo, scheduler_config: SchedulerConfig):
        self.model_info = model_info
        self.scheduler_config = scheduler_config

    def predict_infer_time(self, batch) -> float:
        """Predict inference time based on batch composition."""
        total_prefill_tokens = 0
        total_decode_tokens = 0

        for req in batch.reqs:
            if req.extend_length > 1:
                # This is a prefill request
                total_prefill_tokens += req.extend_length
            else:
                # This is a decode request
                total_decode_tokens += 1

        prefill_time = total_prefill_tokens * self.PREFILL_LATENCY_PER_TOKEN
        decode_time = total_decode_tokens * self.DECODE_LATENCY_PER_TOKEN

        return prefill_time + decode_time


class ConfigManager:
    """Centralized configuration manager with caching."""

    _model_info: Optional[ModelInfo] = None
    _platform_config: Optional[PlatformConfig] = None
    _scheduler_config: Optional[SchedulerConfig] = None
    _raw_config: Optional[dict] = None

    @classmethod
    def _get_raw_config(cls) -> dict:
        if cls._raw_config is None:
            with open(Envs.config_path()) as f:
                cls._raw_config = json.load(f)
        return cls._raw_config

    @classmethod
    def reset_config_cache(cls):
        cls._raw_config = None
        cls._model_info = None
        cls._platform_config = None
        cls._scheduler_config = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(cls) -> ModelInfo | None:
        return cls._model_info

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        config = cls._get_raw_config()
        platform_config = config.get("platform", {})
        acc_info = platform_config.get("accelerator", {})
        hw = AcceleratorInfo.find_by_hw_name(acc_info.get("name"))
        if hw is None:
            logger.debug(
                f"Failed to initialize device info with {acc_info.get('name')}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            hw = AcceleratorInfo(
                name=acc_info.get("name"),
                vendor=acc_info.get("vendor"),
                hbm_bandwidth_gb=acc_info.get("hbm_bandwidth_gb"),
                hbm_capacity_gb=acc_info.get("hbm_capacity_gb"),
                inter_node_bandwidth_gb=acc_info.get("inter_node_bandwidth_gb"),
                intra_node_bandwidth_gb=acc_info.get("intra_node_bandwidth_gb"),
                tflops=acc_info.get("tflops"),
            )
        else:
            logger.info(f"Device info initialized: {hw}")
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            config = cls._get_raw_config()
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        external_config = cls._get_raw_config().get("scheduler", {})
        for field_name in [
            "tp_size",
            "dp_size",
            "ep_size",
            "backend_name",
            "backend_version",
        ]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, field_value)

        for field_name in ["data_type", "kv_cache_data_type"]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, DataType(field_value))

        cls._scheduler_config = config

    @classmethod
    def get_kv_cache_bytes(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_cell_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_kv_cache_bytes_per_layer(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_per_layer_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_scheduler_config(cls):
        return cls._scheduler_config

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=server_args.get("dp_size", 1),
                mem_fraction_static=server_args.get("mem_fraction_static"),
                backend_name="sglang",
            )
        else:
            raise RuntimeError(f"Unsupported backend[{backend}] server args parser.")

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ):
        """Get inference time predictor.
        
        Tries to use AIConfiguratorTimePredictor if available, otherwise falls back
        to SimulatedTimePredictor for basic simulation without the external dependency.
        """
        config = cls._get_raw_config()
        predictor_config = config.get("predictor", {})
        
        if predictor_config.get("name") == "aiconfigurator":
            try:
                from sglang_simulator.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
                database_mode = predictor_config.get("database_mode", "SILICON")
                prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
                decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
                return AIConfiguratorTimePredictor(
                    model,
                    hw=hw,
                    config=sched_config,
                    database_path=predictor_config.get("database_path"),
                    database_mode=database_mode,
                    prefill_scale_factor=prefill_scale_factor,
                    decode_scale_factor=decode_scale_factor,
                )
            except ImportError as e:
                logger.warning(
                    f"AIConfigurator not available ({e}). Using SimulatedTimePredictor instead. "
                    "For accurate simulation, please install aiconfigurator: "
                    "pip install aiconfigurator"
                )
                return SimulatedTimePredictor(model, sched_config)
        else:
            # Use simulated predictor for unknown predictor types
            logger.info(
                f"Unknown predictor '{predictor_config.get('name')}'. "
                "Using SimulatedTimePredictor for basic simulation."
            )
            return SimulatedTimePredictor(model, sched_config)
