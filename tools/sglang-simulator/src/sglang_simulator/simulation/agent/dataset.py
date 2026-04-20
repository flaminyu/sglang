#!/usr/bin/env python3
"""
Multi-Turn Agent Dataset for SGLang Simulator

支持多轮 Agent 程序的数据集，用于模拟长上下文场景下的 KV Cache 策略对比。
每轮使用固定文本和长度，便于精确模拟和重现。

Usage:
    dataset = MultiTurnAgentDataset(tokenizer, args, programs)
    request = dataset[0]  # 获取第一个请求
"""
from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from sglang_simulator.dataset.base_dataset import BaseDataset, GenericRequest


# ============================================================================
# 固定模板配置
# ============================================================================

# 系统提示词模板
SYSTEM_PROMPT = """You are an expert software engineering assistant. You are working on a complex coding task that requires multiple steps. Use the provided tools to analyze code, make changes, run tests, and iterate until the task is complete."""

# 用户提示词模板
USER_PROMPTS = [
    "I need you to help me complete a software engineering task. This will require multiple steps: understanding the codebase, implementing changes, and verifying the solution. Let's start by examining the existing code.",
    "Please continue with the implementation. Make sure to follow best practices and add appropriate error handling.",
    "Good progress! Now let's run the tests to verify our changes work correctly.",
    "The tests are passing. Let's add some additional features to make the solution more robust.",
    "Excellent work! Let's optimize the code for better performance.",
    "Now let's add documentation to make the code more maintainable.",
    "Let's refactor the code to improve its structure and readability.",
    "Now let's add edge case handling to make the solution more complete.",
    "Great! Let's run the full test suite to ensure everything is working.",
    "Excellent! The solution is complete. Let's create a summary of the changes made.",
]

# LLM 思考响应模板
THOUGHT_RESPONSES = [
    "Let me analyze the current state of the codebase to understand what needs to be done.",
    "I need to understand how the existing code works before making any changes.",
    "Looking at the requirements, I should implement this feature step by step.",
    "The test results indicate there might be an issue I need to fix in the implementation.",
    "Let me check if there are any edge cases I need to handle for this feature.",
    "Based on my analysis, I need to modify the implementation to meet the requirements.",
    "I should verify that my changes don't break existing functionality before proceeding.",
    "Let me run the tests to confirm the solution works as expected.",
    "The implementation looks good. Let me add some additional error handling.",
    "Now I need to optimize the code for better performance while maintaining correctness.",
]

# 工具调用响应模板
TOOL_CALL_RESPONSES = [
    "I need to use the read_file tool to examine the existing code structure.",
    "Let me use the grep tool to search for relevant functions in the codebase.",
    "I need to execute bash commands to test the implementation.",
    "Let me use the write_file tool to create the updated implementation.",
    "I need to use the sed tool to make quick text replacements.",
    "Let me use the git tool to check the current state of the repository.",
]

# 工具名称列表
TOOL_NAMES = [
    "read_file",
    "grep",
    "bash",
    "write_file",
    "sed",
    "git",
]

# 工具结果模板
TOOL_RESULTS = {
    "read_file": "[File contents read successfully - showing first 100 lines]",
    "grep": "[Found 3 matches in 2 files at lines 42, 108, 215]",
    "bash": "[Command executed successfully with exit code 0]",
    "write_file": "[File written successfully to disk]",
    "sed": "[Pattern replaced in 2 lines]",
    "git": "[Git command executed - 1 file changed, 3 insertions(+)]",
}


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TurnSpec:
    """单轮对话规格"""
    turn_index: int
    input_tokens: int  # 估算的输入 token 数
    output_tokens: int  # 输出 token 数
    is_tool_call: bool  # 是否为工具调用
    tool_name: Optional[str] = None  # 工具名称
    tool_duration: float = 0.0  # 工具执行时间（秒）
    
    def __post_init__(self):
        if self.is_tool_call and self.tool_name is None:
            self.tool_name = random.choice(TOOL_NAMES)


@dataclass
class AgentProgram:
    """
    单个 Agent 程序规格
    
    包含多轮对话的固定规格，可以精确重现实验。
    """
    program_id: str
    turns: List[TurnSpec] = field(default_factory=list)
    arrival_time: float = 0.0  # Poisson 到达时间
    
    @property
    def total_tokens(self) -> int:
        """总 token 数（所有轮次的输入之和）"""
        return sum(t.input_tokens for t in self.turns)
    
    @property
    def num_tool_calls(self) -> int:
        """工具调用次数"""
        return sum(1 for t in self.turns if t.is_tool_call)
    
    @property
    def tool_ratio(self) -> float:
        """工具调用比例"""
        return self.num_tool_calls / max(1, len(self.turns))
    
    def get_turn_input_text(self, turn_idx: int, tokenizer) -> str:
        """获取指定轮次的输入文本（固定）"""
        if turn_idx == 0:
            # 第一轮：系统 + 初始用户消息
            return SYSTEM_PROMPT + "\n\n" + USER_PROMPTS[0]
        
        prev_turn = self.turns[turn_idx - 1]
        if prev_turn.is_tool_call:
            # 工具调用后的轮次：工具结果 + 继续思考
            tool_result = TOOL_RESULTS.get(prev_turn.tool_name, "[Tool executed]")
            return f"[Tool Result]: {tool_result}\n\n{THOUGHT_RESPONSES[turn_idx % len(THOUGHT_RESPONSES)]}"
        else:
            # 普通轮次后的用户继续
            return USER_PROMPTS[turn_idx % len(USER_PROMPTS)]
    
    def get_turn_output_text(self, turn_idx: int) -> str:
        """获取指定轮次的输出文本（固定）"""
        turn = self.turns[turn_idx]
        if turn.is_tool_call:
            # 工具调用：思考 + 函数调用
            return (f"{THOUGHT_RESPONSES[turn_idx % len(THOUGHT_RESPONSES)]}\n\n"
                   f"<invoke name=\"{turn.tool_name}\">\n"
                    "<parameter name=\"arguments\">{\"query\": \"...\"}</parameter>\n"
                    "</invoke>")
        else:
            return THOUGHT_RESPONSES[turn_idx % len(THOUGHT_RESPONSES)]


# ============================================================================
# 数据集类
# ============================================================================

class MultiTurnAgentDataset(BaseDataset):
    """
    多轮 Agent 程序数据集
    
    用于 SGLang Simulator 的多轮对话场景模拟。
    支持固定文本和长度的精确重现。
    
    Usage:
        config = AgentDatasetConfig(
            num_programs=20,
            turns_per_program=7,
            arrival_rate=0.3,  # requests per second
            target_tokens=50000,
            tool_ratio=0.7,
            tool_time_mean=0.5,
            tool_time_std=0.2,
        )
        programs = MultiTurnAgentDataset.generate_programs(config)
        dataset = MultiTurnAgentDataset(tokenizer, args, programs)
        
        # 获取请求
        for i in range(len(dataset)):
            request = dataset[i]
            print(f"Request {i}: program_id={request.custom_params['program_id']}, "
                  f"turn={request.custom_params['turn_index']}")
    """
    
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        args,  # DatasetArgs
        programs: List[AgentProgram],
    ):
        super().__init__(tokenizer, args)
        self.programs = programs
        self._requests: List[GenericRequest] = []
        self._build_requests()
    
    def _build_requests(self):
        """预构建所有请求"""
        self._requests = []
        total_requests = sum(len(p.turns) for p in self.programs)
        
        for program in self.programs:
            for turn_idx, turn in enumerate(program.turns):
                # 构建输入文本
                input_text = program.get_turn_input_text(turn_idx, self.tokenizer)
                
                # Tokenize
                if hasattr(self.tokenizer, 'encode'):
                    token_ids = self.tokenizer.encode(
                        input_text,
                        truncation=True,
                        max_length=getattr(self.args, 'max_input_len', 32768)
                    )
                else:
                    token_ids = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=getattr(self.args, 'max_input_len', 32768)
                    )['input_ids']
                
                # 计算到达时间（相对于程序到达）
                # 每个程序的到达时间累加 Poisson 间隔
                program_arrival = program.arrival_time
                
                # 输出文本
                output_text = program.get_turn_output_text(turn_idx)
                
                request = GenericRequest(
                    prompt=input_text,
                    token_ids=token_ids,
                    input_length=len(token_ids),
                    output_length=turn.output_tokens,
                    custom_params={
                        "program_id": program.program_id,
                        "turn_index": turn_idx,
                        "is_tool_call": turn.is_tool_call,
                        "tool_name": turn.tool_name,
                        "tool_duration": turn.tool_duration,
                        "created_time": program_arrival,  # 到达时间（秒）
                        "total_request": total_requests,
                        "cumulative_tokens": program.total_tokens,
                    }
                )
                self._requests.append(request)
        
        # 按到达时间排序
        self._requests.sort(key=lambda r: r.custom_params.get("created_time", 0))
    
    def __len__(self) -> int:
        return len(self._requests)
    
    def __getitem__(self, index: int) -> GenericRequest:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        return self._requests[index]
    
    def _get_single_item(self, index: int) -> GenericRequest:
        return self[index]
    
    @staticmethod
    def generate_programs(
        num_programs: int,
        turns_per_program: int,
        arrival_rate: float,
        target_tokens: int,
        input_growth: int,
        output_tokens: int,
        tool_ratio: float,
        tool_time_mean: float,
        tool_time_std: float,
        seed: int = 42,
    ) -> List[AgentProgram]:
        """
        生成多轮 Agent 程序列表
        
        使用固定的文本和长度，便于精确重现实验。
        
        Args:
            num_programs: 程序数量
            turns_per_program: 每程序轮数
            arrival_rate: Poisson 到达率 (requests/second)
            target_tokens: 目标总 tokens
            input_growth: 每轮输入增长 tokens
            output_tokens: 每轮输出 tokens
            tool_ratio: 工具调用比例
            tool_time_mean: 工具执行时间均值（秒）
            tool_time_std: 工具执行时间标准差（秒）
            seed: 随机种子
        
        Returns:
            AgentProgram 列表
        """
        random.seed(seed)
        
        programs = []
        current_time = 0.0
        
        for prog_idx in range(num_programs):
            # Poisson 到达间隔
            if prog_idx > 0 and arrival_rate > 0:
                inter_arrival = random.expovariate(arrival_rate)
                current_time += inter_arrival
            
            turns = []
            cumulative_tokens = 0
            current_input_tokens = 500  # 初始输入约 500 tokens
            
            for turn_idx in range(turns_per_program):
                # 决定是否为工具调用
                is_tool_call = random.random() < tool_ratio
                
                # 工具名称
                tool_name = random.choice(TOOL_NAMES) if is_tool_call else None
                
                # 工具执行时间（对数正态分布）
                tool_duration = 0.0
                if is_tool_call:
                    cv = tool_time_std / tool_time_mean if tool_time_mean > 0 else 1.0
                    phi = math.sqrt(cv ** 2 + 1)
                    mu = math.log(tool_time_mean) - math.sqrt(math.log(phi))
                    sigma = max(0.1, math.sqrt(math.log(phi)))
                    tool_duration = max(0.01, random.lognormvariate(mu, sigma))
                
                turn = TurnSpec(
                    turn_index=turn_idx,
                    input_tokens=current_input_tokens,
                    output_tokens=output_tokens,
                    is_tool_call=is_tool_call,
                    tool_name=tool_name,
                    tool_duration=tool_duration,
                )
                turns.append(turn)
                
                # 累积 tokens
                cumulative_tokens += current_input_tokens + output_tokens
                if is_tool_call:
                    cumulative_tokens += 50  # 工具结果约 50 tokens
                
                # 增长输入
                current_input_tokens = min(
                    current_input_tokens + input_growth,
                    target_tokens
                )
                
                # 达到目标 tokens 后停止
                if cumulative_tokens >= target_tokens:
                    break
            
            program = AgentProgram(
                program_id=f"prog_{prog_idx:04d}",
                turns=turns,
                arrival_time=current_time,
            )
            programs.append(program)
        
        return programs


@dataclass
class AgentDatasetConfig:
    """Agent 数据集配置"""
    num_programs: int = 20
    turns_per_program: int = 7
    arrival_rate: float = 0.3  # requests per second
    target_tokens: int = 20000
    input_growth: int = 1000
    output_tokens: int = 16
    tool_ratio: float = 0.7
    tool_time_mean: float = 0.5  # seconds
    tool_time_std: float = 0.2   # seconds
    seed: int = 42
    
    def generate_programs(self) -> List[AgentProgram]:
        """生成程序列表"""
        return MultiTurnAgentDataset.generate_programs(
            num_programs=self.num_programs,
            turns_per_program=self.turns_per_program,
            arrival_rate=self.arrival_rate,
            target_tokens=self.target_tokens,
            input_growth=self.input_growth,
            output_tokens=self.output_tokens,
            tool_ratio=self.tool_ratio,
            tool_time_mean=self.tool_time_mean,
            tool_time_std=self.tool_time_std,
            seed=self.seed,
        )
