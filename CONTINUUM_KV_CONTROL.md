# DTTL (Dynamic TTL) - Continuum KV Cache 控制实现

该文档对应当前项目内的 `KVBlocking/sglang_continuum`。这是对 [Continuum 论文](https://arxiv.org/abs/2511.02230) 的 **Dynamic TTL (DTTL)** 实现。

**重要说明**：`continuum_apply_policy_to_request()` 函数已在以下位置被调用：
- `tokenizer_manager.py` 第 1039 行（通过 `schedule_policy=continuum` 触发）
- `serving_base.py` 第 321 行（通过 `continuum_ttl_sec` 参数触发）

**核心要点**：DTTL 不是静态 TTL！它是一个**动态计算**的机制，根据历史工具执行时间来自动确定最优的缓存保留时间。

## 1) 启动服务

**方式一：使用环境变量（推荐）**

```bash
# 设置项目根目录
export SGLANG_PROJECT_ROOT="$(pwd)"

# 启动 sglang 服务
python -m sglang.launch_server ...
```

**方式二：使用 .env 文件**

```bash
source .env
python -m sglang.launch_server ...
```

**方式三：直接指定**

```bash
cd $SGLANG_PROJECT_ROOT  # 或 cd 到项目根目录
python -m sglang.launch_server ...
```

默认监听：`http://127.0.0.1:31080`

## 2) 工作原理

### 核心思想

Continuum 论文提出的核心问题是：**多轮 Agent 工作负载中的 KV Cache 管理**。

传统推理引擎在请求完成后会立即驱逐 KV Cache，但对于 Agent 工作负载（LLM 调用与工具调用交错执行），这会导致：
1. **Prefill/Reload 开销**：下次请求需要重新计算 KV
2. **Per-turn 排队延迟**：即使启用了 CPU 卸载，后续请求也需要等待 GPU 内存

### DTTL 机制

Continuum 引入了 **Dynamic TTL (DTTL)** 机制：

```
请求完成（带 tool call）
    ↓
记录工具执行时间到历史
    ↓
计算最优 TTL：τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ
    ↓
在 TTL 窗口内保留 KV Cache
    ↓
如果工具调用在 TTL 内完成 → 复用 KV，节省 Prefill 开销
如果工具调用超过 TTL      → 自动驱逐 KV，防止内存阻塞
```

### 关键公式

论文公式：
```
τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ
```

其中：
- **P(τ,f)** = 工具 f 在 τ 时间内完成的概率（基于历史 CDF）
- **T** = 平均排队延迟
- **η** = 记忆因子 = -Corr(k, N-k)
- **Prefill-Reload** = 重载 KV 的时间开销
- **MemUsage/M** = 相对内存占用

### TTL 动态计算流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   TTL 动态计算流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 记录工具执行时间                                             │
│     • 每个 program 维护自己的工具执行历史                         │
│     • 全局维护跨 program 的工具执行历史                          │
│                                                                  │
│  2. 构建 CDF（累积分布函数）                                     │
│     • Per-tool CDF: 特定工具的历史分布                          │
│     • Per-program CDF: 该程序的历史分布                         │
│     • Global CDF: 全局空闲间隔分布                              │
│                                                                  │
│  3. 计算最优 TTL                                                │
│     • 收益 = P(τ,f) × (排队延迟 + 重载收益)                    │
│     • 成本 = 内存占用 × τ                                      │
│     • τ* = 最大化 (收益 - 成本)                                │
│                                                                  │
│  4. TTL 优先级选择                                              │
│     1. 全局工具历史 (global_tool_cdf)                          │
│     2. 程序工具历史 (tool_cdf)                                  │
│     3. 全局空闲间隔 (global_cdf)                               │
│     4. 程序空闲间隔 (program_cdf)                               │
│     5. 默认 TTL (default_fallback)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3) 配置参数（环境变量）

```bash
# TTL 默认值（秒）- 设为 0 表示无历史数据时不应用 TTL
SGLANG_CONTINUUM_TTL_DEFAULT_SEC=0

# TTL 最小值（秒）- 设为 0 表示不禁用 TTL
SGLANG_CONTINUUM_TTL_MIN_SEC=0

# TTL 最大值（秒）
SGLANG_CONTINUUM_TTL_MAX_SEC=90

# 历史记录最大长度
SGLANG_CONTINUUM_TTL_HISTORY_MAXLEN=4096

# 触发计算的样本数阈值
SGLANG_CONTINUUM_TTL_HISTORY_THRESHOLD=20

# 工具延迟补偿（秒）
SGLANG_CONTINUUM_TTL_TOOL_DELAY_SEC=0

# 工具延迟放大系数
SGLANG_CONTINUUM_TTL_TOOL_DELAY_RATIO=1.5

# 内存压力惩罚系数
SGLANG_CONTINUUM_MEMORY_PRESSURE_PENALTY=0.3
```

**重要说明**：
- TTL 只在请求包含 `tool_name` 时才应用
- 无 `tool_name` 或无历史数据时，使用默认 LRU 驱逐机制
- `DEFAULT_SEC=0` 表示无历史数据时不强制设置 TTL

## 4) 客户端使用方式

### 请求头标记

通过 `x-continuum-worker-id` header 标记请求的 program_id（用于识别同一个多轮 Agent）：

```bash
curl -s http://127.0.0.1:31080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-continuum-worker-id: agent-session-123' \
  -d '{
    "model":"Qwen/Qwen3-32B-AWQ",
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens":64
  }'
```

### 请求上下文（可选）

可以通过 header 提供额外上下文：

| Header | 说明 |
|--------|------|
| `x-continuum-worker-id` | Program/Worker 标识符 |
| `x-continuum-program-id` | 显式的 program ID（优先级更高） |
| `x-continuum-tool-name` | 当前 tool 名称（用于 per-tool TTL） |
| `x-continuum-task-type` | 任务类型 |
| `x-continuum-turn-index` | 当前轮次索引 |
| `x-continuum-turn-count` | 总轮次数 |

## 5) 逐次 load_back 大小观测

服务端会在每次真实 `load_back` 成功时写一条结构化日志，前缀为 `KV_LOAD_BACK_EVENT`：

```bash
grep 'KV_LOAD_BACK_EVENT' logs/.../sglang_serve.log
```

日志包含：
- `num_tokens` - token 数量
- `kv_cache_size_bytes/mb/gb` - KV Cache 大小
- `bytes_per_token` - 每个 token 的字节数
- `duration_sec` - load_back 耗时
- `program_id` - 请求的 program ID
- `ttl_sec` / `ttl_source` - TTL 计算结果和来源

## 6) TTL 决策日志

每次 TTL 决策会记录 `CONTINUUM_TTL_DECISION` 日志：

```bash
grep 'CONTINUUM_TTL_DECISION' logs/.../sglang_serve.log
```

日志包含：
- `program_id` - Program ID
- `tool_name` - Tool 名称
- `ttl_sec` - 计算出的 TTL 值
- `ttl_source` - TTL 来源（default/tool_cdf:xxx/program_cdf/global_cdf）
- `sample_count` - 历史样本数量
- `queue_avg` / `queue_component` - 排队延迟估计
- `eta` - 记忆因子
- `reload_benefit` - 重载收益
- `benefit` - 总收益
- `estimated_memory_pressure` - 内存压力估计

## 7) 与论文的差异

当前实现已与论文保持一致，以下是设计对照：

| 特性 | 论文 | 当前实现 |
|------|------|----------|
| 标识符 | `program_id` | `program_id`（通过 `x-continuum-worker-id` header 传递） |
| TTL 作用域 | 单个请求的 KV | 基于 program 追踪的 TTL 机制 |
| Pin 机制 | 请求级别 pin | 通过 extra_key 后缀追踪 |
| TTL 决策 | 请求完成时计算 | `continuum_apply_policy_to_request()` 在请求入口调用 |
| 调度优先级 | TTL-aware priority | 保持默认调度 |

**实现说明**：
- 论文中使用 `program_id` 来标识同一个多轮 Agent 的多个请求
- 当前实现通过 `x-continuum-worker-id` header 传递，作为 program_id 的来源
- TTL 函数 `continuum_apply_policy_to_request()` 在 `generate_request()` 和 `_do_generate()` 中被调用
