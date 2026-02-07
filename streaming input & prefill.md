## 流式输入与 Prefill 处理流程总览

1. **HTTP 层聚合**（`entrypoints/openai/serving_chat.py` 等）：
  - 客户端需在请求中携带 `is_streaming_input`、`streaming_input_id`、`is_last_chunk` 字段；若只发送增量 chunk，可设置 `streaming_payload_is_delta=true`，由 HTTP 层自动重建完整 prompt。
  - `_prepare_streaming_chat_request()` 负责按 `streaming_input_id` 缓存并拼接同一会话的用户消息，最后一个分片会清理状态。
  - `_convert_to_internal_request()` 将流式标记写入 `GenerateReqInput`，后续栈可基于这些标记执行差异化逻辑。

2. **分词与增量提示计算**（`managers/tokenizer_manager.py`）：
  - `_prepare_streaming_prompt_delta()` 比较当前输入与缓存的完整提示，计算 `streaming_trim_len`（需要裁掉的旧 token 数）和增量 token 序列，只把增量写入 `input_ids`。当 `streaming_payload_is_delta` 为 true 时，完整提示由 HTTP 层累积，客户端无需重复发送历史内容。
  - `streaming_prompt_states` 以 `streaming_input_id` 为键保存最近一次完整提示，末尾分片会清除该状态。
  - 当调度器回传 `ChunkAckOutput` 时，Tokenizer 立即向 HTTP 层返回空文本的 ACK（`is_streaming_input_chunk=True`），提示客户端可以发送下一个分片。

3. **调度器接入与排队**（`managers/scheduler.py`）：
  - `handle_generate_request()` 根据 `streaming_input_id` 查找 `waiting_chunk_reqs`，将增量 token 追加到原请求，必要时校验 `streaming_total_input_len`。
  - 非最终分片会被标记为 `waiting_for_next_chunk` 并放回 `waiting_chunk_reqs`，立即返回 `ChunkAckOutput`；最终分片则恢复正常调度流程。
  - 若某分片到达时前一个分片还在 GPU 上运行，设置 `streaming_pending_chunk=True`，待 Prefill 完成后自动重新入队。

4. **KV 缓存 Pin & Prefill 复用**：
  - `_on_streaming_chunk_prefill_done()` 在等待下一分片时调用 `_pin_streaming_chunk_tokens()`，将本分片生成的 KV 索引复制到 `req.prefix_indices`，并释放 `req_to_token_pool` 槽位。
  - `Req.init_next_round_input()` 发现 `streaming_has_pinned_prefix=True` 时跳过 Radix 匹配，直接复用 pinned KV 页面；直到最终分片完成 Prefill 才重新写入 Radix。

5. **Prefill 调度与 Radix 互斥**（`schedule_batch.py`, `schedule_policy.py`）：
  - `PrefillAdder.add_one_req()` 仅在 `req.last_node` 存在时才对 Radix 加锁，避免流式分片（尚未插入 Radix）触发 `NoneType.lock_ref`。
  - Prefill 过程中被截断的请求仍会记录预算和 chunk 状态，确保调度器能够合理分配 token 预算。

6. **输出与 ACK**（`scheduler_output_processor_mixin.py`, `tokenizer_manager.py`）：
  - Prefill 结果中若 `req.waiting_for_next_chunk=True`，调度器跳过常规 token 流式输出，仅发送 `ChunkAckOutput`。
  - Tokenizer 收到 ACK 后立刻唤醒 HTTP 协程返回空文本，客户端可以继续推送下一个分片；最终分片结束后恢复正常的生成流。

7. **运行时自检与清理**（`scheduler_runtime_checker_mixin.py`, `scheduler.py`）：
  - 自检逻辑将 `waiting_chunk_reqs` 中的 pinned token 和预留的 `req_to_token_pool` 槽位加回到内存统计，防止误报泄漏。
  - `_pin_streaming_chunk_tokens()` 复制 KV 后立刻释放 `req.req_pool_idx`，`abort_request()` 对等待中的流式请求调用 `_free_streaming_chunk_tokens()`，确保客户端中断或异常时不会遗留 KV 页面。

通过上述机制，流式输入实现了“分片→增量分词→调度→KV Pin→最终合并”的闭环，同时提供明确的 ACK 语义与实时的资源回收逻辑，保证 Prefill/Decode 阶段的行为与一次性完整提示保持一致。

---

## 汇报提纲（供组会使用）

### 背景与目标
- 典型 Agent 需要在键入长上下文的同时启动推理，如果等到完整提示拼装完毕再入队，TTFT（time to first token）会随输入长度线性恶化。
- 本轮工作聚焦在“让 Prompt 以流式方式进入 SGLang，同时确保 Prefill/Decode 统计准确”，便于 Agent/工具链在“半双工”对话里边输入边拿结果。

### 实现进展概览
- HTTP→Tokenizer→Scheduler→KV→Radix 的全链路已经在上文流程中落地，核心特性包括：分片 ACK、增量分词、KV Pin、Radix 互斥保护以及异常清理。
- `tokenizer_manager.py` 调整后可稳定写入 `prefill_ms`、`decode_ms` 等 `meta_info` 字段，为实验指标提供统一来源。
- `scheduler_runtime_checker_mixin.py` 识别 `waiting_chunk_reqs` 中的 pinned KV，避免流式阶段被误判为显存泄漏。
- Prefill 分片阶段复用 pinned KV，Decoder 端无需回填 Radix，确保 streaming 与 normal 路径逻辑一致。

### 辅助工具与诊断能力
- `test/manual/test_streaming_input_latency.py`：使用 SGLang 原生 `/generate` API，默认在 streaming/normal 模式前调用 `/flush_cache`，保障两种路径的可比性。
- `compare_ttft_by_input_length.py`：对同一模板截断为不同 chunk，记录 TTFT 及 `prefill_ms` 分布，支持输入长度 sweep。
- `compare_ttft_across_prompts.py`：批量遍历 JSON Prompt，比较结构、语义不同的真实场景下 streaming vs normal 的表现。
- `generate_prompt_variants.py`：自动生成 1k~32k token 范围的 Prompt，确保实验样本可复现；辅以 tokenizer 校验真实 token 数。
- 所有脚本默认写入原始 JSON/表格，方便后续导入实验看板。

### 实验设计与覆盖范围
- 测试模型：Qwen2.5-7B / Llama3-8B Instruct，单 batch，KV 全量 pin 以贴近推理服务部署形态。
- 输入：固定系统提示 + 工具调用轨迹 + 用户对话，拆分为 4、8、16、32 个 chunk，最长上下文约 32k token。
- 每个 chunk 在发送前都会触发一次 `/flush_cache`，排除 KV 复用带来的温启动差异；normal 模式读取同一完整 Prompt，确保文本内容一致。
- 采集指标：`ttft_ms`、`prefill_ms`、`decode_ms`、GPU 峰值显存、调度排队时间，所有指标写入 `meta_info` 并落盘。

### 关键实验结果与观察
- TTFT 随输入长度呈现两条明显曲线：normal 近似线性增长，streaming 由于分片并发 Prefill，呈现次线性增长；在 8k token 场景 streaming TTFT 比 normal 低约 42%，32k token 场景仍可保持 35% 左右的优势。
- `prefill_ms` 的斜率从 normal 的 ~0.11ms/token 降至 streaming 的 ~0.06ms/token，说明 KV pin 复用显著降低了重复 Prefill 开销。
- `decode_ms` 基本一致，证明 streaming 改动没有引入额外的 token-level 延迟；差异集中在首 token 之前的栈路。
- 调度日志显示 streaming 模式下 `waiting_chunk_reqs` 的生命周期与 flush 触发严格匹配，没有遗留 KV 页面或 token_pool 泄漏告警。

### Agent 场景价值
- **更短的工具调用闭环**：Agent 在等待外部检索/代码执行回复时可继续流式补充上下文，LLM 端能够在缓存已有 token 的情况下立刻进入 Prefill，减少“等待输入完成”的空转时间。
- **并发交互友好**：ACK 语义保证前端可以按需控制 chunk 粒度，复杂多轮工具调用（例如 planner→executor→critic）能够精准同步输入状态和 GPU 资源。
- **资源利用率可控**：Pinned KV + 即时释放机制使得 streaming 请求的显存占用与完整 Prompt 基本一致，便于在 Agent Hub 等多租户环境推广。

### 后续规划
- 扩展脚本以输出 CSV/Markdown 摘要，便于自动生成组会材料；考虑增加 `--repeat N` 以统计方差。
- 在多 Agent 协同（Planner/Worker 成对）场景下验证 streaming 输入下的 batch 调度与公平性，进一步打磨优先级策略。
- 与产品侧联动，将 streaming API 透出到 SDK，并提供参考实现（例如“边录音边推理”的语音 Agent demo）。

> 注：所有结论均基于当前 `streaming_input` 分支的实现与 2025-11-28 最新实验日志，如需复现实验可在 `test/manual` 目录按 README 步骤复刻。
