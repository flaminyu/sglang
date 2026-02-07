## 流式输入与 Prefill 处理流程总览

1. **HTTP 层聚合**（`entrypoints/openai/serving_chat.py` 等）：
  - 客户端需在请求中携带 `is_streaming_input`、`streaming_input_id`、`is_last_chunk` 字段；如果只发送增量 chunk，可额外设置 `streaming_payload_is_delta=true`，服务端会在 HTTP 层自动重建完整 prompt。
  - `_prepare_streaming_chat_request()` 负责按 `streaming_input_id` 缓存并拼接同一会话的用户消息，最后一个分片会清理状态。
  - `_convert_to_internal_request()` 将流式标记写入 `GenerateReqInput`，后续栈可基于这些标记执行差异化逻辑。

2. **分词与增量提示计算**（`managers/tokenizer_manager.py`）：
  - `_prepare_streaming_prompt_delta()` 比较当前输入与缓存的完整提示，计算 `streaming_trim_len`（需要裁掉的旧 token 数）和增量 token 序列，只把增量写入 `input_ids`。当启用 `streaming_payload_is_delta` 时，这个完整提示由 HTTP 层累积得到，客户端无需重复发送历史部分。
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
