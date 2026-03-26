# Continuum-Style Worker KV Control (Experimental)

该文档对应当前项目内的 `KVBlocking/sglang_continuum`。旧路径 `LLM/sglang_continuum` 只用于历史记录或启动辅助脚本，不再是项目主入口。

## 1) 启动服务

```bash
cd /home/comp/csgfyu/multi-agents/KVBlocking

# 当前推荐做法：直接用项目脚本或实验脚本启动。
# 若继续使用历史辅助脚本，它仍位于：
bash /home/comp/csgfyu/LLM/start_sglang_continuum_qwen32b.sh
```

默认监听：`http://127.0.0.1:31080`

## 2) Worker KV 策略接口

### 查看策略表

```bash
curl -s http://127.0.0.1:31080/continuum/worker-kv-policy | python -m json.tool
```

### 设置策略（pin 到 GPU）

```bash
curl -s -X PUT http://127.0.0.1:31080/continuum/worker-kv-policy \
  -H 'content-type: application/json' \
  -d '{"worker_id":"worker-1","action":"pin_gpu","priority":-20}'
```

### 设置策略（更容易 spill 到 CPU）

```bash
curl -s -X PUT http://127.0.0.1:31080/continuum/worker-kv-policy \
  -H 'content-type: application/json' \
  -d '{"worker_id":"worker-1","action":"spill_cpu","priority":20}'
```

### 设置策略（TTL 触发迁移）

该策略在 worker 活跃时优先保留 GPU 可复用性；当该 worker 空闲超过 `ttl_sec` 后，会自动降级为 `spill_cpu`，从而允许其 KV 在后续内存压力下被迁移到较慢层。命名空间保持稳定，因此后续请求仍可触发 load-back，而不是直接丢失历史 KV。

```bash
curl -s -X PUT http://127.0.0.1:31080/continuum/worker-kv-policy \
  -H 'content-type: application/json' \
  -d '{"worker_id":"worker-1","action":"ttl_spill","ttl_sec":20}'
```

### 触发一次驱逐（evict_once）

```bash
curl -s -X POST http://127.0.0.1:31080/continuum/worker-kv-evict \
  -H 'content-type: application/json' \
  -d '{"worker_id":"worker-1"}'
```

## 3) 客户端如何标记 slow worker

给每个请求加 header：`x-continuum-worker-id`

```bash
curl -s http://127.0.0.1:31080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-continuum-worker-id: worker-1' \
  -d '{
    "model":"Qwen/Qwen3-32B-AWQ",
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens":64
  }'
```

## 4) 当前实现范围（重要）

这是一个可用于 A/B 对比的实验控制面，当前做了：

- 按 `worker_id` 注入独立 KV 命名空间（`extra_key` 后缀 + epoch）。
- 按策略注入调度优先级（`pin_gpu` 默认更高优先级，`spill_cpu` 默认更低）。
- `evict_once` 通过递增 epoch，使旧命名空间的 KV 不再被该 worker 复用（软驱逐语义）。
- `ttl_spill` 在 worker 活跃时使用 pin 语义，在空闲超过 TTL 后自动降级为 spill 语义，同时保持同一 worker 的 KV 命名空间不变。

尚未实现：

- 对单个 worker 的“硬驱逐”节点级删除（立即从 GPU/CPU 层精准移除该 worker 的全部历史节点）。
- 对单个 worker 的“强制 GPU 常驻”硬保证（当前为优先级 + 命名空间策略）。

这版适合先做 Continuum 风格实验与对照；后续可继续在 `hiradix_cache.py` 增加 node owner 标记和定向清理逻辑，实现硬驱逐/硬 pin。

## 5) 逐次 load_back 大小观测

现在服务端会在每次真实 `load_back` 成功时写一条结构化日志，前缀为 `KV_LOAD_BACK_EVENT`。日志里包含：

- `num_tokens`
- `kv_cache_size_bytes`
- `kv_cache_size_mb`
- `kv_cache_size_gb`
- `bytes_per_token`
- `duration_sec`

因此可以直接在 `sglang_serve.log` 中检索每次 load-back 的 KV 大小：

```bash
grep 'KV_LOAD_BACK_EVENT' logs/.../sglang_serve.log
```

对于 BFCL / KV isolation / KV-long 这类实验脚本，实验目录下还会自动导出：

- `kv_load_back_event_details.csv`

这个 CSV 是从服务端真实事件日志提取出来的逐次明细，适合做“每次 load-back 的 KV cache size”分析；而 `kv_migration_events.csv` 仍然保留 metrics counter 差分得到的近似事件视图，用于兼容旧分析脚本。

如果你手头只有一个独立的 `sglang_serve.log`，也可以手工提取：

```bash
python scripts/extract_kv_load_back_events.py \
  --serve-log logs/.../sglang_serve.log
```
