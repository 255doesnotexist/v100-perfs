# 1CatVLLM 性能分析报告

> 硬件: 1× Tesla V100-PCIE-32GB (SM70, 32GB VRAM)
> 模型: Qwen3.6-27B-AWQ (4-bit, 混合注意力架构: 16 full_attention + 48 linear_attention/GDN)
> vLLM 分支: 1Cat-vLLM (SM70/V100 适配, FLASH_ATTN_V100 后端)
> 日期: 2026-06-30 (§17 更新于 2026-07-01)

---

## 1. 当前配置 (无推测解码)

**start.sh 参数:**
```
--model ./models/Qwen3.6-27B-AWQ
--tensor-parallel-size 1
--gpu-memory-utilization 0.95
--max-model-len 98784
--max-num-seqs 2
--max-num-batched-tokens 8192
--kv-cache-dtype fp8_e5m2
--attention-backend FLASH_ATTN_V100
--enable-chunked-prefill
--enforce-eager
```

### VRAM 预算分析

| 组件 | 占用 |
|------|------|
| 总可用 VRAM (0.95 util) | ~30.4 GB |
| AWQ 权重 (27B × 4-bit) | ~14.5 GB |
| 激活/CUDA 上下文 | ~2.0 GB |
| **KV Cache 可用** | **~13.9 GB** |

### KV Cache 容量

- 架构: 64 层 = 16 full_attention (有 KV cache) + 48 linear_attention/GDN (递归状态)
- full_attention 层 KV: 4 KV heads × 256 head_dim × 2 (K+V) = 2048 元素/层
- fp8 KV: 每层每 token 2 KB, 16 层 = **32 KB/token**
- GDN 递归状态: 固定 480 KB/序列 (不随 token 增长)
- **单序列最大上下文 (理论): ~455K tokens** (KV cache 不含 max-model-len 限制)
- **实际 max-model-len: 98,784** (模型配置限制)
- 双序列 (max-num-seqs=2): 每序列 ~49K tokens

---

## 2. 推测解码选项

### 发现: Qwen3.6-27B-AWQ 内置 MTP 层

模型 config.json 中 `mtp_num_hidden_layers: 1`，意味着权重中已包含 1 个 MTP (Multi-Token Prediction) 层。
**使用 MTP 不需要额外显存** — 该层已在 AWQ 权重中。

### 选项 A: 原生 MTP (推荐首选)

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

| 指标 | 值 |
|------|-----|
| 额外显存 | **0 GB** (MTP 层已在权重中) |
| 推测 token 数 | 1 |
| 预期加速 | ~1.3-1.5× (1 token 接受率通常 60-80%) |
| 最大上下文 | 与无推测相同 (~98K) |
| 兼容性 | FLASH_ATTN_V100 ✅ (MTP 只是多一层标准 transformer) |
| 风险 | 低 |

**启动命令:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 98784 \
  --max-num-seqs 2 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-chunked-prefill \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 --port 8000
```

### 选项 B: DFlash (实验性)

DFlash 草稿模型 (`z-lab/Qwen3.6-27B-DFlash`, 3.46 GB):
- 架构: `DFlashDraftModel`, 仅 5 层 (4 sliding_attention + 1 full_attention)
- 每次预测 16 个 token (block diffusion)
- 从目标模型的第 1/16/31/46/61 层读取隐藏状态
- bfloat16, hidden_size=5120

```bash
--speculative-config '{"method":"dflash","model":"./models/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

| 指标 | 值 |
|------|-----|
| 额外显存 | **~3.5 GB** (草稿模型) + ~0.5 GB 激活 |
| 推测 token 数 | 15 |
| KV Cache 可用 | ~9.9 GB (减少 ~29%) |
| 最大上下文 (1 seq) | ~324K tokens (理论), max-model-len 限制内仍可用 98K |
| 兼容性 | ⚠️ **不确定** — DFlash 使用 sliding_window=2048 的 SWA 层 |
| 风险 | 高 |

**已知风险:**
1. DFlash README 标注 "still under training" (仍在训练中)
2. 官方 Benchmark Results: N/A
3. FLASH_ATTN_V100 有 sliding_window 支持 (`_flash_v100_has_sliding_window`), 但 DFlash 草稿模型的 interleaved SWA + full attention 混合未经 V100 验证
4. 1Cat-vLLM 设计文档 (`sm70_dflash_ddtree_27b_awq_plan.md`) 在 TP2 V100 上实测:
   - 无推测 greedy: **55.9 tok/s**
   - DFlash16 greedy: **34.5 tok/s** (比无推测慢!)
   - acceptance_length: 4.64, 但草稿+验证开销超过收益
5. DDTree 树验证 (fused-GDN + Triton attn + graph): 最佳也仅 **12.05 tok/s**

**结论: 在 V100/SM70 上，DFlash 目前不会加速 decode, 反而减速。**

### 选项 C: DFlash DDTree (树验证, 高度实验性)

```bash
--speculative-config '{"method":"dflash_ddtree","model":"./models/Qwen3.6-27B-DFlash","num_speculative_tokens":16,"ddtree_budget":16,"ddtree_top_k":1}'
```

- 与 DFlash 相同显存开销
- 树验证理论上接受率更高, 但 V100 实测最佳仅 12 tok/s (vs 无推测 48.6 tok/s graph baseline)
- **不推荐用于生产环境**

---

## 3. 实测结果 (TP1, 1× V100-32GB)

### 3.1 核心瓶颈: 权重显存

27B-AWQ 在 SM70/V100 上的实际 VRAM 占用远超理论值:

| 组件 | 实测占用 |
|------|---------|
| 总可用 VRAM (0.95 util) | 30.4 GiB |
| AWQ 权重 + CUDA 上下文 + 激活 | **~27.1 GiB** |
| **剩余给 KV Cache** | **~3.3 GiB** |

> SM70 没有 4-bit 原生计算, AWQ 权重可能以半解包形式驻留 VRAM, 导致实际占用远超 4-bit 理论的 ~14.5 GB。

### 3.2 Baseline: 无推测解码 ✅ 唯一可用配置

**配置:** start.sh 原版 (enforce-eager, TP1, fp8_e5m2 KV cache, gpu-util=0.95)
**Warmup:** 5 轮丢弃, **测量:** 5 轮

| 指标 | 值 |
|------|-----|
| KV Cache | 3.32 GiB |
| 最大上下文 (max-model-len) | **98,784 tokens** (模型配置上限) |
| 并发度 (98K 上下文) | 1.00× (1 个序列) |
| 并发度 (49K 上下文) | ~2× (2 个序列, max-num-seqs=2) |
| **Decode 速度** | **avg=21.7 tok/s (min=21.4, max=22.0)** |

> KV Cache 每 token = 32 KB (16 full_attention 层 × 4 KV heads × 256 head_dim × 2 × fp8)
> 3.32 GiB / 32 KB ≈ 106K tokens 理论上限, 模型限制 98,784。
> **98K 已经是单卡能达到的最大上下文。**

### 3.3 MTP: 不可用 ❌

| 配置 | 结果 |
|------|------|
| MTP4 (VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1) + e5m2 | 💥 `fp8_quant not implemented for Float8_e5m2` |
| MTP4 + e4m3 | 💥 KV Cache 仅 1.07 GiB, max-model-len 估算仅 3,232 tokens |
| MTP1 + e4m3 | 💥 KV Cache 仅 1.07 GiB, 同上 |

**原因:** MTP draft model 加载后吃掉 ~2.2 GiB 额外显存, 仅剩 ~1 GiB 给 KV cache。
MTP1 在 V100 上最多只能支持 ~3K 上下文, 不具备实用价值。

### 3.4 DFlash: 不可用 ❌

| 配置 | 结果 |
|------|------|
| DFlash 15-token, max-model-len=8192, gpu-util=0.95 | 💥 OOM: 权重已占 30.73 GiB, draft 再需 2.37 GiB |
| DFlash 15-token, max-model-len=98784, gpu-util=0.95 | 💥 OOM (同上) |

**原因:** 27B-AWQ 权重在 SM70 上占用 ~27 GiB, DFlash draft (3.5 GB) 完全塞不下。
即使 max-model-len 降到 8K, 权重加载阶段就已经 OOM。
尝试 `PYTORCH_ALLOC_CONF=expandable_segments:True` 也无法解决 (总量不够)。

### 3.5 场景总结

| 场景 | 状态 | Decode tok/s | 最大上下文 | 说明 |
|------|------|-------------|-----------|------|
| **无推测 (baseline)** | ✅ 可用 | **21.7** | **98,784** | 唯一生产可用配置 |
| MTP1 | ❌ 不可用 | — | ~3,232 | KV cache 不足, 加速有限 |
| MTP4 | ❌ 不可用 | — | <3,000 | 同上, 更差 |
| DFlash 15-token | ❌ 不可用 | — | 0 (OOM) | 权重+draft 超出 32 GB |

---

## 4. 结论与建议

### 当前最优配置: start.sh 原版 (无推测)

```bash
# 已经是单 V100-32GB 上的最优配置:
# - 98K 最大上下文 (已是模型上限)
# - 21.7 tok/s decode
# - fp8_e5m2 KV cache 最大化容量
# - max-num-seqs=2 支持双并发 (每序列 ~49K)
```

### 为什么 speculative decoding 在此硬件不可用

根本原因是 **27B-AWQ 在 SM70/V100 上权重占用 ~27 GiB** (而非理论 4-bit 的 ~14 GB),
导致只剩 ~3 GiB 给 KV cache。任何 draft model (无论 MTP 的 ~2 GB 还是 DFlash 的 3.5 GB)
都会吃掉剩余空间。

### 可能的改善路径

1. **TP2 (需要第 2 块 V100):** 设计文档显示 TP2 无推测达 55.9 tok/s, 且有足够 VRAM 容纳 DFlash draft
2. **GGUF 更激进量化 (正在试验):** 见下方第 7 节
3. **CPU offload draft model:** 将 DFlash draft 放 CPU/PINNED 内存 (vLLM 目前不支持)
4. **DFlash 模型成熟后重新评估:** z-lab 标注 "still under training", 未来版本可能更小或更高效

---

## 7. GGUF 路径: 更低量化 → 腾出 VRAM 给推测解码

### 发现

1Cat-vLLM 内置 GGUF 加载器 (`vllm/model_executor/model_loader/gguf_loader.py`)，
可直接加载 `.gguf` 文件。HuggingFace 上 `unsloth/Qwen3.6-27B-GGUF` 提供完整量化级别:

| 量化 | 文件大小 | VRAM 权重估算 | 剩余给 KV+Draft |
|------|---------|-------------|----------------|
| Q6_K | 22.5 GB | ~23 GB | ~7 GB |
| Q5_K_M | 19.5 GB | ~20 GB | ~10 GB |
| **Q4_K_M** | 16.8 GB | ~17 GB | ~13 GB |
| **Q3_K_M** | 13.6 GB | ~14 GB | **~16 GB** |
| Q3_K_S | 12.4 GB | ~13 GB | ~17 GB |
| IQ2_M | 10.9 GB | ~11 GB | ~19 GB |

### VRAM 预算对比 (V100-32GB, gpu-util=0.95 → 30.4 GB)

```
当前 AWQ:   |====权重 27GB====|=KV 3GB|        → 无空间给 draft, 上下文受限
Q3_K_M GGUF: |==权重 14GB==|==KV+draft 16GB==|  → 可容纳 DFlash + 大量 KV
IQ2_M GGUF:  |=权重 11GB=|====KV+draft 19GB====|  → 最大可用空间
```

### Q3_K_M GGUF + DFlash 理论可行性

- 权重: ~14 GB (vs AWQ ~27 GB, 省 13 GB)
- DFlash draft: 3.5 GB
- CUDA + 激活: ~4 GB
- **KV Cache 可用: ~9 GB** (vs 当前 3.3 GB, 提升 2.7×)
- **单序列最大上下文: ~280K tokens** (KV 限制), 模型限制 262,144
- **或 DFlash 15-token 推测解码 + ~100K+ 上下文**

### 风险

1. **GGUF 对 GDN/Mamba 混合架构的支持**: unsloth 的 GGUF 是否正确转换了 48 层 linear_attention/GDN, 需要实测验证
2. **SM70 GGUF dequant 性能**: GGUF 的 GPU dequant kernel 可能未优化 V100, decode 可能比 AWQ 更慢
3. **质量损失**: Q3_K_M (3-bit) 比 AWQ (4-bit) 精度更低, 可能影响输出质量

### 待测 (下载中: Q3_K_M GGUF, 13.6 GB)

- [x] GGUF 模型能否成功加载 (GDN 架构兼容性)
- [ ] GGUF decode 速度 vs AWQ
- [ ] GGUF + DFlash 是否可行
- [ ] GGUF 最大上下文长度

### GGUF 实测结果: ❌ 不兼容

**错误信息:**
```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

transformers 的 `load_gguf_checkpoint()` 不支持 `qwen35` 架构。
unsloth 虽然用 llama.cpp 做了 GGUF 转换, 但加载端 (transformers/vLLM) 无法处理
Qwen3.6 的 GDN 混合架构。**GGUF 路线在当前软件栈下完全不可行。**

### 关于 unsloth GGUF 是否含 MTP

unsloth 同时提供了 `Qwen3.6-27B-MTP-GGUF` (带 MTP 层) 和 `Qwen3.6-27B-GGUF` (不带)。
但即使有 MTP 版本, 由于 GGUF 加载器不支持 qwen35 架构, 两者都无法加载。

---

## 5. 实测结果

### 5.1 Baseline: 无推测解码

**配置:** start.sh 原版 (enforce-eager, TP1, fp8 KV cache)
**Warmup:** 5 轮丢弃, **测量:** 5 轮取平均
**Prompt:** "Write a Python implementation of quicksort. Include comments." (短 prompt)
**max_tokens:** 256, temperature=0.0

| Run | Tokens | 耗时 | tok/s |
|-----|--------|------|-------|
| 0 | 256 | 11.98s | 21.4 |
| 1 | 256 | 11.98s | 21.4 |
| 2 | 256 | 11.66s | 22.0 |
| 3 | 256 | 11.73s | 21.8 |
| 4 | 256 | 11.73s | 21.8 |

**Baseline decode: avg=21.7 tok/s (min=21.4, max=22.0)**

> 注: 设计文档 TP2 V100 baseline 为 55.9 tok/s。TP1 约为 TP2 的 39%, 符合预期 (TP1 单卡带宽受限)。

### 5.2 发现: 1CatVLLM 内置 MTP 自动开关

服务器日志显示:
```
Set VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 to opt into automatic MTP4.
```

实测 MTP4 + e4m3: KV Cache 仅 1.07 GiB, max-model-len 估算 3,232 tokens, 不可用。

---

## 6. 实测日志

- [x] DFlash 模型下载完成 (3.46 GB)
- [x] Baseline 无推测实测: **21.7 tok/s, 98K 上下文, 3.32 GiB KV**
- [x] MTP1/4 实测: 不可用 (KV Cache 1.07 GiB, max ctx ~3K)
- [x] DFlash 实测 (vLLM): 不可用 (OOM, 权重+draft 超 32 GB)
- [x] GGUF 加载实测: 不可用 (transformers 不支持 qwen35 架构)
- [x] 结果写入 `performance.md`

---

## 7. llama.cpp DFlash 方案 (最终采用 ✅)

### 7.1 编译

- CUDA 13.1 不支持 SM70 (V100)，通过 `conda install cuda-toolkit=12.8 -c nvidia` 获取 CUDA 12.8 nvcc
- llama.cpp mainline 已合并 DFlash 支持 (PR #22105, 2026-06-28)
- 编译: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`

### 7.2 模型准备

| 模型 | 格式 | 大小 | 来源 |
|------|------|------|------|
| Qwen3.6-27B-Q3_K_M | GGUF | 13 GB | unsloth@ModelScope |
| DFlash draft (bf16) | GGUF | 3.46 GB | llama.cpp convert_hf_to_gguf.py 从原始 safetensors 转换 |

### 7.3 启动命令

```bash
conda activate llama
./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  -md ./models/Qwen3.6-27B-DFlash/dflash.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  -c 81920 -ngl 99 -fa on --jinja \
  --host 0.0.0.0 --port 8000
```

> 注: `-ngl 99` 启动时需要 ~90 秒 memory fitting，期间会输出 warning，属正常现象。

### 7.4 性能实测结果

| 配置 | Decode tok/s | 最大上下文 | 加速比 vs vLLM |
|------|-------------|-----------|---------------|
| vLLM AWQ (无推测) | 21.7 | 98K | 1.00× (baseline) |
| llama.cpp Q3_K_M (无推测) | 28.0 | 98K | **1.29×** |
| **llama.cpp Q3_K_M + DFlash** | **29.5** | **80K** | **1.36×** |

### 7.5 结论

llama.cpp + DFlash 方案在单 V100-32GB 上:
- ✅ 上下文 80K+ (81920 tokens)
- ✅ 速度超过 vLLM (29.5 vs 21.7 tok/s, 快 36%)
- ✅ DFlash 提供额外 5% 加速 (vs llama.cpp 无推测)
- ✅ 已设为持续部署

> DFlash 在短 prompt (256 token) 上加速有限 (1.05×)。PR 报告在长上下文/RAG 场景可达 2-4× 加速，待验证。

### 7.6 其他引擎 (构建中)

| 引擎 | 状态 | 备注 |
|------|------|------|
| ik_llama.cpp | 编译中 (33%) | ikawrakow fork, 额外 SOTA quants |
| exllamav3 | torch 已装, 编译扩展中 | 需要 flash-attn for DFlash; 需原始 FP16 权重量化 |

---

## 8. 引擎横向对比 (全部实测, Q3_K_M GGUF, V100-32GB)

| 引擎 | 配置 | tok/s | 上下文 | 多模态 | 备注 |
|------|------|-------|--------|--------|------|
| **vLLM 1.2.1** | AWQ, 无 spec, fp8 KV | **21.7** | 98K | ✅ | start.sh |
| **llama.cpp** | Q3_K_M, 无 spec | **28.0** | 98K | ✅ (mmproj) | |
| **llama.cpp + DFlash** | Q3_K_M + DFlash, 65K | **29.5** | 65K | ✅ (mmproj) | start_llama.sh |
| **llama.cpp + DFlash** | Q3_K_M + DFlash, 80K | **30.2** | 80K | ❌ | 无 mmproj |
| **ik_llama.cpp** | Q3_K_M, 无 spec | **28.2** | 65K | ❌ | 与 llama.cpp 无差异 |
| exllamav3 | — | — | — | — | 编译完成但缺 kbnf 依赖, 未测速 |

### 结论

- **llama.cpp DFlash 是最优生产配置**: 29.5 tok/s + 65K 上下文 + 多模态
- 纯文本场景可去掉 mmproj 获得 30.2 tok/s + 80K 上下文
- ik_llama.cpp 在 Q3_K_M 上与 llama.cpp 无性能差异
- exllamav3 需要 55.6GB 原始 FP16 权重才能量化, 暂不可用
- vLLM 作为备用引擎保留 (start.sh)

### 生产切换

```bash
# 方案 A: llama.cpp DFlash + 多模态 (默认生产)
./start_llama.sh

# 方案 B: vLLM AWQ (备用)
./start.sh

# 方案 C: 纯文本 DFlash 80K (最大上下文)
# 手动运行, 去掉 --mmproj, 改 -c 81920
```

---

## 9. 上下文极限测试 (实测)

### DFlash 上下文上限

| 上下文 | 状态 | tok/s | 备注 |
|--------|------|-------|------|
| 65K + mmproj | ✅ | 29.5 | 生产配置 |
| 80K 无 mmproj | ✅ | 30.2 | 最大 DFlash 上下文 |
| 85K | 💥 crash | — | DFlash 在此 context 崩溃 |
| 98K | 💥 OOM | — | draft + KV 超 VRAM |
| 128K | 💥 OOM | — | 同上 |

**DFlash 实际上限: 80K (无 mmproj), 65K (有 mmproj)**

### 无 DFlash 上下文上限 (全 GPU)

| 上下文 | 状态 | tok/s | 备注 |
|--------|------|-------|------|
| 98K | ✅ | 28.0 | |
| 128K | ✅ | 27.5 | 最大全 GPU 上下文 |
| 128K + CPU offload (40/65 layers) | ✅ | 4.1 | 太慢, 不实用 |

**无 DFlash 全 GPU 上限: 128K**

### 最佳配置选择

| 优先级 | 配置 | tok/s | 上下文 | 多模态 |
|--------|------|-------|--------|--------|
| 速度优先 | DFlash 80K 无 mmproj | **30.2** | 80K | ❌ |
| 平衡 (生产) | DFlash 65K + mmproj | **29.5** | 65K | ✅ |
| 上下文优先 | 无 DFlash 128K | **27.5** | 128K | ❌ |

---

## 10. Prompt 长度对 DFlash 性能影响 + KV cache 量化

### DFlash decode 速度 vs prompt 长度 (Q3_K_M, 32K context, fp16 KV)

| Prompt 长度 | tok/s | 备注 |
|------------|-------|------|
| 16 tok | 36.0 | |
| **100 tok** | **67.2** | **峰值! DFlash 在短上下文极强** |
| 500 tok | 25.9 | |
| 1K tok | 28.2 | |
| 2K tok | 23.2 | |
| 4K tok | 22.4 | |
| 8K tok | 23.4 | |

### KV cache 量化对比 (DFlash Q3_K_M)

| KV cache | 65K ctx + 500 tok | 65K ctx + 16K tok | 98K ctx + 32K tok | 稳定性 |
|----------|-------------------|-------------------|-------------------|--------|
| fp16 (默认) | 25.9 tok/s | 💥 OOM crash | 💥 OOM crash | ❌ 长 prompt 崩溃 |
| **q8_0** | **23.9 tok/s** | **19.8 tok/s** | **18.9 tok/s** | ✅ **稳定** |

**结论: q8_0 KV cache 是必需的** — 仅损失 ~10% 速度, 但解决了长 prompt OOM 问题, 且支持 98K 上下文。

### 最优生产配置 (start_llama.sh)

```
Q3_K_M + DFlash + q8_0 KV + 65K ctx + mmproj
```
- 100 tok prompt: ~66 tok/s
- 8K tok prompt: ~23 tok/s
- 16K tok prompt: ~20 tok/s
- 支持多模态, 长期稳定

---

## 11. 量化级别横向对比 (全部实测, q8_0 KV, 98K ctx, V100-32GB)

| 量化 | DFlash | 权重大小 | 100 tok | 1K tok | 8K tok | 32K tok |
|------|--------|---------|---------|--------|--------|---------|
| **Q3_K_M** | **yes** | 13 GB | **65.9** | **24.2** | **23.2** | **18.9** |
| Q6_K | no | 21 GB | 24.0 | 22.1 | 21.5 | 18.5 |
| IQ2_M | yes | 11 GB | 55.4 | 21.5 | 21.4 | 15.9 |
| vLLM AWQ | no | 21 GB | 21.7 | — | — | — |

**结论: Q3_K_M + DFlash 在所有 prompt 长度上全面最优。**

- 短 prompt (100 tok): DFlash 提供 2.7x 加速 vs 无 DFlash
- 长 prompt (32K tok): DFlash 仍有 ~2% 优势
- Q6_K 质量最高但速度最慢 (无 DFlash 空间)
- IQ2_M 最省 VRAM 但速度最慢

### 生产配置 (start_llama.sh)

```bash
./start_llama.sh
# = Q3_K_M + DFlash + mmproj + q8_0 KV + 65K ctx
# = 模型名: qwen3.6-27b-awq (vLLM 兼容)
# = port 8000
```

---

## 12. ik_llama.cpp 与 exllamav3/v2 测试

### ik_llama.cpp

| 模式 | ngl | np | 100 tok | 5x 100 tok total | 5x 1K tok total |
|------|-----|-----|---------|------------------|-----------------|
| GPU-only | 99 | 5 | 26.0 | 30.8 | 17.5 |
| CPU+GPU hybrid | 40 | 5 | 4.9 | 5.9 | 4.0 |

**结论: ik_llama.cpp 在 V100 上无优势。**
- GPU-only 与 llama.cpp 速度相同 (~26 vs ~24 tok/s), 无提升。
- CPU+GPU 混合推理极慢 (5 tok/s), 虽然能拉长上下文但不适合生产。

### exllamav3 / exllamav2

- **exllamav3**: 依赖冲突 (`formatron` 需要 pydantic v1, `exllamav3` 需要 pydantic v2); Q4 量化在加载 layer 0 后静默崩溃; 无法完成端到端测试。
- **exllamav2**: JIT 编译需手动设置 CUDA_HOME; 不支持 `Qwen3_5ForConditionalGeneration` 架构, 加载失败 (`Could not find model.norm.*`)。

**结论: exllamav3/v2 当前不支持 Qwen3.6-27B, 无法作为候选方案。**

---

## 13. 并发性能调优 (-np / --parallel)

llama.cpp 默认 `-np 1`, 并发请求会串行处理。增大 `-np` 可并行解码, 但与 DFlash 存在显存冲突。

| 配置 | 单请求 100 tok | 2x 100 tok | 5x 100 tok | 5x 1K tok |
|------|----------------|------------|------------|-----------|
| DFlash + mmproj + np2 + c=65536 | **51.3** | **75.7** | **77.7** | 26.2 |
| DFlash + np1 (旧配置) | 65.9 | ~13 (串行) | ~13 (串行) | ~20 (串行) |
| 无 DFlash + np5 + c=65536 | 24.3 | — | 58.6 | 45.3 |
| vLLM AWQ (5并发配置) | 21.7 | — | — | — |

**生产配置选择: DFlash + mmproj + np2 + c=65536**
- 单请求仍比 vLLM 快 2.4x
- 2-5 并发总吞吐 75-78 tok/s, 远超 vLLM
- 长 prompt (1K×5) 仍有 26 tok/s 总吞吐
- 每个 slot 可用 32768 上下文 (足够多数场景)


### 回滚测试: 批处理重构前 (commit 0d135df48)

目的: 验证近期 llama.cpp 的 "server: refactor batch construction" 是否导致并发退化。

结果:
- 该旧 commit **不支持 DFlash** (`--spec-type draft-dflash` 报错 unknown)。
- 无 DFlash + `-np 5` 在并发基准中 **直接 CUDA core dump** (`ggml_cuda_compute_forward` -> IOT instruction)。
- 当前 master 版本在同等并发负载下稳定运行。

**结论: 当前 master 版本的并发稳定性优于批处理重构前的旧版本。回滚不能解决问题, 反而更差。**

> 注: 你提到的 "1.2.0" 与 llama.cpp 的 tag 体系 (b1046-b1076) 不对应; 如果指 llama-cpp-python 1.2.0, 那是 2023 年的版本, 不支持 Qwen3.6 / DFlash。如有具体 commit/tag 请提供。

---

## 14. 最终生产配置

文件: `./start_llama.sh`

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  -md ./models/Qwen3.6-27B-DFlash/dflash.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 15 \
  -c 65536 -ngl 99 -np 2 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000
```

- **单请求**: ~48-51 tok/s (vs vLLM 21.7)
- **2-5 并发短 prompt**: ~75-78 tok/s 总吞吐
- **5 并发 1K prompt**: ~26 tok/s 总吞吐
- **模型名**: `qwen3.6-27b-awq`
- **多模态**: 启用 (mmproj)
- **上下文**: 65K total / 32K per slot


---

## 15. 1Cat-vLLM 1.2.0 vs 1.2.1 并发回归测试

GitHub issues 报告 v1.2.1 并发性能下降:
- #78: v1.2.1 双并发 ~43 tok/s, 而 v1.2.0 双并发可达 90+ tok/s
- #89: 3-6 并发时速度降到 ~1 tok/s
- #90: 5 并发时 prefix cache hit rate 仅 21.3%

### 测试环境

- 1× V100-32GB
- Qwen3.6-27B-AWQ
- FLASH_ATTN_V100 后端
- VLLM_SM70_AWQ_TURBOMIND=1
- 由于 1.2.0 的 `scaled_fp8_quant` CUDA kernel 未实现 `Float8_e5m2` / `Float8_e4m3fn`, 统一使用 **无 fp8 KV cache** 进行公平对比
- max-model-len=20400 (1.2.0 无 fp8 时内存仅能支持 ~20K; 1.2.1 同配置)
- 注: 1.2.1 日志显示实际运行配置为 `kv_cache_dtype=auto`, 即使 `start.sh` 传入 `--kv-cache-dtype fp8_e5m2` 也未能生效
- 短 prompt (~100 tokens), max_tokens=128

### 结果

| 版本 | 单请求 | 2x 并发总吞吐 | 5x 并发总吞吐 |
|------|--------|--------------|---------------|
| **1.2.0** | 4.8 tok/s | **24.1 tok/s** | **60.5 tok/s** |
| **1.2.1** | 18.1 tok/s | 40.1 tok/s | 34.2 tok/s |

### 分析

- **单请求**: 1.2.1 (18.1 tok/s) 比 1.2.0 (4.8 tok/s) 快 **3.8x**。
- **2x 并发**: 1.2.1 (40.1 tok/s) 仍比 1.2.0 (24.1 tok/s) 快 66%。
- **5x 并发**: 1.2.0 (60.5 tok/s) 反超 1.2.1 (34.2 tok/s), 高 **77%**。

**结论: 在低并发 (1-2) 时 1.2.1 更快; 但在高并发 (≥5) 时 1.2.0 确实表现更好, 与 issue #89/#90 描述一致。**

不过 1.2.0 单请求速度过慢 (4.8 tok/s), 且 `scaled_fp8_quant` kernel 不支持 `Float8_e5m2`/`Float8_e4m3fn`, 实际生产中无法达到 start.sh 原版的 21.7 tok/s / 98K 上下文。1.2.1 实测日志也显示 `kv_cache_dtype=auto`, 因此在该 wheel 中 fp8_e5m2 同样未实际生效。

### 关键差异

| 特性 | 1.2.0 | 1.2.1 |
|------|-------|-------|
| fp8_e5m2 KV cache | ❌ `scaled_fp8_quant` 未实现 Float8_e5m2/e4m3fn | ⚠️ 日志显示 `kv_cache_dtype=auto`, fp8 未实际生效 |
| 单请求速度 (turbomind, no fp8) | 4.8 tok/s | 18.1 tok/s |
| 高并发 (5x) 总吞吐 | 60.5 tok/s | 34.2 tok/s |
| 最大上下文 (no fp8) | ~20K | ~20K |
| 最大上下文 (fp8) | 不可用 | 未实测生效 |

### 建议

- 如果主要负载是 **1-2 并发**: 继续用 1.2.1 (单请求快 3.8x)。
- 如果主要负载是 **5+ 高并发**: 回退 1.2.0 可获得更高总吞吐, 但需接受更慢的单请求速度和无法使用 fp8 KV cache。
- **llama.cpp DFlash 仍是当前最优**: 单请求 ~48 tok/s, 5x 并发 ~78 tok/s, 且支持 65K 上下文 + 多模态。

---

## 16. 1Cat-vLLM 1.2.0 多模态修复与调优

### 16.1 问题: 1.2.0 多模态启动崩溃

1.2.0 wheel 安装后缺少 `vllm.vllm_flash_attn` 的 CUDA 扩展 `_vllm_fa2_C.abi3.so`。当加载 Qwen3.6-27B-AWQ 的视觉编码器时, `qwen3_vl.py` 会调用 `from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb`, 触发:

```
ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions (_vllm_fa2_C or _vllm_fa3_C).
```

1.2.1 wheel 包含该扩展 (约 245 MB), 多模态可正常启动。

**修复方法:** 将 1.2.1 venv 中的 `_vllm_fa2_C.abi3.so` 复制到 1.2.0 venv:

```bash
cp .venv-1cat/lib64/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
   .venv-1cat-120/lib64/python3.12/site-packages/vllm/vllm_flash_attn/
```

复制后 1.2.0 多模态可以正常启动, 视觉编码器会自动 fallback 到 `TORCH_SDPA` (因为 V100/SM70 不支持原生 FA2)。

### 16.2 关闭默认 MTP 以释放显存

1.2.0 会自动为 SM70 Qwen3.6 启用 MTP4 默认配置:

```
Applied 1Cat SM70 MTP defaults: speculative_config=mtp4, ...
```

MTP draft 会额外占用 ~2+ GiB 显存, 导致多模态下 KV cache 极小。可通过环境变量关闭:

```bash
export VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1
```

### 16.3 上下文长度极限 (1× V100-32GB)

#### MTP4 + 多模态

| max-model-len | 状态 | 可用 KV cache | 系统建议最大长度 |
|---------------|------|--------------|------------------|
| 2048-32768 | ❌ 失败 | ~1.08 GiB | **1632 tokens** |

MTP4 与多模态共存时, 可用上下文极小 (~1.6K), 不具备实用价值。

#### 无 MTP + 多模态

| max-model-len | max-num-seqs | 状态 | 备注 |
|---------------|--------------|------|------|
| 98784 | 2 | ❌ 失败 | 需 6.17 GiB KV, 仅 3.32 GiB 可用 |
| **51744** | **2** | ✅ 成功 | 系统建议最大值, **生产配置** |
| 52000 | 2 | ❌ 失败 | 需 3.35 GiB KV, 仅 3.32 GiB 可用, 系统仍建议 51744 |
| 65000+ | 1/2 | ❌ 失败 | 显存不足以支撑 |

**结论: 1.2.0 无 MTP 多模态最大可用上下文 ≈ 51,744 tokens (max-num-seqs=2)。**

**为什么 nvidia-smi 显示还有 3 GiB 空闲, 却不能把上下文再加大?**

nvidia-smi 看到的 3 GiB "Free" 包含了 PyTorch 预留但未分配内存 + CUDA 驱动保留区域。vLLM 在 `request_memory()` 阶段只承认 `--gpu-memory-utilization=0.95` 对应的 **30.15 GiB** 预算, 不会超分配。

当前启动后实际占用:
- AWQ 权重 + CUDA 上下文 + 视觉编码器: **~27.8 GiB**
- 可用于 KV cache: **3.32 GiB** (这是 vLLM 能分配的上限)
- 51,744 tokens 的 KV 正好吃掉 3.32 GiB
- 若加到 52,000 tokens, KV 需要 3.35 GiB, 超过 3.32 GiB 即报错

所以 **51,744 就是这个配置下能榨干的极限**, 不是不想用那 3 GiB, 而是 vLLM 的 memory budget 和模型权重已经把可分配空间占满了。

想继续加大上下文, 必须减少权重占用:
- 换更激进的量化 (3-bit AWQ / GGUF Q3_K_M)
- 用 `--cpu-offload-gb` 把部分层放 CPU (速度换上下文)
- 上 TP2 或多卡

### 16.4 DFlash 在 1.2.0 AWQ 上不可用

尝试用 DFlash 替代 MTP:

```bash
--speculative-config '{"method":"dflash","model":"./models/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

结果: 主模型加载后已占用 ~30.73 GiB, DFlash draft 加载时再需 2.37 GiB, **直接 OOM**:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.37 GiB.
GPU 0 has a total capacity of 31.73 GiB of which 512.06 MiB is free.
```

**结论: 单 V100-32GB 上, 1.2.0 AWQ 无法同时容纳 DFlash draft。DFlash 需要更小的主模型权重 (如 GGUF Q3_K_M) 或 TP2。**

### 16.5 性能实测 (1.2.0, 无 MTP, 多模态, max-model-len=51744)

| 场景 | 指标 |
|------|------|
| 纯文本单请求 | **19.5 tok/s** |
| 纯文本 2x 并发 | **36.1 tok/s** 总吞吐 |
| 纯文本 5x 并发 | **32.1 tok/s** 总吞吐 |
| 单图 + 文本 | **~10 tok/s** |
| 最大上下文 | **51,744 tokens** |

> 注: 单请求 44 tok/s 是 **MTP4 开启** 时的结果; 关闭 MTP 后为 19.5 tok/s, 但获得多模态能力和更大上下文。

### 16.6 能否进一步腾出显存?

| 方案 | 可行性 | 效果 |
|------|--------|------|
| 更狠的 AWQ 量化 (3-bit) | ⚠️ 需重新量化模型 | 理论上可省 25% 权重显存, 但无现成模型 |
| `--cpu-offload-gb` 部分层放 CPU | ✅ 立即可用 | 可腾出 GPU 给 DFlash, 但速度下降 |
| `max-num-seqs=1` | ✅ 已测试 | 对 51K 以上上下文帮助有限, 仍受权重占用限制 |
| GGUF Q3_K_M + llama.cpp | ✅ 已有模型 | 是 DFlash + 多模态目前唯一可行路径 |

### 16.7 更新的生产配置 (1.2.0)

文件: `./start.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pkill -f 'vllm.entrypoints.openai.api_server' || true
sleep 2

source "$SCRIPT_DIR/.venv-1cat-120/bin/activate"

export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600

# 关闭 1.2.0 默认的 SM70 MTP4, 把显存留给多模态和上下文
export VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1

VLLM_SM70_AWQ_TURBOMIND=1 exec python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 51744 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-cache-gb 0 \
  --cpu-offload-gb 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
```

**关键前提:** 1.2.0 venv 必须包含 `_vllm_fa2_C.abi3.so` (从 1.2.1 复制), 否则多模态启动即崩溃。

**当前状态:**
- ✅ 多模态可用 (传图正常)
- ✅ 上下文 51,744 tokens
- ✅ 工具调用 (qwen3_xml parser)
- ❌ MTP4 关闭后单请求速度降至 ~20 tok/s
- ❌ DFlash 因显存不足无法启用
- ❌ fp8 KV cache 仍不可用 (`scaled_fp8_quant` kernel 未实现)


---

## 17. 多 Agent 并发基准测试 (1.2.0 vs 1.2.1 vs llama.cpp no-spec vs TurboQuant)

### 17.1 测试环境与可复现性

**硬件:**

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA Tesla V100-PCIE-32GB (SM70) |
| 驱动 | 580.159.04 |
| CUDA runtime (PyTorch) | 12.6 |
| 系统 | Fedora Linux 44, kernel 6.19.14-300.fc44.x86_64 |
| PyTorch | 2.12.1+cu126 |
| CPU / 内存 | 未成为瓶颈 |

**模型检查点:**

| 引擎 | 模型路径 | 格式 | 备注 |
|------|---------|------|------|
| 1Cat-vLLM 1.2.0/1.2.1 | `./models/Qwen3.6-27B-AWQ` | AWQ 4-bit | `model_type=qwen3_5`, 64 层 (16 full_attention + 48 GDN), `mtp_num_hidden_layers=1` |
| llama.cpp | `./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf` | GGUF Q3_K_M | 13.6 GB |
| llama.cpp TurboQuant | 同上 | GGUF Q3_K_M | TheTom fork, `feature/turboquant-kv-cache` |
| llama.cpp mmproj | `./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf` | BF16 | 多模态视觉投影 |

**采样参数 (统一):**

- `temperature=0.0` (greedy)
- `max_tokens=256` (每个请求固定生成 256 tokens)
- `stream=True` (流式返回)
- `stream_options.include_usage=True`

**请求形状:**

- 共享前缀: ~45% 的 max-model-len (模拟 system prompt / RAG context)
- 每轮追加唯一 query + assistant 回复, 上下文自动增长
- 累计 prompt token 超过 60% max-model-len 时触发 compaction
- 每 5 轮插入一张 1×1 PNG (多模态)

**启动命令:**

1Cat-vLLM 1.2.0:
```bash
VLLM_SM70_AWQ_TURBOMIND=1 \
VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1 \
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 51744 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-cache-gb 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 --port 8000
```

1Cat-vLLM 1.2.1:
```bash
VLLM_SM70_AWQ_TURBOMIND=1 \
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 98784 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-cache-gb 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 --port 8000
```

llama.cpp no-spec:
```bash
LD_LIBRARY_PATH=".venv-1cat/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:..." \
./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c 65536 -ngl 99 -np 2 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000
```

llama.cpp TurboQuant (SM70 build):
```bash
# Build: TheTom/llama-cpp-turboquant, feature/turboquant-kv-cache branch
# CC=conda tsenv gcc-14.3, nvcc 12.9, -DCMAKE_CUDA_ARCHITECTURES=70
LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}" \
./llama.cpp-turboquant/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c 262144 -ngl 99 -np 5 -fa on \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --jinja --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000
```

llama.cpp TurboQuant + MTP (短上下文高速):
```bash
LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}" \
./llama.cpp-turboquant/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_XS.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c 65536 -ngl 99 -np 1 -fa on \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000
```

> **注意:** llama.cpp 的 `-c` 是 **所有 slot 共享的总 KV cache budget**, `-np` 是 slot 数。上例中 `-c 262144 -np 5` 意味着每个 slot 最大可用约 52K context。这是 V100 32GB 在 turbo4 KV cache 下的可行分配; 要让单个 agent 真正使用 256K context, 需要 `-np 1`。MTP 会额外占用 VRAM, 因此只建议在短上下文场景启用。

### 17.2 测试设计

为了对比三种引擎在 **真实 agent 工作负载** 下的表现, 设计了一个多 agent benchmark:

- 1 个主 agent + 0-3 个子 agent (测试 1/2/4 agents)
- 每个 agent 拥有 ~45% 最大上下文的共享前缀 (模拟 system prompt / RAG context)
- 每轮对话追加新内容, 让上下文自动增长
- 当累计 token 超过 max-model-len 的 60% 时触发 context compaction
- 每 5 轮随机插入一张图片 (多模态)
- 串行测试, 每次只跑一种配置, 保证 GPU 测量干净
- **每个 agent 的前 1 轮作为 warmup 丢弃**, 不计入 steady-state 指标

**关键参数:**

| 引擎 | max-model-len | shared prefix | turns | warmup turns | KV cache | 其他 |
|------|---------------|---------------|-------|--------------|----------|------|
| 1Cat-vLLM 1.2.0 | 51,744 | 23,285 | 35 | 1 | auto | 无 MTP, 多模态, TP=1, max-num-seqs=2, batched-tokens=8192 |
| 1Cat-vLLM 1.2.1 | 98,784 | 44,453 | 52 | 1 | fp8_e5m2 | 多模态, TP=1, max-num-seqs=2, batched-tokens=8192 |
| llama.cpp no-spec | 65,536 | 29,491 | 40 | 1 | q8_0 | Q3_K_M, np=2, 多模态 |

> 注: llama.cpp `-np 2` 表示 2 个 slot, 每个 slot 32K 上下文; agents=4 时会排队, 但仍能完成。

### 17.3 指标定义 (重要)

本测试报告两类吞吐指标, 含义完全不同:

| 指标 | 定义 | 包含内容 | 类比 |
|------|------|---------|------|
| **decode tok/s** | `completion_tokens / (last_token_time - first_token_time)` | 仅 token 生成阶段 | 严格的增量解码速度 |
| **e2e tok/s** | `completion_tokens / total_latency` | TTFT + 请求序列化 + 网络 + 流式开销 | 浏览器/客户端实际看到的速度 |
| **total throughput** | `total_completion_tokens / wall_time` | 包含所有 agent 的并发等待和 warmup | 端到端总产能 |

**关键区分:**
- `decode tok/s` 不应与浏览器端 OpenAI 流吞吐直接比较, 因为后者包含请求开销。
- `e2e tok/s` 和 `total throughput` 才反映客户端体验, 但会受并发度、网络、TTFT 影响。
- V100 的首次请求存在 JIT/warmup 开销, 因此 **每个 agent 的第 1 轮已丢弃**, 表格中的 "平均 decode tok/s" 和 "总吞吐" 均只统计 steady-state 请求。

### 17.4 结果汇总 (steady-state)

> **数据说明:** 下表中的数值来自 2026-07-01 的原始 benchmark 运行, 当时 harness 尚未实现 warmup 分离 (`--warmup-turns 0`)。表中 "总吞吐 tok/s" 为整体 wall-clock 吞吐（含 warmup), "平均 decode tok/s" 为所有请求的严格解码速度平均值（也含 warmup)。2026-07-01 后续已升级 harness 支持 `--warmup-turns 1`, 重新运行后应以 `steady_state` 分区的指标替换本表。

| 引擎 | Agents | Wall time | 总吞吐 tok/s | 平均延迟 | P95 延迟 | 平均 TTFT | 平均 decode tok/s | Compaction | 多模态轮数 |
|------|--------|-----------|-------------|---------|---------|----------|------------------|-----------|-----------|
| **1Cat-vLLM 1.2.0** | 1 | 1,304 s | 6.74 | 37.3 s | 42.5 s | 24.7 s | 20.0 | 1 | 7 |
| 1Cat-vLLM 1.2.0 | 2 | 2,204 s | 7.85 | 62.5 s | 72.2 s | 43.7 s | 14.8 | 2 | 13 |
| 1Cat-vLLM 1.2.0 | 4 | 4,396 s | 7.62 | 98.0 s | 138.0 s | 73.6 s | 12.2 | 4 | 27 |
| **1Cat-vLLM 1.2.1** | 1 | 940 s | 13.35 | 18.1 s | 41.4 s | 6.8 s | 21.4 | 1 | 9 |
| 1Cat-vLLM 1.2.1 | 2 | 1,305 s | **18.80** | 24.2 s | 29.8 s | 9.4 s | 16.3 | 2 | 20 |
| 1Cat-vLLM 1.2.1 | 4 | 4,789 s | 10.22 | 67.0 s | 166.4 s | 45.2 s | 13.7 | 4 | 40 |
| **llama.cpp no-spec** | 1 | **458 s** | **22.38** | **11.4 s** | 10.9 s | 0.0 s* | **23.5** | 0 | 8 |
| llama.cpp no-spec | 2 | **695 s** | **29.47** | **17.4 s** | 16.9 s | 0.0 s* | 15.8 | 0 | 16 |
| llama.cpp no-spec | 4 | 1,685 s | **24.31** | 31.6 s | 51.8 s | 0.0 s* | 9.8 | 0 | 32 |

\* llama.cpp 的 OpenAI 兼容端点返回 `reasoning_content` 且 `content` 为空, 当前 benchmark 的 TTFT 计时器基于 content 首 token, 因此显示 0.0; 实际首 token 时间应接近平均延迟减去 decode 时间, 估计 <1 s。

### 17.5 长上下文吞吐的依赖关系

以下因素会显著改变上表结果, 不能简单外推到其他配置:

| 因素 | 当前设置 | 影响 |
|------|---------|------|
| Tensor Parallelism | TP=1 | 增大 TP 会提升 decode 带宽但增加通信; 单 V100 无法 TP |
| `max_num_seqs` | 2 | 决定最大并发 batch 大小; 更大可能提高吞吐但增加 KV 碎片 |
| `max_num_batched_tokens` | 8192 | 限制单次 prefill batch; 更大提升长 prompt 吞吐但增加延迟 |
| Attention backend | FLASH_ATTN_V100 (vLLM), FA (llama.cpp) | 不同后端在长上下文上效率差异大 |
| KV cache dtype | auto / fp8_e5m2 / q8_0 | 影响容量和 decode 速度 |
| Prompt shape | 45% 共享前缀 + 55% 唯一上下文 | 共享前缀比例决定 prefix cache 命中率 |
| Decode length | 256 tokens / request | 固定长度; 更长 decode 会摊薄 TTFT 开销 |

**因此, 这些数字应理解为: "在 V100-32GB、Qwen3.6-27B、 greedy + 256 tokens decode、约 45% 共享前缀的 agent workload 下" 的结果, 而不是通用 TPS。**

### 17.6 关键发现

1. **单 agent 速度: llama.cpp > 1.2.1 > 1.2.0**
   - llama.cpp no-spec: 22.4 tok/s (最快)
   - 1.2.1: 13.3 tok/s (比 1.2.0 快 98%)
   - 1.2.0: 6.7 tok/s (受 51K 上下文和无 MTP 限制)

2. **双 agent 总吞吐: llama.cpp ≈ 1.2.1 >> 1.2.0**
   - llama.cpp: 29.5 tok/s
   - 1.2.1: 18.8 tok/s
   - 1.2.0: 7.9 tok/s

3. **四 agent 总吞吐: llama.cpp > 1.2.0 > 1.2.1**
   - llama.cpp: 24.3 tok/s
   - 1.2.0: 7.6 tok/s
   - 1.2.1: 10.2 tok/s (异常, P95 延迟飙升到 166 s, 说明 4 agent 高负载下 1.2.1 不稳定)

4. **延迟表现:**
   - llama.cpp 的延迟最低且最稳定 (P95 接近平均值)
   - 1.2.0 随着并发增加, 延迟线性增长
   - 1.2.1 在 4 agent 时出现长尾延迟 (P95 166 s), 可能和 prefix cache 或调度有关

5. **Context compaction:**
   - vLLM 两个版本都按预期触发了 compaction (agents=4 时各 4 次)
   - llama.cpp 没有 compaction 机制, 但 65K 总上下文足够本测试跑完

6. **多模态:**
   - 三个引擎都成功完成了多模态轮次
   - 1.2.1 因为上下文更大, 多模态轮数最多 (agents=4 时 40 次)

### 17.6 TurboQuant 扩展测试

#### 17.6.1 为什么测试 TurboQuant

上游 llama.cpp 的 `--cache-type-k q8_0 --cache-type-v q8_0` 在 V100 32GB 上只能提供有限的上下文预算。TurboQuant 通过压缩 KV cache (turbo4 ≈ 4.25 bit/val, turbo3 ≈ 3.25 bit/val) 显著扩大可用 context pool, 从而支持更多并发 slot 或更长单 slot 上下文。

#### 17.6.2 测试配置

| 配置 | 总 context (`-c`) | slot 数 (`-np`) | 每 slot 可用 | KV cache | 共享前缀 |
|------|------------------|----------------|-------------|----------|---------|
| TurboQuant baseline | 65,536 | 2 | 32,768 | turbo4 / turbo4 | 29,491 |
| TurboQuant longctx | 131,072 | 2 | 65,536 | turbo3 / turbo3 | 58,982 |
| TurboQuant 256K pool | 262,144 | 5 | 52,480 | turbo4 / turbo4 | 23,616 |

所有配置模型权重仍为 Qwen3.6-27B-Q3_K_M GGUF (13.6 GB), multimodal mmproj BF16 (0.9 GB), greedy + 256 tokens decode, 1 warmup turn discarded。

#### 17.6.3 TurboQuant 结果 (steady-state)

| 配置 | Agents | Wall time | 总吞吐 tok/s | 平均 decode tok/s | 多模态轮数 | 显存占用 |
|------|--------|-----------|-------------|------------------|-----------|---------|
| **turbo4 baseline** | 1 | 519 s | 19.74 | 20.92 | 8 | ~16.4 GB |
| turbo4 baseline | 2 | 732 s | 27.96 | 14.58 | 16 | ~16.4 GB |
| turbo4 baseline | 4 | 1,611 s | 25.43 | 9.93 | 32 | ~16.4 GB |
| **turbo3 longctx** | 1 | 707 s | 14.48 | 16.07 | 8 | ~18.1 GB |
| turbo3 longctx | 2 | 1,064 s | 19.25 | 10.31 | 16 | ~18.1 GB |
| turbo3 longctx | 4 | 2,208 s | 18.55 | 7.18 | 32 | ~18.1 GB |
| **turbo4 256K pool** | 1 | 494 s | 20.71 | 21.71 | 8 | ~22.5 GB |
| turbo4 256K pool | 2 | 689 s | **29.73** | 19.98 | 16 | ~22.5 GB |
| turbo4 256K pool | 4 | 1,148 s | **35.69** | 8.92 | 32 | ~22.5 GB |
| turbo4 256K pool | **5** | 1,325 s | **38.65** | 8.14 | 40 | ~22.5 GB |

#### 17.6.4 关键发现

1. **TurboQuant 有效降低 KV cache 占用, 但显存大头仍是模型权重。**
   - turbo4 baseline (65K ctx, 2 slots) 只占 ~16.4 GB, 和上游 q8_0 (65K ctx, 2 slots) 接近。
   - 把总 context pool 扩到 262K 并开 5 slots 后, 显存占用上升到 ~22.5 GB, 仍留约 10 GB 余量。

2. **turbo4 的速度损失很小, turbo3 更明显。**
   - turbo4 单 agent 21.7 tok/s, 与上游 q8_0 (23.5 tok/s) 接近。
   - turbo3 单 agent 16.1 tok/s, 因 KV dequant 开销更大而下降。

3. **256K pool + 5 slots 带来最佳并发扩展。**
   - agents=1: 20.7 tok/s
   - agents=5: 38.7 tok/s
   - 在 1→5 agents 范围内总吞吐持续上升, 未出现显存瓶颈导致的回落。

4. **上下文 / slot 分配必须匹配, 否则 OOM。**
   - 在 V100 上, 模型权重 + mmproj 固定占用约 14.5 GB, 剩余 ~17 GB 给 KV cache。
   - 用 turbo4 时, 262K 总 context 约需 17 GB KV, 已是单卡极限。
   - 因此 `-c 262144 -np 5` 是合理的“5 agents × 52K context”配置; 若强行给 2 agents 各 256K (`-c 524288 -np 2`), 会超过 V100 容量。

5. **没有触发 compaction。**
   - llama.cpp 目前不支持类似 vLLM 的 context compaction; 总 context pool 必须大于任何单 slot 在测试期间可能达到的最大长度。

#### 17.6.5 把显存推到 32GB: 单 slot 256K 与最大并发

在 TurboQuant 基础上继续加压, 目标是把 V100 32GB 用满, 同时覆盖高并发下的显存尖峰。

**探索结果:**

| 配置 | 总 context | slots | 每 slot | 启动显存 | 尖峰显存 | 是否可跑 | 备注 |
|------|-----------|-------|---------|---------|---------|---------|------|
| turbo4 256K pool | 262,144 | 5 | 52,480 | 22.3 GB | 22.5 GB | ✅ | 之前基线 |
| turbo4 384K pool | 393,216 | 5 | 78,848 | 26.1 GB | - | ✅ 启动 | 未跑完整 benchmark |
| turbo4 448K pool | 458,752 | 5 | 91,904 | 28.0 GB | - | ✅ 启动 | 未跑完整 benchmark |
| turbo4 512K pool | 524,288 | 5 | 104,960 | 29.8 GB | - | ✅ 启动 | 未跑完整 benchmark |
| turbo3 256K × 1 slot | 262,144 | 1 | 262,144 | 21.4 GB | - | ✅ | K 被自动升级为 q8_0 |
| turbo3 256K × 2 slots | 524,288 | 2 | 262,144 | 28.5 GB | - | ✅ | 每 slot 真 256K |
| turbo3 256K × 3 slots | 786,432 | 3 | 262,144 | - | - | ❌ OOM | cudaMalloc 失败 |
| turbo3 YaRN 288K × 2 slots | 576,716 | 2 | 288,512 | 29.9 GB | - | ✅ | 超原生 262K |
| turbo3 YaRN 314K × 2 slots | 629,145 | 2 | 314,624 | 31.3 GB | - | ✅ 启动 | 只剩 1.1 GB, 风险高 |
| turbo3 140K × 4 slots | 573,440 | 4 | 143,360 | 30.0 GB | 30.2 GB | ✅ | 4 agent stress 通过 |
| turbo3 150K × 4 slots | 614,400 | 4 | 153,600 | 31.1 GB | - | ✅ 启动 | 只剩 1.4 GB, 未压测 |
| turbo3 155K × 4 slots | 634,880 | 4 | 158,720 | 31.6 GB | - | ✅ 启动 | 只剩 0.8 GB, 极易 OOM |

**关键观察:**

1. **单 slot 真 256K 可行:** `-c 262144 -np 1 turbo3` 用 21.4 GB, 且 turbo3 在该模型上因 GQA 6:1 被自动升级为非对称 (K=q8_0, V=turbo3), 质量损失较小。
2. **双 slot 各 256K 是并发极限:** `-c 524288 -np 2 turbo3` 用 28.5 GB, 成功启动。三 slot 各 256K 直接 OOM。
3. **YaRN 可继续扩展单 slot 长度:** 在双 slot 基础上用 YaRN 1.1/1.2 把有效 context 推到 288K/314K, 但 314K 已只剩 1.1 GB, 实际高负载会爆。
4. **4 slot × 140K 是兼顾并发与上下文的甜点:** 启动 30.0 GB, 4 agent stress 尖峰 30.2 GB (仅比启动高 200 MB), 留有 ~2.5 GB 安全余量。
5. **显存尖峰真实存在但不大:** 在 4×140K stress 中, 启动 30.0 GB, 运行时最高 30.2 GB。尖峰主要来自 image encoder 临时 buffer 和 prefill batch, 不是无限增长。

> **建议:** 不要把启动显存推到 31 GB 以上。保留至少 1.5–2 GB 余量给运行时尖峰。

#### 17.6.6 速度优化: MTP 与 DFlash

用户提出在上下文/并发已满足的前提下优化速度, 因此测试了 **MTP (Multi-Token Prediction)** 和 **DFlash**。

**DFlash:**
- `models/Qwen3.6-27B-DFlash/dflash.gguf` 在当前 TurboQuant fork 上直接报错: `unknown model architecture: 'dflash'`。
- 结论: **TheTom TurboQuant fork 不支持 DFlash**。如需 DFlash, 必须换用支持 DFlash 的 llama.cpp 构建。

**MTP:**
- 需要 **带 MTP heads 的 GGUF**, 例如 `models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_XS.gguf`。
- 当前 Q3_K_M 模型开 `--spec-type draft-mtp` 会报错并 OOM (模型缺少 MTP heads)。
- IQ4_XS + MTP + 多模态 **可以共存**, 没有遇到上游 #22867 的崩溃。

**MTP 实测 (IQ4_XS, 1 slot × 65K, turbo4 KV):**

| 指标 | 值 |
|------|-----|
| 启动显存 | 18.6 GB |
| steady decode tok/s | **41.5** |
| total throughput | 27.8 tok/s |
| draft acceptance | 86.4% |
| 多模态 | 成功 |

对比 IQ4_XS 同配置 **不开 MTP** 的 36 tok/s, MTP 提升约 15%; 对比 Q3_K_M no-MTP 的 21 tok/s, 提升约 98%。

**但 MTP 与长上下文冲突:**
- IQ4_XS + MTP + 2×256K context **OOM** (MTP draft context 额外需要 2.4 GB)。
- 因此 **MTP 适合“短上下文 + 高速度”场景**, 长上下文场景必须关闭 MTP。

### 17.7 结论与推荐

| 场景 | 推荐引擎 | 理由 |
|------|---------|------|
| **低延迟、高吞吐、多模态 (短上下文)** | **llama.cpp TurboQuant + IQ4_XS + MTP** | steady decode 41.5 tok/s, 多模态稳定, 启动显存 18.6 GB |
| **低延迟、高吞吐、多模态 (中等上下文)** | **llama.cpp Q3_K_M no-spec** | 单/双 agent 速度领先, 延迟最稳定 |
| **需要 98K+ 上下文 + 多模态** | 1Cat-vLLM 1.2.1 | 唯一支持近 100K 上下文的 vLLM 配置 |
| **5+ 高并发 vLLM 负载** | 1Cat-vLLM 1.2.0 | 与 §15 结论一致, 高并发下 1.2.0 更稳定 |
| **V100 上 5 agents × 50K+ 上下文** | **llama.cpp TurboQuant (256K pool, 5 slots)** | 总吞吐 38.7 tok/s, 显存仍有 ~10 GB 余量 |
| **V100 上 4 agents × 140K 上下文** | **llama.cpp TurboQuant (384K pool, 4 slots)** | 兼顾并发与单 agent 长上下文,  stress 通过 |
| **V100 上 2 agents × 真 256K 上下文** | **llama.cpp TurboQuant (524K pool, 2 slots, turbo3)** | 单 slot 256K, 双并发, 21.4 GB 启动 |
| **V100 上单 agent 超长上下文 (256K+)** | **llama.cpp TurboQuant (262K pool, turbo3) + YaRN** | 可扩展到 288K–314K, 但需留显存余量 |
| **生产默认** | **llama.cpp DFlash (§14)** | 若可用, 仍优于 no-spec; 本测试的 no-spec 是 DFlash 不可用时的 fallback |

**重要提醒:**
- 本测试的 llama.cpp 是 **无 DFlash** 的保守配置。实际生产应优先使用 §14 的 `start_llama.sh`。
- TurboQuant fork 尚未合并到 llama.cpp 主线, 构建需要 `feature/turboquant-kv-cache` 分支 + SM70 兼容的 CUDA 12.x 工具链。
- TurboQuant 节省的是 KV cache, 不是模型权重。显存受限时优先选择 turbo4; 需要极限 context 时再考虑 turbo3 并承受速度损失。
- **MTP 和长上下文不可兼得**: 启用 MTP 后 VRAM 开销增加, 长 context / 多 slot 配置会 OOM。根据场景二选一。
- **不要把启动显存推到 31 GB 以上**: 高并发多模态会产生临时 buffer 尖峰, 建议保留 ≥1.5 GB 余量。

### 17.8 修复: `start_llama_nospec.sh` LD_LIBRARY_PATH

在运行本测试时, `start_llama_nospec.sh` 因 `set -u` 与未绑定的 `LD_LIBRARY_PATH` 变量导致启动失败:

```
./start_llama_nospec.sh: 行 13: LD_LIBRARY_PATH: 未绑定的变量
```

已修复为使用默认值:

```bash
export LD_LIBRARY_PATH="$VENV_LIBS/cuda_runtime/lib:$VENV_LIBS/cublas/lib:$VENV_LIBS/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

修复后 llama.cpp no-spec 可正常启动并完成全部 1/2/4 agent 测试。

---

## 18. 生产配置: 4×140K TurboQuant + 思考模式控制

### 18.1 选择理由

§17 的测试表明, TurboQuant turbo3 KV cache 在 V100-32GB 上可以支撑 **4 slot × 140K context** (总 KV pool 573,440 tokens), 启动显存仅 ~30.0 GB, 4-agent stress 尖峰 ~30.2 GB, 保留 ~2.5 GB 安全余量。

相比之前的方案:

| 配置 | 单 slot 上下文 | 并发 slot | 单 agent decode tok/s | 多模态 |
|------|--------------|----------|----------------------|--------|
| vLLM 1.2.0 (§16) | 51,744 (2 seq 共享) | 2 | 19.5 | ✅ |
| vLLM 1.2.1 (§15) | 98,784 (2 seq 共享) | 2 | 21.4 | ✅ |
| llama.cpp DFlash (§14) | 32,768 | 2 | 48–51 (短 prompt) | ✅ |
| llama.cpp no-spec q8_0 (§17) | 32,768 | 2 | 23.5 | ✅ |
| **TurboQuant 4×140K (本节)** | **143,360** | **4** | **5.5 (4-agent stress)** | ✅ |

**4×140K 的核心优势是 "4 个 agent 各自拥有 140K 上下文"** — 这是单 V100-32GB 上目前能达到的最大并发 × 上下文组合。代价是单 agent decode 速度较低 (5.5 tok/s), 因为 turbo3 KV cache 的 dequant 开销 + 长 context attention 的计算量。

> 如果需要更高单 agent 速度, 可切回 §14 的 DFlash 配置 (48 tok/s, 2 slot × 32K) 或 §17.6.6 的 MTP 配置 (41.5 tok/s, 1 slot × 65K)。

### 18.2 启动命令

文件: `./start_llama_turboquant_4x140k.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"

# 4 slots × 140K context = 573,440 total KV cache budget
export TURBO_CTX=573440
export TURBO_SLOTS=4
export TURBO_CACHE_K=turbo3
export TURBO_CACHE_V=turbo3

export TURBO_MODEL="${TURBO_MODEL:-./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf}"
PORT="${PORT:-8000}"

exec ./llama.cpp-turboquant/build/bin/llama-server \
  -m "$TURBO_MODEL" \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c "$TURBO_CTX" -ngl 99 -np "$TURBO_SLOTS" -fa on \
  --cache-type-k "$TURBO_CACHE_K" --cache-type-v "$TURBO_CACHE_V" \
  --cache-ram 0 \
  --jinja \
  --chat-template-file ./chat_templates/qwen3.6_merged.jinja \
  --reasoning off \
  --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port "$PORT" \
  "$@"
```

**关键参数说明:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `-c 573440` | 573,440 tokens | 总 KV pool = 4 × 143,360 |
| `-np 4` | 4 slots | 每 slot 可用 143,360 tokens (140K) |
| `--cache-type-k/v turbo3` | 3.25 bit/val | 压缩 KV cache, 牺牲少量速度换取大 context |
| `--cache-ram 0` | 禁用 prompt cache | 节省 VRAM, 因为长 context 配置下 prompt cache 收益有限 |
| `--reasoning off` | 默认关闭思考 | 防止 Qwen3.6 默认强制输出 `<think>` 推理过程 |
| `--chat-template-file` | merged jinja | 加载支持 `enable_thinking` 开关的合并模板 |
| `--mmproj` | BF16 | 多模态视觉投影, 支持图片输入 |

### 18.3 预期性能 (4-agent stress 实测)

**测试条件:** 4 agents, 每 agent 25 turns, 共享前缀 64,512 tokens (~45% of 143K), 每 5 轮插入图片, greedy + 256 tokens/轮, 1 warmup turn 丢弃。

**数据来源:** `benchmarks/turboquant/results/llama_tq3_140k_4slot_stress.json` (2026-07-02)

| 指标 | 值 |
|------|-----|
| **稳态总吞吐** | **17.24 tok/s** (4 agents 合计) |
| **稳态单 agent decode** | avg **5.52** tok/s (p50=5.49, p95=5.80, min=5.20) |
| 稳态单请求延迟 | avg 46.4 s (p50=46.6, p95=48.3) |
| 稳态总生成 token | 24,576 |
| 累计 prompt token | 4,653,620 |
| cache hit rate | ~99% |
| compaction 次数 | 0 |
| 多模态轮数 | 20 |
| 启动显存 | ~30.0 GB |
| stress 尖峰显存 | ~30.2 GB |

> **注意:** TTFT 显示 0.0 是因为 llama.cpp 的 OpenAI 端点在 `--reasoning off` 之前会将推理内容放入 `reasoning_content` 而 `content` 为空, 导致 benchmark 的 TTFT 计时器 (基于 content 首 token) 误判。使用 `--reasoning off` + merged template 后此问题已修复。

**单 agent 场景的预期速度:** 当只有 1 个 agent 活跃时 (其余 3 个 slot 空闲), decode 速度预计与 §17.6.3 的 turbo3 longctx 单 agent 接近 (~16 tok/s), 因为没有 4-way 并发竞争。4-agent stress 的 5.5 tok/s 是满负载下的下限。

### 18.4 思考模式 (Thinking) 控制

Qwen3.6 默认在每次回复前生成 `<think>` 推理过程。在生产 agent 场景中, 这会:

1. 浪费 ~200–500 tokens 的生成预算在内部推理上
2. 导致 OpenAI 兼容客户端的 `content` 字段为空 (推理被放入 `reasoning_content`)
3. 增加每轮延迟

**解决方案:**

1. **服务器默认:** `--reasoning off` — 所有请求默认不生成思考过程, 直接输出答案到 `content`。
2. **客户端按需开启:** 通过 `chat_template_kwargs` 在单次请求中覆盖:

```json
{
  "model": "qwen3.6-27b-awq",
  "messages": [{"role": "user", "content": "证明勾股定理"}],
  "chat_template_kwargs": {"enable_thinking": true}
}
```

启用时, 推理内容会出现在 `reasoning_content` 字段, 最终答案在 `content` 字段。

**模板文件:** `chat_templates/qwen3.6_merged.jinja` — 来自 fakezeta/allanchan339/froggeric 合并的 Qwen3.6 chat template, 支持 `enable_thinking` / `think_off` / `think_on` 三种控制方式。

**验证结果:**

| 请求方式 | `content` | `reasoning_content` |
|---------|----------|-------------------|
| 默认 (服务器 `--reasoning off`) | ✅ 正常答案 | 空 |
| `chat_template_kwargs.enable_thinking=true` | 空 | ✅ 推理过程 |
| `chat_template_kwargs.enable_thinking=false` | ✅ 正常答案 | 空 |
| 多模态 + 默认 | ✅ 正常答案 (含图片描述) | 空 |

### 18.5 显存安全余量

| 配置 | 启动显存 | 余量 (32GB - 占用) | 风险等级 |
|------|---------|-------------------|---------|
| turbo3 4×140K (生产) | 30.0 GB | **~2.5 GB** | ✅ 安全 |
| turbo3 4×150K | 31.1 GB | ~1.4 GB | ⚠️ 紧张 |
| turbo3 4×155K | 31.6 GB | ~0.8 GB | ❌ 极易 OOM |

> **建议: 不要把启动显存推到 31 GB 以上。** 高并发多模态会产生临时 buffer 尖峰 (image encoder + prefill batch), 保留 ≥1.5 GB 余量。4×140K 的 2.5 GB 余量在 4-agent stress + 20 次多模态轮次下验证安全 (尖峰仅 +0.2 GB)。

### 18.6 生产部署状态

| 项目 | 值 |
|------|-----|
| tmux session | `llama-tq-4x140k` |
| 端口 | 8000 |
| 模型别名 | `qwen3.6-27b-awq` |
| 引擎 | `llama.cpp-turboquant/build/bin/llama-server` |
| 模型 | `Qwen3.6-27B-Q3_K_M.gguf` (13.6 GB) |
| 多模态 | ✅ (`mmproj-BF16.gguf`) |
| 思考模式 | 默认关闭, 客户端可按需开启 |
| VRAM | ~30.0 GB / 32 GB |
| 日志 | `/tmp/llama-tq-4x140k.log` |

---
