# V100-32GB (SM70) LLM 推理经验总结

> 本文记录了在 Tesla V100-PCIE-32GB 上运行 Qwen3.6-27B (混合注意力架构) 过程中遇到的所有坑、解决方案和调优经验。适用于任何在 SM70 GPU 上部署大模型的场景。

---

## 1. SM70/V100 的根本限制

### 1.1 没有原生 4-bit 计算

V100 (SM70) 硬件不支持 4-bit 整数运算。AWQ/GPTQ 等 4-bit 量化权重在推理时必须 **半解包**到 16-bit 才能参与矩阵乘法, 这意味着:

- 理论上 27B × 4-bit = ~14 GB 权重
- V100 上实际占用 **~27 GB** (几乎翻倍)
- 剩余给 KV cache 的空间极小 (~3 GB)

**这是 V100 上一切显存问题的根源。** A100/H100 有原生 4-bit 支持, 不会有这个问题。

### 1.2 CUDA 13.x 不支持 SM70

NVIDIA CUDA Toolkit 13.0+ 移除了对 compute capability 7.0 的支持。必须使用 **CUDA 12.8 或 12.9** 工具链。

```bash
# 检查 nvcc 支持的架构
nvcc --list-gpu-arch
# 如果看不到 sm_70, 说明 CUDA 版本太新

# 解决方案: 用 conda 安装 CUDA 12.8
conda install -c nvidia cuda-toolkit=12.8
# 或 12.9 (turboquant fork 实测可用)
conda install -c nvidia cuda-toolkit=12.9
```

### 1.3 没有 Flash Attention 2/3 原生支持

V100 支持 FA1 但不支持 FA2/FA3 的某些优化路径。1Cat-vLLM 提供了 `FLASH_ATTN_V100` 后端作为替代, 但功能受限 (例如 sliding window 支持不完整)。

---

## 2. 引擎选择决策树

```
你的首要需求是什么?
│
├─ 单 agent 速度最快
│   └─ llama.cpp + IQ4_XS GGUF + MTP (41.5 tok/s, 65K ctx)
│       └─ 需要带 MTP heads 的 GGUF
│       └─ MTP 与长上下文互斥 (VRAM 不够)
│
├─ 平衡速度 + 上下文 + 多模态
│   └─ llama.cpp + Q3_K_M + DFlash + q8_0 KV (48-51 tok/s, 65K ctx)
│       └─ DFlash 需要 llama.cpp 主线 (TurboQuant fork 不支持)
│
├─ 多 agent × 长上下文
│   └─ llama.cpp TurboQuant turbo3, 4 slots × 140K
│       └─ 单 agent 满负载时仅 5.5 tok/s (attention 平方复杂度)
│       └─ 但 4 个 agent 各自独享 140K, 这是 V100 极限
│
├─ 必须 vLLM (工具调用/兼容性)
│   ├─ 1-2 并发 → 1.2.1 (单请求快 3.8×)
│   └─ 5+ 并发 → 1.2.0 (高并发更稳定)
│
└─ 需要最大单序列上下文
    └─ TurboQuant turbo3, 1 slot × 256K + YaRN
        └─ 可扩展到 314K, 但只剩 ~1 GB VRAM 余量
```

---

## 3. KV Cache 量化: 最关键的调优杠杆

### 3.1 为什么 KV cache 量化如此重要

Qwen3.6-27B 有 16 层 full_attention (有 KV cache) + 48 层 GDN (递归状态, 不占 token 级 KV)。

- fp16 KV: 每 token 64 KB → 65K context 需要 4 GB
- q8_0 KV: 每 token 32 KB → 65K context 需要 2 GB
- turbo4 KV: ~4.25 bit/val → 65K context 需要 ~1.1 GB
- turbo3 KV: ~3.25 bit/val → 65K context 需要 ~0.85 GB

在 V100 32GB 上, 权重固定占 14-27 GB, KV cache 的每一 GB 都很珍贵。

### 3.2 量化级别选择

| KV cache 类型 | 速度损失 | VRAM 节省 | 质量影响 | 建议 |
|--------------|---------|----------|---------|------|
| fp16 | 基线 | — | 无 | 仅短上下文可用 |
| q8_0 | ~10% | 50% | 极小 | llama.cpp 生产推荐 |
| turbo4 | ~10% | ~70% | 小 | TurboQuant 默认 |
| turbo3 | ~30% | ~80% | 可感知 | 需要极限上下文时 |

**经验法则:** 先用 turbo4/q8_0, 只有需要 140K+ 单 slot 上下文时才降到 turbo3。

### 3.3 turbo3 在 GQA 模型上的自动升级

Qwen3.6 使用 GQA (grouped query attention), KV head 数与 attention head 数比例为 6:1。TurboQuant fork 在检测到 GQA 时, 会自动将 K cache 升级为 q8_0 (因为 K 的精度对 attention score 影响更大), 只对 V 使用 turbo3。这是一个合理的设计, 不需要手动干预。

---

## 4. 推测解码 (Speculative Decoding) 在 V100 上的现实

### 4.1 AWQ + 任何 draft model = OOM

27B-AWQ 在 V100 上占 27 GB, 剩余 3 GB 不够任何 draft model:
- MTP draft: ~2 GB → KV cache 仅剩 1 GB, 上下文降到 ~3K
- DFlash draft: 3.5 GB → 直接 OOM

**结论: AWQ + speculative decoding 在单 V100-32GB 上不可行。**

### 4.2 GGUF Q3_K_M + DFlash: 可行且有效

Q3_K_M 仅 14 GB, 给 DFlash 留出足够空间:

```
权重 14 GB + DFlash 3.5 GB + CUDA/激活 4 GB = 21.5 GB
剩余 ~10 GB 给 KV cache → 65K context (q8_0 KV)
```

实测加速 ~5% (短 prompt) 到 2.7× (100 token prompt)。DFlash 的优势在短上下文场景最明显。

### 4.3 GGUF IQ4_XS + MTP: 最快但上下文受限

IQ4_XS (12.2 GB) + MTP draft, 单 slot 65K context:
- steady decode: **41.5 tok/s** (acceptance 86.4%)
- 但 MTP + 2×256K context 会 OOM

**MTP 和长上下文不可兼得**, 根据场景二选一。

### 4.4 TurboQuant fork 不支持 DFlash

TheTom 的 TurboQuant fork (`feature/turboquant-kv-cache` 分支) 没有 DFlash 支持 (`unknown model architecture: 'dflash'`)。如果需要 DFlash, 必须用 llama.cpp 主线; 如果需要 TurboQuant KV 压缩, 必须用 fork。两者不可兼得。

---

## 5. 构建指南

### 5.1 llama.cpp 主线 (SM70)

```bash
# 需要 CUDA 12.8 (CUDA 13 不支持 SM70)
conda activate tsenv  # 或其他有 CUDA 12.8 的环境

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --config Release -j$(nproc)
# 二进制: build/bin/llama-server
```

### 5.2 TurboQuant fork (SM70)

```bash
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant

# 需要兼容的 gcc 和 nvcc
# 实测组合: gcc 14.3 + nvcc 12.9 (conda tsenv)
conda install -c nvidia cuda-toolkit=12.9
conda install -c conda-forge gcc=14.3 gxx=14.3

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_C_COMPILER=gcc-14.3 \
  -DCMAKE_CXX_COMPILER=g++-14.3
cmake --build build --config Release -j$(nproc)
```

### 5.3 运行时库依赖

TurboQuant 二进制链接的是 conda tsenv 的 CUDA 12.9 运行时库, 必须在启动脚本中设置:

```bash
export LD_LIBRARY_PATH="/home/$USER/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"
```

如果忘了设置, 会报 `libcudart.so.12: cannot open shared object file`。

### 5.4 GGUF 加载器在 vLLM/transformers 中不可用

1Cat-vLLM 内置了 GGUF 加载器, 但 transformers 的 `load_gguf_checkpoint()` 不支持 `qwen35` 架构。因此 **不能** 通过 vLLM 加载 GGUF 格式的 Qwen3.6, 只能用 llama.cpp。

---

## 6. 1Cat-vLLM 1.2.0 vs 1.2.1 的并发回归

### 6.1 问题描述

GitHub issues 报告 1.2.1 在高并发下性能下降:
- #78: 1.2.1 双并发 ~43 tok/s, 1.2.0 可达 90+ tok/s
- #89: 3-6 并发时降到 ~1 tok/s
- #90: 5 并发 prefix cache hit rate 仅 21.3%

### 6.2 实测确认

| 版本 | 单请求 | 2 并发 | 5 并发 |
|------|--------|--------|--------|
| 1.2.0 | 4.8 tok/s | 24.1 tok/s | **60.5 tok/s** |
| 1.2.1 | **18.1 tok/s** | **40.1 tok/s** | 34.2 tok/s |

1.2.1 单请求快 3.8×, 但 5 并发时 1.2.0 反超 77%。P95 延迟在 1.2.1 的 4-agent 测试中飙到 166 秒。

### 6.3 选择建议

- **1-2 并发**: 用 1.2.1
- **5+ 高并发**: 用 1.2.0 (但单请求慢, 且无 fp8 KV)
- **最佳方案**: 换 llama.cpp, 两者都优于 vLLM

---

## 7. 1.2.0 多模态修复

### 7.1 缺失的 CUDA 扩展

1.2.0 wheel 缺少 `vllm.vllm_flash_attn` 的 CUDA 扩展 `_vllm_fa2_C.abi3.so`, 加载视觉编码器时报:

```
ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions
```

**修复:** 从 1.2.1 venv 复制:

```bash
cp .venv-1cat/lib64/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
   .venv-1cat-120/lib64/python3.12/site-packages/vllm/vllm_flash_attn/
```

复制后多模态正常启动, 视觉编码器 fallback 到 `TORCH_SDPA`。

### 7.2 关闭默认 MTP

1.2.0 自动为 SM70 启用 MTP4:

```
Applied 1Cat SM70 MTP defaults: speculative_config=mtp4, ...
```

MTP4 吃掉 ~2+ GiB, 导致 KV cache 极小。关闭:

```bash
export VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1
```

### 7.3 fp8 KV cache 不生效

1.2.0 的 `scaled_fp8_quant` CUDA kernel 未实现 `Float8_e5m2`/`Float8_e4m3fn`。1.2.1 日志显示 `kv_cache_dtype=auto` 即使传了 `--kv-cache-dtype fp8_e5m2` 也没生效。

---

## 8. Qwen3.6 思考模式 (Thinking) 控制

### 8.1 问题

Qwen3.6 默认在每次回复前生成 `<think>` 推理过程。在 OpenAI 兼容 API 中:
- 推理内容进入 `reasoning_content` 字段
- `content` 字段为空

大多数 OpenAI 客户端只读 `content`, 导致看起来 "模型没有回复"。

### 8.2 解决方案

**方法 1 (推荐):** llama-server 启动时加 `--reasoning off`:

```bash
./llama-server ... --reasoning off --jinja
```

客户端按需开启:
```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

**方法 2:** 加载支持 thinking 开关的 chat template:

```bash
--chat-template-file ./chat_templates/qwen3.6_merged.jinja
```

我们使用的 merged template 来自 fakezeta/allanchan339/froggeric 的合并版本, 支持 `enable_thinking` / `think_off` / `think_on` 三种控制方式。

**注意:** `--chat-template-kwargs '{"enable_thinking":false}'` 已被废弃, 推荐用 `--reasoning off`。如果必须传 JSON, 注意 shell 转义 — 用单引号包裹, 不要用 `\\\"`。

---

## 9. 显存极限探索

### 9.1 安全余量

| 配置 | 启动 VRAM | 余量 | 安全性 |
|------|----------|------|--------|
| turbo3 4×140K | 30.0 GB | 2.5 GB | 安全 |
| turbo3 4×150K | 31.1 GB | 1.4 GB | 紧张 |
| turbo3 4×155K | 31.6 GB | 0.8 GB | 极易 OOM |

**经验法则: 启动 VRAM 不要超过 31 GB, 保留 ≥1.5 GB 给运行时尖峰。**

### 9.2 运行时尖峰来源

- Image encoder 临时 buffer (多模态)
- Prefill batch 内存分配
- CUDA malloc 碎片

在 4×140K 的 4-agent stress 测试中 (含 20 次多模态), 尖峰仅 +0.2 GB (30.0 → 30.2)。但更大 context 或更多图片时尖峰会增加。

### 9.3 nvidia-smi 显示的 "Free" 不可全用

nvidia-smi 看到的 3 GB "Free" 包含 PyTorch 预留但未分配内存 + CUDA 驱动保留。vLLM 的 `--gpu-memory-utilization` 不会超分配。**不要用 nvidia-smi Free 来计算可用 KV cache, 要看引擎日志报告的实际 KV cache 大小。**

---

## 10. 基准测试方法论

### 10.1 必须串行测试

同时跑两个引擎会导致 GPU 资源争抢, 数据完全失真。每次只启动一个引擎, 测完再换。

### 10.2 Warmup 必须丢弃

V100 首次请求有 JIT 编译 / CUDA kernel 缓存开销。每个 agent 的第 1 轮结果必须丢弃, 只统计 steady-state。

### 10.3 decode tok/s vs e2e tok/s

- `decode tok/s`: 仅 token 生成阶段, 严格解码速度
- `e2e tok/s`: 含 TTFT + 网络, 浏览器/客户端实际体验

两者不能混用。发布基线时必须注明使用了哪个指标。

### 10.4 Context 长度对 decode 速度的影响巨大

| Context 长度 | decode tok/s (turbo3, 4 agents) |
|-------------|-------------------------------|
| 32K/slot | ~10 |
| 65K/slot | ~8 |
| 140K/slot | ~5.5 |

Attention 的计算量随 context 平方增长。**不能把短上下文的 tok/s 外推到长上下文场景。**

### 10.5 发布基线的检查清单

每次发布基线数据时, 必须包含:

- [ ] 完整启动命令
- [ ] GPU 型号
- [ ] 驱动版本
- [ ] CUDA 运行时版本
- [ ] 模型检查点 (量化级别/文件名)
- [ ] 采样参数 (temperature, max_tokens)
- [ ] Prompt 长度
- [ ] Decode 长度
- [ ] 使用的吞吐指标定义 (decode vs e2e)

---

## 11. 其他引擎测试结论

| 引擎 | 结论 |
|------|------|
| **ik_llama.cpp** | Q3_K_M 上与 llama.cpp 速度相同 (~26 tok/s), 无提升。CPU+GPU 混合推理极慢 (5 tok/s)。 |
| **exllamav3** | 依赖冲突 (pydantic v1 vs v2); Q4 量化在 layer 0 后静默崩溃。不可用。 |
| **exllamav2** | 不支持 `Qwen3_5ForConditionalGeneration` 架构, 加载失败。 |
| **GGUF via vLLM** | transformers `load_gguf_checkpoint()` 不支持 `qwen35` 架构。不可用。 |
| **DFlash via TurboQuant fork** | TurboQuant fork 不识别 `dflash` 架构。需用 llama.cpp 主线。 |

---

## 12. 工具和脚本说明

### 12.1 agent_bench.py

多 agent 基准测试 harness, 模拟真实 agent 工作负载:
- N 个 agent 并发对话, 各自维护独立上下文
- 共享前缀 (模拟 system prompt / RAG context)
- 上下文增长 + 60% 阈值 compaction
- 每 5 轮插入图片 (多模态)
- warmup 分离 (前 N 轮丢弃)
- 输出 JSON 含 steady-state 分区统计

### 12.2 start_llama_turboquant.sh

通用 TurboQuant 启动器, 支持环境变量覆盖:

```bash
# 基本用法
./start_llama_turboquant.sh

# 自定义配置
TURBO_CTX=262144 TURBO_SLOTS=5 \
TURBO_CACHE_K=turbo4 TURBO_CACHE_V=turbo4 \
./start_llama_turboquant.sh

# 启用 YaRN 扩展上下文
USE_YARN=1 YARN_SCALE=2 ./start_llama_turboquant.sh

# 启用 MTP (需要 IQ4_XS MTP GGUF)
USE_MTP=1 MTP_DRAFT_N_MAX=2 \
TURBO_MODEL=./models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_XS.gguf \
./start_llama_turboquant.sh
```

### 12.3 chat_templates/qwen3.6_merged.jinja

合并的 Qwen3.6 chat template, 支持三种 thinking 控制方式:
- `enable_thinking` (true/false) — 通过 chat_template_kwargs 传入
- `<|think_off|>` / `<|think_on|>` — 通过 system message 控制
- `--reasoning off/on` — 通过 llama-server CLI 控制

---

## 13. Thinking Proxy: 让标准 API 客户端控制思考模式

### 13.1 问题

llama-server 的 `enable_thinking` 只能通过 `chat_template_kwargs` 控制, 这是 llama.cpp 专有扩展。标准 OpenAI 和 Anthropic SDK 不会发送这个字段。

具体来说, llama-server 的请求解析 (`server-common.cpp:1069-1096`) 只查找:
- `reasoning_format` (请求体, 支持逐请求覆盖)
- `chat_template_kwargs.enable_thinking` (请求体, 合并 CLI 默认值后覆盖)

它完全不识别:
- OpenAI 的 `reasoning_effort` ("low"/"medium"/"high")
- OpenAI 的 `reasoning` 对象
- Anthropic 的 `thinking` (`{"type":"enabled","budget_tokens":N}`)
- Cherry Studio 的 `enable_thinking` (顶层字段, 非标准)

而且 Anthropic SDK 使用 `/v1/messages` 端点, 请求/响应/SSE 格式与 OpenAI 完全不同。llama-server 没有这个端点。

### 13.2 解决方案: thinking_proxy.py

单文件 FastAPI 代理, 运行在 llama-server 前面。架构:

```
Client (OpenAI SDK / Cherry Studio)  ─┐
                                      ├──► Proxy (port 8000)
Client (Anthropic SDK)                ─┘    ├─ reasoning_effort / enable_thinking → chat_template_kwargs
                                             ├─ Anthropic /v1/messages 格式转换
                                             ├─ IP 认证 (Tailscale/localhost 绕过)
                                             └─ llama-server 进程管理 (崩溃自动重启)
                                                      │
                                                      ▼
                                             llama-server (port 8001)
```

### 13.3 客户端兼容性 (踩过的坑)

不同客户端发送 thinking 控制参数的方式完全不同。代理必须全部拦截:

| 客户端 | 参数名 | 位置 | 格式 |
|--------|--------|------|------|
| **OpenAI SDK** (o1/o3 模式) | `reasoning_effort` | 请求体顶层 | `"high"` / `"low"` / `"none"` |
| **OpenAI SDK** (替代格式) | `reasoning` | 请求体顶层 | `{"effort":"high"}` 或 `"on"` |
| **Cherry Studio** | `enable_thinking` | 请求体顶层 | `true` / `false` |
| **Cherry Studio** | `thinking_budget` | 请求体顶层 | 数字 (token 预算) |
| **Anthropic SDK** | `thinking` | 请求体顶层 | `{"type":"enabled","budget_tokens":N}` |
| **llama.cpp 原生** | `chat_template_kwargs.enable_thinking` | 嵌套对象 | `true` / `false` |

代理的策略: 从请求体中 pop 出所有已知变体, 统一映射到 `chat_template_kwargs.enable_thinking`, 然后转发给 llama-server。

**关键教训:** Cherry Studio 是最隐蔽的坑。它不发 `reasoning_effort`, 而是在请求体顶层放 `enable_thinking: true` + `thinking_budget: 65536`。llama-server 完全忽略这两个字段 (不在它的解析范围内), 导致思考模式始终关闭。只有通过代理日志 dump 完整请求体才发现这个问题。

### 13.4 Anthropic 格式转换

参考 vLLM 的 `vllm/entrypoints/anthropic/serving.py` 实现, 主要转换点:

**请求 (Anthropic → OpenAI):**
- 顶层 `system` → 第一条 `{"role":"system"}` 消息
- `content` 数组中的 `text` / `image` / `thinking` / `tool_use` / `tool_result` 块 → OpenAI 对应格式
- `thinking.type == "enabled"` → `chat_template_kwargs.enable_thinking = true`
- `max_tokens` / `temperature` / `top_p` / `top_k` / `stop_sequences` → 直接映射
- `tools` 数组 (Anthropic `input_schema`) → OpenAI `function.parameters`

**响应 (OpenAI → Anthropic):**
- `choices[0].message.reasoning_content` → `content[].type = "thinking"` (附随机 `signature`)
- `choices[0].message.content` → `content[].type = "text"`
- `choices[0].message.tool_calls` → `content[].type = "tool_use"`
- `finish_reason` 映射: `stop → end_turn`, `length → max_tokens`, `tool_calls → tool_use`
- `usage.prompt_tokens → input_tokens`, `usage.completion_tokens → output_tokens`

**流式 SSE (OpenAI → Anthropic):**
- OpenAI 的 `data: {"choices":[{"delta":{"reasoning_content":"..."}}]}` → Anthropic 的 `event: content_block_delta` + `{"delta":{"type":"thinking_delta","thinking":"..."}}`
- OpenAI 的 `data: {"choices":[{"delta":{"content":"..."}}]}` → Anthropic 的 `event: content_block_delta` + `{"delta":{"type":"text_delta","text":"..."}}`
- 需要跟踪当前 block 类型 (thinking/text), 在切换时发出 `content_block_stop` + `content_block_start`
- 生命周期: `message_start` → `content_block_start` → `content_block_delta` × N → `content_block_stop` → `message_delta` → `message_stop`

### 13.5 认证设计

代理同时接受 OpenAI 和 Anthropic 的认证方式:

| 来源 | 认证方式 | 绕过条件 |
|------|---------|---------|
| Tailscale (100.64.0.0/10) | 无需认证 | IP 在 Tailscale 网段 |
| Localhost | 无需认证 | IP 是 loopback |
| FRP / 公网 | `Authorization: Bearer <token>` | — |
| FRP / 公网 | `x-api-key: <token>` | — |

默认 token: `AUTH_TOKEN` 环境变量 (建议通过环境变量设置, 不要硬编码)。

### 13.6 进程管理

代理内部管理 llama-server 子进程:
1. 启动时 spawn llama-server, 等待 `/health` 返回 200
2. 后台 monitor task 每 15 秒健康检查
3. 进程退出或健康检查失败 → kill + 重启, 指数退避 (5s → 10s → 20s → ... → 60s)
4. 代理关闭时 terminate llama-server

**模型加载时间:** 4x140K TurboQuant 配置从启动到 health ready 约 60-90 秒。重启后客户端会经历这段时间的 503。如果需要更快的恢复, 可以考虑:
- 降低 context 配置 (如 4x65K, 加载更快)
- 预加载模型到系统 page cache
- 使用 systemd 的 watchdog 机制替代代理内部监控

### 13.7 部署

```bash
# 在 tmux 中启动 (推荐)
tmux new-session -d -s proxy
tmux send-keys -t proxy "cd /path/to/project && ./start_proxy.sh" Enter

# 或直接运行
AUTH_TOKEN=your_token ./start_proxy.sh

# 客户端连接
# OpenAI: base_url=http://<host>:8000/v1, api_key=CHANGE_ME
# Anthropic: base_url=http://<host>:8000, api_key=CHANGE_ME
```

所有日志 (代理 + llama-server) 输出到 tmux 同一窗口, 可以实时观察。
