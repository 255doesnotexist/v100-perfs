# V100-32GB LLM 推理性能实测

在单块 Tesla V100-PCIE-32GB (SM70) 上对 **Qwen3.6-27B** 混合注意力模型进行的多引擎性能基准测试。

覆盖 vLLM (1Cat-vLLM 1.2.0/1.2.1)、llama.cpp (上游 + TurboQuant fork)、推测解码 (DFlash/MTP)、多种量化级别和 KV cache 压缩方案。

## 硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA Tesla V100-PCIE-32GB (compute capability 7.0 / SM70) |
| 驱动 | 580.159.04 |
| VRAM | 32 GB HBM2 |
| 系统 | Fedora Linux 44, kernel 6.19.14 |

## 模型

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-27B (z-lab) |
| 架构 | 64 层 = 16 full_attention + 48 linear_attention/GDN (混合注意力) |
| 原生上下文 | 262,144 tokens |
| MTP 层 | 内置 1 层 (`mtp_num_hidden_layers: 1`) |

测试过的权重格式:

| 格式 | 文件 | 大小 |
|------|------|------|
| AWQ 4-bit | `Qwen3.6-27B-AWQ` | 21 GB |
| GGUF Q3_K_M | `Qwen3.6-27B-Q3_K_M.gguf` | 13.6 GB |
| GGUF Q6_K | `Qwen3.6-27B-Q6_K.gguf` | 22.5 GB |
| GGUF IQ4_XS (含 MTP) | `Qwen3.6-27B-IQ4_XS.gguf` | 12.2 GB |
| GGUF IQ2_M | `Qwen3.6-27B-IQ2_M.gguf` | 10.9 GB |
| DFlash draft | `dflash.gguf` | 3.46 GB |
| mmproj (视觉) | `mmproj-BF16.gguf` | 0.93 GB |

## TL;DR 结论

| 场景 | 最佳配置 | 单 agent tok/s | 上下文 | 多模态 |
|------|---------|---------------|--------|--------|
| **速度优先 (短上下文)** | llama.cpp TurboQuant + IQ4_XS + MTP | **41.5** | 65K | yes |
| **平衡 (中等上下文)** | llama.cpp Q3_K_M + DFlash + q8_0 KV | **48–51** | 65K | yes |
| **4 agent × 140K 上下文 (生产)** | llama.cpp TurboQuant turbo3, 4 slots | **~30** (1-2 活跃) / 5.5 (4 并发) | 4×140K | yes |
| **vLLM 兼容** | 1Cat-vLLM 1.2.1 AWQ + fp8 KV | **21.4** | 98K | yes |
| **vLLM 高并发** | 1Cat-vLLM 1.2.0 AWQ | **60.5** (5 并发合计) | 20K | no |

> 4×140K 配置在真实 agent 负载中, agent 通常是轮流工作而非同时生成。1-2 个 agent 活跃时单 agent 约 30 tok/s; 只有 4 个 agent 同时在 140K context 上并发解码时才会降到 5.5 tok/s (attention 计算量随 context 平方增长)。这个配置的核心价值是 "4 个 agent 各自拥有 140K 上下文", 这是单 V100-32GB 上能达到的最大并发 × 上下文组合。

### V100 上的核心限制

**AWQ 4-bit 权重在 SM70 上实际占用 ~27 GB** (而非理论 4-bit 的 ~14 GB), 因为 V100 没有原生 4-bit 计算, 权重以半解包形式驻留 VRAM。这导致:

- AWQ + fp8 KV: 仅剩 ~3 GB 给 KV cache, 单序列上限 ~98K
- AWQ + DFlash draft (3.5 GB): 直接 OOM
- AWQ + MTP draft (~2 GB): KV cache 仅剩 1 GB, 上下文降到 ~3K

**GGUF 路线绕过了这个问题**: Q3_K_M 权重仅 ~14 GB, 给 KV cache 和 draft model 留出 ~16 GB。llama.cpp + GGUF 是 V100 上的最优路径。

## 目录结构

```
.
├── README.md                           # 本文件
├── thinking_proxy.py                   # Thinking Proxy: API 翻译 + 认证 + 进程管理
├── docs/
│   ├── performance.md                  # 完整性能报告 (18 个章节, 含所有实测数据)
│   └── EXPERIENCE.md                   # 经验总结: SM70 踩坑、构建、调优
├── scripts/
│   ├── start_proxy.sh                  # Thinking Proxy 启动 (管理 llama-server)
│   ├── start.sh                        # vLLM 1.2.0 生产 (51K ctx, multimodal)
│   ├── start_120.sh                    # vLLM 1.2.0 纯文本 (20K ctx)
│   ├── start_121.sh                    # vLLM 1.2.1 (98K ctx, fp8 KV)
│   ├── start_llama.sh                  # llama.cpp + DFlash 生产 (65K, multimodal)
│   ├── start_llama_nospec.sh           # llama.cpp 无推测 (65K, multimodal)
│   ├── start_llama_turboquant.sh       # TurboQuant 通用启动器 (env 可配置)
│   └── start_llama_turboquant_4x140k.sh # TurboQuant 4×140K 生产配置
├── benchmarks/
│   ├── agent_bench.py                  # 多 agent 基准测试 harness
│   ├── run_agent_bench.sh             # 串行多引擎测试 runner
│   ├── vllm-120/                       # 1.2.0 结果 (1/2/4 agents)
│   ├── vllm-121/                       # 1.2.1 结果 (1/2/4 agents)
│   ├── llama-nospec/                   # llama.cpp no-spec 结果
│   └── turboquant/
│       ├── results/                    # 所有 TurboQuant JSON 结果
│       └── logs/                       # 所有 TurboQuant 服务器日志
└── chat_templates/
    └── qwen3.6_merged.jinja            # 合并的 Qwen3.6 chat template (支持 thinking 开关)
```

## 如何复现

### 前提条件

1. **V100-32GB** (或其他 SM70 GPU), 驱动 >= 550
2. **CUDA 12.8/12.9 工具链** (系统 CUDA 13.x 不支持 SM70)
3. 模型权重 (按上方表格自行下载)
4. llama.cpp 编译 (见 `docs/EXPERIENCE.md` 的构建章节)

### 运行基准测试

```bash
# 1. 启动要测试的引擎, 例如 TurboQuant 4×140K
./scripts/start_llama_turboquant_4x140k.sh &

# 2. 等待 health
curl http://127.0.0.1:8000/health

# 3. 运行多 agent 基准测试
python benchmarks/agent_bench.py \
  --endpoint http://127.0.0.1:8000 \
  --model qwen3.6-27b-awq \
  --num-agents 4 \
  --max-model-len 143360 \
  --shared-prefix-tokens 64512 \
  --turns 25 \
  --max-tokens 256 \
  --compact-ratio 0.6 \
  --warmup-turns 1 \
  --output results.json
```

### 运行完整串行对比

```bash
# 依次启动 vLLM 1.2.0, 1.2.1, llama.cpp no-spec 并测试 1/2/4 agents
./benchmarks/run_agent_bench.sh /tmp/results
```

> **重要:** 必须串行测试, 每次只跑一个引擎。并发测试会导致 GPU 资源争抢, 数据失真。

## 指标说明

| 指标 | 定义 |
|------|------|
| **decode tok/s** | `completion_tokens / (last_token_time - first_token_time)`, 仅 token 生成阶段 |
| **e2e tok/s** | `completion_tokens / total_latency`, 含 TTFT + 网络开销 |
| **total throughput** | `total_completion_tokens / wall_time`, 所有 agent 合计 |

> `decode tok/s` 不应与浏览器端 OpenAI 流吞吐直接比较。V100 首次请求有 JIT warmup, 因此每个 agent 的第 1 轮被丢弃, 仅统计 steady-state。

## License

测试数据和脚本供社区参考, 按 MIT 许可发布。模型权重版权归各自所有者。
