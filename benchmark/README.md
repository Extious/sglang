# Benchmark 目录总览

本目录收录了 SGLang 的各类 benchmark：任务/数据集评测、程序化推理/Agent 工作负载、受约束解码、Serving 吞吐与缓存、以及 kernel/通信 microbench。多数脚本默认连接本机 `http://127.0.0.1:30000` 的服务端（SGLang OpenAI-compatible 或 `/generate`）。

## 通用运行方式

### 1) 启动 SGLang Server（示例）

```bash
python3 -m sglang.launch_server --model-path <hf_or_local_model> --port 30000
```

### 2) 运行 benchmark 脚本（常见参数）

许多脚本使用 `python/sglang/test/test_utils.py` 里的公共参数（`add_common_sglang_args_and_parse`）：

- `--backend`：默认 `srt`（SGLang RuntimeEndpoint）；也可能使用 `gpt-*`（OpenAI）等
- `--host`：默认 `http://127.0.0.1`
- `--port`：默认 `30000`
- `--parallel`：默认 `64`（脚本内的并行线程/并发，用于 `run_batch` 等）
- `--result-file`：默认 `result.jsonl`（多数脚本会追加写入一行 JSONL 结果）

不少目录同时提供 `bench_other.py`，用于对比 vLLM / guidance / outlines / lmql / lightllm 等后端（各目录 README 中通常给了启动与运行命令）。

## 目录索引（按主题）

### A) 任务/数据集评测（Accuracy/Latency/Throughput）

#### `boolq/`
- 内容：BoolQ 二分类（few-shot），输出 `Accuracy / Latency / Output throughput(token/s)`。
- 运行：下载 HF 数据集并将 parquet 转 json 后跑 `bench_sglang.py`；见 `boolq/README.md`。

#### `ceval/`
- 内容：C-Eval 考试题（few-shot），正则抽取选项 `A-D`，输出 `Accuracy / Latency / Output throughput`。
- 运行：下载 `ceval/ceval-exam` 数据集后跑 `bench_sglang.py`；见 `ceval/README.md`。

#### `gsm8k/`
- 内容：GSM8K 数学题（few-shot），抽取最终数值，输出 `Accuracy / Invalid / Latency / Output throughput`。
- 运行：`bench_sglang.py`（SGLang）以及 `bench_other.py`（vLLM/lightllm/guidance/lmql）；见 `gsm8k/README.md`。

#### `hellaswag/`
- 内容：HellaSwag 多选题（few-shot），使用 `sgl.select` 做选项选择，输出 `Accuracy / Latency`。
- 运行：`bench_sglang.py`（SGLang）以及 `bench_other.py`（vLLM/lightllm/guidance/lmql）；见 `hellaswag/README.md`。

#### `mmlu/`
- 内容：MMLU 多学科多选题（few-shot），输出各学科 acc 与整体平均 acc；支持 `--backend gpt-*`。
- 运行：先 `download_data.sh`，再跑 `bench_sglang.py` / `bench_other.py`；见 `mmlu/README.md`。

#### `mmmu/`
- 内容：MMMU 多模态评测（图片+文本问答），通过 OpenAI-compatible `/v1/chat/completions` 调用 SGLang VLM；支持并发、LoRA、答案抽取正则与额外请求参数。
- 运行：`bench_sglang.py`（SGLang server）与 `bench_hf.py`（HF baseline）；见 `mmmu/README.md`。

#### `mtbench/`
- 内容：MT-Bench 双轮对话生成（主要产出回答文件，便于后续 judge），并提供 SGLang EAGLE speculative 示例。
- 运行：`bench_sglang.py` / `bench_sglang_eagle.py` / `bench_other.py`；见 `mtbench/README.md`。

#### `llava_bench/`
- 内容：LLaVA Bench in the Wild（图像问答）与 MME 相关脚本；可用 SGLang VLM、本地 HF、OpenAI、llama.cpp 等后端。
- 运行：先 `download_images.py`，再跑 `bench_sglang.py` 或脚本 `bench_hf_*.sh`；见 `llava_bench/README.md`。

#### `reasoning_benchmark/`
- 内容：偏“推理模型”的数学/竞赛评测（默认 LIMO，也可 AIME 2024/2025），支持 `--num-tries` 统计上界（mean standard error）与超长输出（`max_new_tokens=32768`）。
- 运行：安装 `antlr4-python3-runtime`（用于数学等价判断），启动推理模型服务后跑 `bench_sglang.py`；见 `reasoning_benchmark/README.md`。

#### `dspy/`
- 内容：DSPy 框架的端到端示例/评测（HotPotQA + 检索 + 生成 + teleprompt 编译 + evaluate），用于对比 SGLang / TGI / vLLM 等后端在 DSPy 场景下的表现。
- 运行：按 `dspy/README.md` 安装 `dspy-ai` 并关闭其 cache（`DSP_CACHEBOOL=false`），启动对应 server 后跑 `bench_dspy_intro.py`；见 `dspy/README.md`。

### B) 程序化推理 / Agent / 多分支工作负载（fork/join、复现 trace）

#### `generative_agents/`
- 内容：复现 “Generative Agents” 风格的多种 agent function 调用序列（强调串行执行以保留依赖）。
- 运行：下载 `agent_calls.jsonl`，跑 `bench_sglang.py`（SGLang）或 `bench_other.py`（vLLM/guidance/lmql）；见 `generative_agents/README.md`。

#### `react/`
- 内容：ReAct 工作负载“回放”（不是完整 agent 实现），按给定 trace 拼接 Thought/Action/Observation，衡量吞吐/延迟。
- 运行：`bench_sglang.py` 或 `bench_other.py`；见 `react/README.md`。

#### `tree_of_thought_v0/`
- 内容：Tree-of-Thought（v0）在 GSM8K 上的多分支计划/求解/反思流程（用于吞吐/延迟对比，非追求最优准确率）。
- 运行：`bench_sglang.py` / `bench_other.py`；见 `tree_of_thought_v0/README.md`。

#### `tree_of_thought_deep/`
- 内容：更“深”的 Tree-of-Thought：计划 → 求解 → 自评 → 最终答案，多轮 fork 扩展分支并做投票。
- 运行：`bench_sglang.py` / `bench_other.py`；见 `tree_of_thought_deep/README.md`。

#### `multi_chain_reasoning/`
- 内容：同题多条 chain 并行生成（fork），再提示模型做 majority vote 汇总答案；输出 `Accuracy / Invalid / Latency`。
- 运行：`bench_sglang.py` / `bench_other.py`；见 `multi_chain_reasoning/README.md`。

#### `multi_document_qa/`
- 内容：多文档问答：将多个 docs 通过 `fork + join(concate_and_append)` 拼接进上下文，回答短答案并做简单匹配准确率。
- 运行：可用 `build_dataset.py` 从 PDF 构建数据；跑 `bench_sglang.py` / `bench_other.py`；见 `multi_document_qa/README.md`。

#### `multi_turn_chat/`
- 内容：合成多轮对话（可短/长输出），用于比较多轮场景下的延迟/吞吐。
- 运行：`bench_sglang.py --tokenizer <...> [--long]` 或 `bench_other.py`；见 `multi_turn_chat/README.md`。

#### `tip_suggestion/`
- 内容：嵌套生成工作负载：先生成多个短 tip，再对每个 tip 生成扩展段落（多次 `sgl.gen`/函数调用）。
- 运行：`bench_sglang.py` / `bench_other.py`；见 `tip_suggestion/README.md`。

#### `llm_judge/`
- 内容：多维度打分型 judge：对同一文章按多个维度 fork 生成评价，再汇总并给出最终分数（衡量 fork/join 编排开销）。
- 运行：`bench_sglang.py` / `bench_other.py`；见 `llm_judge/README.md`。

#### `line_retrieval/`
- 内容：长文本行检索任务，用 `fork(position_ids_offset=...)` 将上下文切块并行编码，观察准确率/延迟变化（用于评估并行编码/缓存策略）。
- 运行：下载随机词表并用 `gen_data.py` 生成数据后跑 `bench_sglang.py`；见 `line_retrieval/README.md`。

### C) 结构化输出 / 受约束解码（Regex / JSON Schema）

#### `json_schema/`
- 内容：`json_schema=` 约束生成（HuggingFace 数据集 `NousResearch/json-mode-eval`），并用 `jsonschema` 校验输出合法性；统计延迟与输出 token 数。
- 运行：启动模型后跑 `bench_sglang.py`；见 `json_schema/README.md`。

#### `json_decode_regex/`
- 内容：多字段 JSON 抽取（长文档）+ 每字段 regex 约束（`REGEX_STR/INT/FLOAT`），测试约束解码性能与鲁棒性。
- 运行：`build_dataset.py`（wikipedia）生成 `questions.jsonl`，再跑 `bench_sglang.py` / `bench_other.py`（outlines/guidance）；见 `json_decode_regex/README.md`。

#### `json_jump_forward/`
- 内容：单次 `sgl.gen(regex=...)` 生成整段固定结构 JSON（character/city 两种模式），用于对比“复杂 regex FSM”场景性能。
- 运行：`bench_sglang.py --mode character|city` / `bench_other.py`；见 `json_jump_forward/README.md`。

#### `long_json_decode/`
- 内容：从长文档抽取多个字段，但不使用复杂 regex/FSM，而是用多次 `sgl.gen(stop='\"')` 拼 JSON（更贴近“逐字段抽取”）。
- 运行：`build_dataset.py`（wikipedia）后跑 `bench_sglang.py` / `bench_other.py`；见 `long_json_decode/README.md`。

### D) Serving 吞吐 / 缓存 / LoRA / 非生成 API

#### `hicache/`
- 内容：层级缓存（Hierarchical Cache）相关 benchmark：合成多轮、Shared Prefix、以及更多数据集（sharegpt/ultrachat/loogle/nextqa 等，含 WIP 多模态）。
- 运行：按 `hicache/README.md` 启动不同缓存策略的 server，并运行 `bench_multiturn.py` 或 `bench_serving.py`；见 `hicache/README.md`。

#### `hf3fs/`
- 内容：HiCache 的 HF3FS 存储后端与零拷贝链路 microbench（分页读写带宽、批量 set/get、zerocopy backup/transfer）。
- 入口：`hf3fs/bench_client.py`、`hf3fs/bench_storage.py`、`hf3fs/bench_zerocopy.py`，以及整合示例 `hf3fs/bench.sh`（含 server+hicache 场景）。

#### `bench_in_batch_prefix/`
- 内容：大量共享前缀场景的 prefix caching 性能对比：逐 batch、带 hint 预热前缀、以及一次性全量发送三种模式。
- 运行：先启动 server（脚本顶部给了示例），再运行 `bench_in_batch_prefix/bench_in_batch_prefix.py`。

#### `benchmark_batch/`
- 内容：HTTP `/generate` 压测脚本（按“请求=一组 prompts”串行发送），统计 per-request/per-prompt latency 与 prompts/s；另有 tokenizer 批处理 vs 单条 `encode` 的 microbench。
- 入口：`benchmark_batch/benchmark_batch.py`（需改 `ENDPOINT_URL`、`TOKENIZER_DIR` 等配置）、`benchmark_batch/benchmark_tokenizer.py`（本地 tokenizer 性能）。

#### `prefill_only/`
- 内容：HTTP 压测框架（Poisson/Constant 流量、RPS/时长/批大小、可选 profiler/GC freeze），目前包含：
  - `bench_embeddings.py`：`/v1/embeddings`
  - `bench_score.py`：`/v1/score`
- 入口：`prefill_only/util.py` 提供通用构造请求、并发发送与统计逻辑。

#### `lora/`
- 内容：在线 LoRA serving benchmark：随机选择多个 LoRA adapter 发送请求，统计 TTFT/ITL/TPOT/吞吐，并将结果写入 JSONL。
- 入口：`lora/launch_server.py`（启动带 LoRA 的 server）、`lora/lora_bench.py`（压测客户端）。

### E) Kernel / Microbench（通常不依赖 server）

#### `bench_attention_sink/`
- 内容：Triton attention-sink 的 decode/extend kernel TFLOPS microbench（基于 `sglang.srt.layers.attention.triton_ops`）。
- 运行：`python3 bench_attention_sink/bench_attention_sink_triton.py --bench all|decode|extend`。

#### `kernels/`
- 内容：SGLang/sgl-kernel 的算子与通信 microbench、tuning 工具集合；详细索引见下文“`kernels/` 细分索引”。

### F) 复现实验/文章与对外发布结果

#### `benchmark_vllm_060/`
- 内容：复现 SGLang v0.3.0 vs vLLM v0.6.0 的在线/离线对比（重点指标：TTFT/ITL/throughput），给出安装与命令行。
- 运行：主要是文档；见 `benchmark_vllm_060/README.md`。

#### `blog_v0_2/`
- 内容：复现 SGLang v0.2.x 博客中的在线/离线 benchmark；包含 vLLM / TensorRT-LLM 对比与 405B dummy 权重跑分脚本。
- 入口：`blog_v0_2/README.md`、`blog_v0_2/405b_{sglang,vllm,trt}.sh`、`blog_v0_2/config.md`。

#### `deepseek_v3/`
- 内容：DeepSeek V3/V3.1/R1 在 SGLang 的部署与优化选项（MLA、DP-attention、torch.compile、量化、多机），并包含基于 `gsm8k/`、`sglang.bench_serving`、`sglang.bench_one_batch_server` 的 benchmark 示例命令。
- 运行：主要是文档；见 `deepseek_v3/README.md`。

#### `gpt_oss/`
- 内容：复现 GPT-OSS（120B）在不同硬件/量化下的吞吐、评测，以及 speculative decoding（EAGLE3）速度提升与 acceptance length 测试方法。
- 运行：主要是文档；见 `gpt_oss/README.md`。

## `kernels/` 细分索引

> 该目录以 microbench/tuning 为主，很多脚本需要 CUDA、多卡与 `torchrun`，并依赖 `flashinfer`、`sgl-kernel`、以及特定第三方库（如 DeepGEMM、DeepEP）。

#### `kernels/all_reduce/`
- 内容：多种 all-reduce 路径对比：SGLang 自定义 all-reduce vs Aiter、MSCCL++、Torch symmetric memory 等（含 CUDAGraph/eager 计时）。
- 运行：参考各脚本开头注释，通常用 `torchrun --nproc_per_node=<N> ...`。

#### `kernels/decoding_attention_triton/`
- 内容：decode attention 的实现对比与计时：SGLang Triton vs FlashInfer vs cuDNN（`triton_flashinfer_cudnn.py`）。

#### `kernels/deepep/`
- 内容：DeepEP（DeepSeek MoE/EP 通信相关）调参与校验（产出 `deepep_tuned.json`）。
- 运行：按脚本注释多机/多 rank 启动；见 `kernels/deepep/tuning_deepep.py`。

#### `kernels/deepseek/`
- 内容：DeepSeek FP8 GEMM / group GEMM kernel benchmark，对比 DeepGEMM 与 SGLang/vLLM Triton 实现。
- 运行：需先安装 DeepGEMM；见 `kernels/deepseek/README.md`。

#### `kernels/elementwise/`
- 内容：DeepSeek MLA 相关的拼接/elementwise kernel microbench（torch/torch.compile/triton/cuda 对比）。
- 入口：`kernels/elementwise/benchmark_concat_mla.py`。

#### `kernels/flashinfer_allreduce_fusion/`
- 内容：FlashInfer fused AllReduce + Residual Add + RMSNorm（可选 FP8/FP4 quant）对比标准实现，支持导出 Markdown 结果。
- 运行：`torchrun ... kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py`；见 `kernels/flashinfer_allreduce_fusion/README.md`。

#### `kernels/fused_moe_triton/`
- 内容：MoE fused kernel tuning/benchmark 工具（支持 TP/EP/MLLM、多 dtype、分离 kernel 调参、与 vLLM 对比）。
- 运行：以 `tuning_fused_moe_triton.py` / `tuning_fused_moe_triton_sep.py` 为主；见 `kernels/fused_moe_triton/README.md`。

#### `kernels/quantization/`
- 内容：W8A8 block-wise 量化 kernel 调参（Triton vs DeepGEMM 选择逻辑）、以及 FP4/INT8 量化 microbench。
- 运行：`tuning_block_wise_kernel.py`、`bench_fp4_quant.py`、`bench_int8_quant.py`；见 `kernels/quantization/README.md`。

#### `kernels/scheduler_batch/`
- 内容：调度器相关小 kernel 的 Triton vs Torch microbench（如 get_last_loc、write_req_to_token_pool）。
- 入口：`kernels/scheduler_batch/benchmark_get_last_loc_triton.py`、`kernels/scheduler_batch/benchmark_write_req_to_token_pool_triton.py`。

#### `kernels/sliding_window_attention_triton/`
- 内容：sliding-window attention 的 extend kernel（Triton vs Torch reference）性能与正确性验证。
- 入口：`kernels/sliding_window_attention_triton/bench_triton_swa_kernel.py`。
