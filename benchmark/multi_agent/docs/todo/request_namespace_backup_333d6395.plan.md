---
name: request namespace backup
overview: 将当前基于全局 token trie 的 remote_backup 设计，重构为基于 `request_id + generation` 的请求级命名空间，并按 `page_idx` 顺序存储 prefill/decode KV 页。方案保留同请求 failover/retry 的 prefill 召回能力，同时把远端内存生命周期收紧到请求级。
todos:
  - id: design-data-model
    content: 定义请求级命名空间数据模型与页生命周期，替换全局 token trie 作为主索引
    status: in_progress
  - id: design-protocol
    content: 设计 remote backup server/client 新读写协议与兼容迁移策略
    status: pending
  - id: migrate-write-path
    content: 规划 prefill/decode 写入链路如何统一到 request_id+generation+page_idx
    status: pending
  - id: migrate-read-path
    content: 规划 failover/retry prefetch 如何从 token 前缀匹配迁移到请求级范围召回
    status: pending
  - id: validation-tests
    content: 定义单测与端到端验证，覆盖 prefill 召回、正常完成释放、failover generation 切换
    status: pending
isProject: false
---

# Request-Level Remote Backup 方案

## 目标
把 `remote_backup` 从“全局共享 token 前缀缓存”重构为“按请求生命周期管理的恢复缓存”：
- 以 `request_id + generation` 作为主命名空间。
- 以 `page_idx` 顺序存储该请求的 `prefill` 和 `decode` KV 页。
- failover/retry 时只恢复**同一个请求**自己的远端页，不做跨请求共享前缀。
- 请求正常完成后，远端数据尽快释放，避免像当前实验一样持续占满 `40GB`。

现有代码里，写入侧已经具备必要的元数据入口，可以直接作为迁移锚点：[`cache_controller.py`](python/sglang/srt/managers/cache_controller.py) 的 `write_storage(... full_token_ids, page_start, request_id, request_generation)` 会把请求信息透传到存储层；[`remote_backup_storage.py`](python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py) 的 `batch_set_v1()` 也已经按 `request_id` 是否存在选择 v1/v2 写入路径。真正需要替换的是 server 端当前“token trie + all-source prefix match”的主索引和读路径。

## 方案对比
### 方案 A：保留全局 token trie，只把所有 prefill/decode 写入都改成 lease/v2
- 优点：改动较小，可复用现有 `insert_pages_v2()`、`start_request()`、`finish_request()`。
- 缺点：主索引仍然是“全局共享前缀”，不符合“只服务同请求 retry”的目标；读路径仍需 token 前缀匹配，写入仍有 `O(page_start)` 的前缀导航成本；内存语义仍偏共享缓存而不是恢复缓存。

### 方案 B：请求级命名空间 + 请求内顺序页仓库（推荐）
- 优点：写入/召回都直接围绕 `(rid, generation, page_idx)`，最贴合当前需求；`prefill` 和 `decode` 都可恢复；请求完成后可整批释放；避免全局 trie 的共享与扫描开销。
- 缺点：需要改 server/client 协议与读写 API，属于中等规模重构。

### 方案 C：双层结构（请求级恢复缓存 + 全局共享 trie）
- 优点：同时支持同请求恢复和跨请求共享。
- 缺点：明显超出当前需求，写入最重、状态最复杂，不建议进入本次范围。

推荐采用**方案 B**。

## 推荐架构
```mermaid
flowchart LR
    Scheduler[Scheduler]
    HiRadix[HiRadixCache]
    CacheCtrl[HiCacheController]
    Storage[RemoteBackupStorage]
    Client[RemoteBackupClient]
    Server[RemoteBackupServer]
    Namespace[RequestNamespaceStore]

    Scheduler --> HiRadix
    HiRadix --> CacheCtrl
    CacheCtrl --> Storage
    Storage --> Client
    Client --> Server
    Server --> Namespace
```

### 1. 服务端主索引替换
将 [`remote_backup_server.py`](python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_server.py) 中的 `RemoteBackupRadixBuffer` 从主索引位置降级/移除，新增请求级存储结构，例如：
- `namespaces[(dp_rank, rid, generation)] -> RequestNamespace`
- `RequestNamespace.pages[page_idx] -> bytes | page_ref`
- `RequestNamespace.state` 跟踪 `max_page_idx`、字节数、last_touch、active/inactive 状态

可保留现有 lease 状态机：
- `start_request(dp_rank, rid, generation)`
- `finish_request(dp_rank, rid, generation, reason)`
- `_gc_worker()` 的异步回收框架

但 `RequestLeaseEntry.pages` 不再存 trie `page_id` 集合，而改成命名空间页范围或页引用列表。这样 GC 可以直接按命名空间/页集合释放，而不是再走 `_node_by_page_id -> trie node`。

### 2. 远端协议升级
当前协议是 token 驱动的：
- `CMD_PUT_TOKENS(_V2)`
- `CMD_MATCH_PREFIX`
- `CMD_GET_PAGES`

对于请求级命名空间，计划新增或迁移为请求级命令：
- `PUT_PAGES_BY_REQUEST(dp_rank, rid, generation, page_start, page_count, pages[, token_meta])`
- `GET_PAGES_BY_REQUEST(dp_rank, rid, generation, page_start, count)`
- 可选 `QUERY_REQUEST_RANGE(dp_rank, rid, generation)`，用于返回已存在的最大连续页范围

其中 `full_token_ids` 可以作为调试/一致性校验元数据保留，但不再作为服务端主索引。这样写入复杂度从当前 trie 模型的 `O(page_start + pages)` 收敛到接近 `O(pages)`。

### 3. 写入链路迁移
#### prefill 写入
当前 prefill/radix 备份主要通过 [`hiradix_cache.py`](python/sglang/srt/mem_cache/hiradix_cache.py) 的 `write_backup_storage()` 进入 [`cache_controller.py`](python/sglang/srt/managers/cache_controller.py) 的 `write_storage()`；这条路径今天默认没有稳定绑定到请求级命名空间。

计划改为：
- prefill 写入时显式携带当前 `req.rid` 和 `req.remote_backup_generation`
- 继续传 `page_start`，但仅作为页序号，不再驱动 token trie 建路径
- `request_id/request_generation` 成为必须字段，而不是可选优化字段

#### decode 写入
[`decode_kv_replicator.py`](python/sglang/srt/mem_cache/decode_kv_replicator.py) 已经天然按页组织 decode 写入，并能算出：
- `full_token_ids`
- `page_start`
- `generation`

这条路径非常适合作为新模型的 decode 写入基线，直接迁移到请求级 `PUT_PAGES_BY_REQUEST` 即可。

### 4. prefetch / retry 召回链路迁移
当前 failover 重试仍走 token 驱动 prefetch：[`scheduler.py`](python/sglang/srt/managers/scheduler.py) 会基于 `req.fill_ids`、`matched_len`、`prefix_token_ids` 调用 [`hiradix_cache.py`](python/sglang/srt/mem_cache/hiradix_cache.py) 的 `prefetch_from_storage()`，再走到 [`remote_backup_storage.py`](python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py) 的 `match_prefix_from()` / `get_pages_tokens()`。

计划把这条链改成两段：
- **本地命中阶段**：仍保留现有本地 radix / host cache 匹配，确定本地已拥有的页数
- **远端恢复阶段**：对 failover/retry 请求，直接根据 `(rid, old_generation)` 和 `start_page` 发起范围读取，返回连续页数据到 host

这样保留了“prefill 不必整段重算”的能力，但去掉了远端全局 token 前缀匹配的复杂度。`failover_prefix_ids` 和 `full_token_ids` 仍可保留在请求对象里，作为页数计算和一致性校验依据，而不是远端主检索键。

## 生命周期与释放策略
结合你确认的需求，生命周期如下：
- 请求开始：创建 `(rid, generation)` 命名空间并标记 active
- prefill/decode 写入：持续向该命名空间追加或覆盖页
- failover：新 generation 创建新命名空间；旧 generation 保留，供 retry 读取
- failover 接管完成：旧 generation 进入可清理状态
- 正常完成：当前 generation 的命名空间立即进入 GC，尽快释放页
- abort / supersede：同样进入 GC

这要求服务端 GC 从“页级 lease 解绑 + LRU 回收”转为“命名空间级快速释放 + 必要时页级回收”。现有 `_gc_worker()` 仍可复用为异步回收执行器，但回收对象从 trie page_ids 改成 request namespace。

## 文件级改动计划
### 核心改动
- [`python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_server.py`](python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_server.py)
  - 新增请求级命名空间存储结构
  - 新增/替换请求级 PUT / GET / QUERY 命令
  - 重写 GC 回收对象与统计逻辑
  - 逐步下线 `RemoteBackupRadixBuffer` 作为主索引的职责

- [`python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py`](python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py)
  - 写路径切到请求级 PUT API
  - 读路径为 failover/retry 引入请求级范围读取接口
  - 保留 token 上下文作为调试/校验元数据，而非 match 主索引

- [`python/sglang/srt/managers/cache_controller.py`](python/sglang/srt/managers/cache_controller.py)
  - 强化 `write_storage()` 的请求级语义，确保 prefill/decode 都稳定携带 `request_id/request_generation/page_start`
  - 为新请求级读 API 预留 operation 类型或额外参数

- [`python/sglang/srt/mem_cache/hiradix_cache.py`](python/sglang/srt/mem_cache/hiradix_cache.py)
  - 调整 `write_backup_storage()`，让 prefill 备份绑定请求上下文
  - 调整 `prefetch_from_storage()` 的远端阶段，支持按请求页范围恢复
  - 保留本地 cache 匹配，但不再假设远端也必须 token prefix match

- [`python/sglang/srt/mem_cache/decode_kv_replicator.py`](python/sglang/srt/mem_cache/decode_kv_replicator.py)
  - 迁移 decode 写入到请求级 PUT API
  - 继续复用其 `page_start/full_token_ids/generation` 计算逻辑

### 测试与验证
- [`test/registered/unit/mem_cache/test_remote_backup_storage.py`](test/registered/unit/mem_cache/test_remote_backup_storage.py)
  - 补请求级 PUT / GET / release 的单测
- 新增 request namespace server 单测（建议放在 `test/registered/unit/mem_cache/`）
  - prefill 页按 `page_idx` 存取
  - decode 增量页追加与覆盖
  - `finish_request(normal)` 后命名空间被释放
  - failover generation 切换时旧 generation 可读、新 generation 独立
- benchmark/failover 场景验证
  - retry 时 prefill 不整段重算
  - backup server 内存不再长期顶住 40GB

## 风险与迁移策略
- **兼容性风险**：当前 `RemoteBackupStorage` 暴露的是 token-based API，直接替换会影响现有 `supports_token_matching` 假设。迁移时应优先在 `remote_backup` 后端内部新增请求级读写分支，避免一次性改掉全部 HiCache 抽象。
- **语义风险**：如果仍需保留某些 debug/metrics 依赖 token match，需要保留最小的 token 元数据，而不要完全删除 `full_token_ids/page_start`。
- **迁移建议**：先让写入链路全部进入请求级命名空间，再改 failover 读取链路，最后删除旧的 all-source token match 主路径。这样可以降低回归面并保留分阶段验证能力。

## 推荐实施顺序
1. 在 server 端增加请求级命名空间存储与请求级 PUT/GET 协议，不立即删除旧 trie 代码。
2. 让 `RemoteBackupStorage.batch_set_v1()` 在 remote_backup 场景下统一走请求级写入，并确保 prefill/decode 都带 `request_id/generation`。
3. 在 failover/retry 路径上新增请求级范围读取，先只用于 retry 请求。
4. 用单测和 benchmark 验证 prefill 召回、正常完成释放、generation 切换。
5. 验证通过后，再收缩或删除旧的 token-trie 主路径。
