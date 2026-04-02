from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _get_no_proxy_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


_session = _get_no_proxy_session()


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return s.strip("_") or "item"


@dataclass
class WorkerPromptTree:
    worker_url: str
    dp_trees: List[Dict[str, Any]]


class PromptTreeVisualizer:
    """
    Fetch and visualize radix tree prompt content from SGLang workers via /radixtree.

    The endpoint is expected to return a list (one entry per DP rank). Each entry contains:
    - nodes: list of {nodeId, parentId, depth, segmentTokenIds?, prefixTokenIds?, segmentText?, prefixText?}
    - edges: list of {parentId, childId}
    """

    def __init__(self, worker_urls: List[str], output_dir: str):
        self.worker_urls = worker_urls
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_worker_tree(
        self,
        worker_url: str,
        include_prefix: bool = True,
        include_segment: bool = True,
        include_text: bool = True,
        max_nodes: int = 2000,
        max_depth: int = 64,
        max_tokens_per_node: int = 4096,
        strict_sync: bool = True,
        sync_timeout_s: float = 5.0,
        timeout_s: int = 30,
    ) -> WorkerPromptTree:
        params = {
            "include_prefix": str(bool(include_prefix)).lower(),
            "include_segment": str(bool(include_segment)).lower(),
            "include_text": str(bool(include_text)).lower(),
            "max_nodes": str(int(max_nodes)),
            "max_depth": str(int(max_depth)),
            "max_tokens_per_node": str(int(max_tokens_per_node)),
            "strict_sync": str(bool(strict_sync)).lower(),
            "sync_timeout_s": str(float(sync_timeout_s)),
        }
        resp = _session.get(f"{worker_url}/radixtree", params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("radixtree response must be a list")
        if strict_sync:
            for i, tree in enumerate(data):
                if not isinstance(tree, dict):
                    continue
                converged = bool(tree.get("strictSyncConverged", True))
                if not converged:
                    raise RuntimeError(
                        f"strict sync not converged for worker={worker_url}, dp={i}. "
                        "Please increase --sync-timeout-s and retry."
                    )
        return WorkerPromptTree(worker_url=worker_url, dp_trees=data)

    def capture_all(
        self,
        strategy_name: str,
        include_prefix: bool = True,
        include_segment: bool = True,
        include_text: bool = True,
        max_nodes: int = 2000,
        max_depth: int = 64,
        max_tokens_per_node: int = 4096,
        strict_sync: bool = True,
        sync_timeout_s: float = 5.0,
        timeout_s: int = 30,
    ) -> Dict[str, Any]:
        captured: Dict[str, Any] = {
            "strategy": strategy_name,
            "timestamp": time.time(),
            "workers": {},
        }

        for url in self.worker_urls:
            try:
                tree = self.fetch_worker_tree(
                    worker_url=url,
                    include_prefix=include_prefix,
                    include_segment=include_segment,
                    include_text=include_text,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    max_tokens_per_node=max_tokens_per_node,
                    strict_sync=strict_sync,
                    sync_timeout_s=sync_timeout_s,
                    timeout_s=timeout_s,
                )
                captured["workers"][url] = {
                    "success": True,
                    "dp_trees": tree.dp_trees,
                }
            except Exception as e:
                captured["workers"][url] = {
                    "success": False,
                    "error": str(e),
                    "dp_trees": [],
                }

        return captured

    def export_to_json(self, captured: Dict[str, Any], filename: str) -> str:
        path = self.output_dir / filename
        with path.open("w") as f:
            json.dump(captured, f, indent=2, ensure_ascii=False)
        return str(path)

    def generate_html_report(
        self,
        captured: Dict[str, Any],
        filename: str,
        title: Optional[str] = None,
    ) -> str:
        path = self.output_dir / filename
        safe_title = title or "Radix Tree Prompt Visualization"
        payload_js = json.dumps(captured, ensure_ascii=False)
        payload_js = payload_js.replace("</", "<\\/")
        title_html = html.escape(safe_title)

        html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__SGLANG_PROMPT_TREE_TITLE__</title>
  <style>
    :root {{
      --c-bg: #f7f8fa;
      --c-surface: #ffffff;
      --c-border: #e2e5ea;
      --c-text: #1a1d23;
      --c-text-secondary: #6b7280;
      --c-accent: #4f6df5;
      --c-accent-hover: #3b5de7;
      --c-accent-light: #eef1fe;
      --c-badge-bg: #f0f2f5;
      --c-badge-text: #374151;
      --c-error: #dc2626;
      --c-success: #16a34a;
      --c-node-fill: #ffffff;
      --c-node-stroke: #94a3b8;
      --c-node-hover: #4f6df5;
      --c-link: #cbd5e1;
      --c-node-gpu-only-fill: #e0f2fe;
      --c-node-gpu-only-stroke: #0284c7;
      --c-node-gpu-host-fill: #dcfce7;
      --c-node-gpu-host-stroke: #16a34a;
      --c-node-host-only-fill: #fef3c7;
      --c-node-host-only-stroke: #d97706;
      --c-node-none-fill: #fee2e2;
      --c-node-none-stroke: #dc2626;
      --c-node-unknown-fill: #f1f5f9;
      --c-node-unknown-stroke: #64748b;
      --radius: 8px;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 0; background: var(--c-bg); color: var(--c-text); }}

    /* ---- Top bar ---- */
    .topbar {{
      padding: 16px 24px;
      background: var(--c-surface);
      border-bottom: 1px solid var(--c-border);
      position: sticky; top: 0; z-index: 10;
      box-shadow: var(--shadow-sm);
    }}
    .topbar-header {{
      display: flex; align-items: center; gap: 12px;
    }}
    .logo {{
      width: 28px; height: 28px; border-radius: 6px;
      background: linear-gradient(135deg, var(--c-accent), #7c3aed);
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-weight: 700; font-size: 14px; flex-shrink: 0;
    }}
    .title {{ font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }}
    .meta {{ color: var(--c-text-secondary); font-size: 12px; margin-top: 2px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

    /* ---- Controls ---- */
    .controls {{
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px;
    }}
    .control-group {{
      display: flex; align-items: center; gap: 4px;
      background: var(--c-badge-bg); border-radius: var(--radius); padding: 4px 8px;
    }}
    .control-group label {{
      font-size: 11px; font-weight: 600; color: var(--c-text-secondary);
      text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;
    }}
    select, input[type="text"], input[type="number"] {{
      padding: 6px 10px; border: 1px solid var(--c-border); border-radius: 6px;
      font-size: 13px; background: var(--c-surface); color: var(--c-text);
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    select:focus, input:focus {{
      outline: none; border-color: var(--c-accent);
      box-shadow: 0 0 0 3px var(--c-accent-light);
    }}
    input[type="text"] {{ width: 400px; max-width: 60vw; }}
    input[type="number"] {{ width: 80px; text-align: center; }}
    input[type="checkbox"] {{ accent-color: var(--c-accent); width: 16px; height: 16px; cursor: pointer; }}
    .btn {{
      padding: 6px 14px; border: 1px solid var(--c-border); border-radius: 6px;
      background: var(--c-surface); cursor: pointer; font-size: 13px; font-weight: 500;
      color: var(--c-text); transition: all 0.15s;
    }}
    .btn:hover {{ background: var(--c-badge-bg); border-color: #ccc; }}
    .btn-primary {{
      background: var(--c-accent); color: #fff; border-color: var(--c-accent);
    }}
    .btn-primary:hover {{ background: var(--c-accent-hover); border-color: var(--c-accent-hover); }}

    /* ---- Layout ---- */
    .layout {{
      display: grid; grid-template-columns: 1fr 440px; gap: 0;
      height: calc(100vh - 130px);
    }}
    .canvas {{
      background: var(--c-bg); overflow: hidden; position: relative;
      background-image: radial-gradient(circle, #ddd 1px, transparent 1px);
      background-size: 20px 20px;
    }}

    /* ---- Sidebar ---- */
    .sidebar {{
      background: var(--c-surface); border-left: 1px solid var(--c-border);
      padding: 20px; overflow: auto;
    }}
    .sidebar-title {{
      font-size: 13px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--c-text-secondary); margin-bottom: 12px;
    }}
    .legend {{
      display: none;
      margin-bottom: 10px;
      padding: 10px;
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      background: var(--c-bg);
    }}
    .legend-title {{
      font-size: 11px;
      font-weight: 700;
      color: var(--c-text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .agent-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--c-border);
      background: var(--c-surface);
      font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      margin: 2px 6px 2px 0;
      white-space: nowrap;
    }}
    .agent-chip .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: none;
    }}
    .hint {{
      font-size: 13px; color: var(--c-text-secondary); margin-top: 12px;
      padding: 12px; background: var(--c-accent-light); border-radius: var(--radius);
      line-height: 1.5;
    }}
    .badge {{
      display: inline-block; padding: 3px 8px; border-radius: 20px;
      background: var(--c-badge-bg); font-size: 12px; font-weight: 500;
      color: var(--c-badge-text); margin: 2px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .inspector {{
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .inspector-section {{
      border: 1px solid var(--c-border);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--c-bg);
    }}
    .inspector-section summary {{
      cursor: pointer;
      padding: 10px 12px;
      background: var(--c-surface);
      border-bottom: 1px solid var(--c-border);
      font-size: 12px;
      font-weight: 700;
      color: var(--c-text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      list-style: none;
    }}
    .inspector-section summary::-webkit-details-marker {{ display: none; }}
    .inspector-seg summary {{ border-left: 4px solid #2563eb; }}
    .inspector-pref summary {{ border-left: 4px solid #7c3aed; }}
    .inspector-pre {{
      margin: 0;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      line-height: 1.55;
      max-height: 32vh;
      overflow: auto;
    }}
    .err {{ color: var(--c-error); font-weight: 600; }}

    /* ---- SVG tree ---- */
    svg {{ width: 100%; height: 100%; }}
    .link {{ fill: none; stroke: var(--c-link); stroke-width: 1.6px; }}
    .node .node-dot {{
      fill: var(--c-node-fill);
      stroke: var(--c-node-stroke);
      stroke-width: 1.6px;
      cursor: pointer;
      transition: fill 0.15s, stroke 0.15s, stroke-width 0.15s;
    }}
    .node:hover .node-dot {{
      fill: var(--c-accent-light);
      stroke: var(--c-node-hover);
      stroke-width: 2.2px;
    }}
    .node .node-dot.node-dot-gpu-only {{
      fill: var(--c-node-gpu-only-fill);
      stroke: var(--c-node-gpu-only-stroke);
    }}
    .node .node-dot.node-dot-gpu-host {{
      fill: var(--c-node-gpu-host-fill);
      stroke: var(--c-node-gpu-host-stroke);
    }}
    .node .node-dot.node-dot-host-only {{
      fill: var(--c-node-host-only-fill);
      stroke: var(--c-node-host-only-stroke);
    }}
    .node .node-dot.node-dot-none {{
      fill: var(--c-node-none-fill);
      stroke: var(--c-node-none-stroke);
    }}
    .node .node-dot.node-dot-unknown {{
      fill: var(--c-node-unknown-fill);
      stroke: var(--c-node-unknown-stroke);
    }}
    .node.selected .node-dot {{
      fill: var(--c-accent);
      stroke: var(--c-accent);
      stroke-width: 2.2px;
    }}
    .node .node-label {{
      font-size: 11px;
      fill: var(--c-text);
      pointer-events: none;
    }}
    .node .node-seglen {{
      font-size: 10px;
      fill: var(--c-text-secondary);
      pointer-events: none;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .node.selected .node-label {{
      fill: var(--c-accent);
      font-weight: 600;
    }}
	    .agent-hit circle {{
	      stroke: rgba(255,255,255,0.95);
	      stroke-width: 1.4px;
	    }}
	    .agent-hit .hit-initial {{
	      font-size: 9px;
	      font-weight: 800;
	      fill: #fff;
	      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	      pointer-events: none;
	    }}
	    .agent-hit .hit-count {{
	      font-size: 11px;
	      font-weight: 700;
	      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	      pointer-events: none;
	    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-header">
      <div class="logo">R</div>
      <div>
        <div class="title">__SGLANG_PROMPT_TREE_TITLE__</div>
        <div class="meta" id="meta"></div>
      </div>
    </div>
    <div class="controls">
      <div class="control-group">
        <label>Worker</label>
        <select id="worker"></select>
      </div>
      <div class="control-group">
        <label>DP</label>
        <select id="dp"></select>
      </div>
      <div class="control-group">
        <label>Filter</label>
        <input id="filter" type="text" placeholder="Search prefix or segment text...">
      </div>
      <div class="control-group">
        <label>Nodes</label>
        <input id="maxNodes" type="number" value="2000" min="1">
      </div>
      <div class="control-group">
        <label>X</label>
	        <input id="dx" type="number" value="48" min="4">
      </div>
      <div class="control-group">
        <label>Y</label>
        <input id="dy" type="number" value="180" min="40">
      </div>
      <div class="control-group">
        <label>Labels</label>
        <input id="labels" type="checkbox">
      </div>
      <button class="btn btn-primary" id="apply">Apply</button>
      <button class="btn" id="fit">Fit</button>
      <button class="btn" id="resetZoom">Reset</button>
    </div>
  </div>

  <div class="layout">
    <div class="canvas">
      <svg id="svg"></svg>
    </div>
    <div class="sidebar">
      <div class="sidebar-title">Node Inspector</div>
      <div id="agentLegend" class="legend"></div>
      <div id="residencyLegend" class="legend"></div>
      <div id="selMeta"></div>
      <div class="hint" id="hintBox">Click any node in the tree to inspect its prefix and segment text.</div>
      <div id="sel" class="inspector" style="display:none;"></div>
    </div>
  </div>

  <script>
    const DATA = __SGLANG_PROMPT_TREE_DATA__;

    function keys(obj) {{
      return Object.keys(obj || {{}}).sort();
    }}

    function nodeText(n) {{
      const p = (n && n.prefixText) ? String(n.prefixText) : "";
      const s = (n && n.segmentText) ? String(n.segmentText) : "";
      return (p + "\\n" + s).trim();
    }}

    function escapeHtml(s) {{
      return String(s ?? "").replace(/[&<>"']/g, (ch) => {{
        switch (ch) {{
          case "&": return "&amp;";
          case "<": return "&lt;";
          case ">": return "&gt;";
          case '"': return "&quot;";
          case "'": return "&#39;";
          default: return ch;
        }}
      }});
    }}

    function hexToRgba(hex, alpha) {{
      let h = String(hex || "").trim();
      if (!h) return `rgba(100,116,139,${alpha})`;
      if (h.startsWith("#")) h = h.slice(1);
      if (h.length === 3) h = h.split("").map((c) => c + c).join("");
      if (h.length !== 6) return `rgba(100,116,139,${alpha})`;
      const r = parseInt(h.slice(0, 2), 16);
      const g = parseInt(h.slice(2, 4), 16);
      const b = parseInt(h.slice(4, 6), 16);
      if (![r, g, b].every((v) => Number.isFinite(v))) return `rgba(100,116,139,${alpha})`;
      return `rgba(${r},${g},${b},${alpha})`;
    }}

    const AGENT_PALETTE = [
      "#2563eb", // blue
      "#dc2626", // red
      "#16a34a", // green
      "#7c3aed", // purple
      "#d97706", // amber
      "#0891b2", // cyan
      "#db2777", // pink
      "#0f766e", // teal
    ];

    let _agentList = [];
    let _agentColors = {{}};
    const RESIDENCY_META = {{
      gpu_only: {{ label: "GPU only", color: "#0284c7" }},
      gpu_host: {{ label: "GPU+CPU", color: "#16a34a" }},
      host_only: {{ label: "CPU only", color: "#d97706" }},
      none: {{ label: "Not resident", color: "#dc2626" }},
      unknown: {{ label: "Unknown", color: "#64748b" }},
    }};

    function normalizeResidency(v) {{
      if (v === null || v === undefined) return "unknown";
      const s = String(v).trim().toLowerCase().replace(/[-\\s]/g, "_");
      if (s === "gpu_only" || s === "gpu") return "gpu_only";
      if (s === "gpu_host" || s === "gpu_cpu" || s === "gpu+cpu") return "gpu_host";
      if (s === "host_only" || s === "cpu_only" || s === "host" || s === "cpu") return "host_only";
      if (s === "none" || s === "not_resident") return "none";
      return "unknown";
    }}

    function toInt(v) {{
      const x = Number(v);
      if (!Number.isFinite(x)) return 0;
      if (x <= 0) return 0;
      return Math.floor(x);
    }}

    function readyTokenStats(node) {{
      const gpuRaw = toInt(node && node.gpuKvTokensRaw);
      const cpuRaw = toInt(node && node.cpuKvTokensRaw);
      const writePending = !!(node && node.writePending);
      const loadPending = !!(node && node.loadPending);
      const gpuReady = loadPending ? 0 : gpuRaw;
      const cpuReady = writePending ? 0 : cpuRaw;
      return {{
        gpuRaw,
        cpuRaw,
        gpuReady,
        cpuReady,
        writePending,
        loadPending,
      }};
    }}

    function inferResidency(node) {{
      if (!node) return "unknown";
      const stats = readyTokenStats(node);
      if (stats.gpuReady > 0) return stats.cpuReady > 0 ? "gpu_host" : "gpu_only";
      if (stats.cpuReady > 0) return "host_only";
      if (stats.gpuRaw > 0 || stats.cpuRaw > 0) return "unknown";
      const explicit = normalizeResidency(node.cacheResidency);
      if (explicit !== "unknown") return explicit;
      const hasEvicted = Object.prototype.hasOwnProperty.call(node, "evicted");
      const hasBackuped = Object.prototype.hasOwnProperty.call(node, "backuped");
      if (!hasEvicted && !hasBackuped) return "unknown";
      const evicted = !!node.evicted;
      const backuped = !!node.backuped;
      if (evicted) return backuped ? "host_only" : "none";
      return backuped ? "gpu_host" : "gpu_only";
    }}

    function residencyLabel(key) {{
      const meta = RESIDENCY_META[key] || RESIDENCY_META.unknown;
      return meta.label;
    }}

    function residencyColor(key) {{
      const meta = RESIDENCY_META[key] || RESIDENCY_META.unknown;
      return meta.color;
    }}

    function computeAgentColors(nodes) {{
      const set = new Set();
      for (const n of (nodes || [])) {{
        const hits = (n && n.agentHits) ? n.agentHits : {{}};
        for (const a of Object.keys(hits)) set.add(a);
      }}
      const list = Array.from(set).sort();
      const colors = {{}};
      for (let i = 0; i < list.length; i++) {{
        colors[list[i]] = AGENT_PALETTE[i % AGENT_PALETTE.length];
      }}
      _agentList = list;
      _agentColors = colors;
    }}

    function renderAgentLegend() {{
      const el = document.getElementById("agentLegend");
      if (!el) return;
      if (!_agentList.length) {{
        el.style.display = "none";
        el.innerHTML = "";
        return;
      }}
      const parts = [];
      parts.push('<div class="legend-title">Agents</div>');
      for (const a of _agentList) {{
        const c = _agentColors[a] || "#64748b";
        parts.push(
          '<span class="agent-chip"><span class="swatch" style="background:' +
            c +
            '"></span>' +
            escapeHtml(a) +
            "</span>"
        );
      }}
      el.innerHTML = parts.join("");
      el.style.display = "";
    }}

    function renderResidencyLegend(nodes) {{
      const el = document.getElementById("residencyLegend");
      if (!el) return;

      const counts = {{
        gpu_only: 0,
        gpu_host: 0,
        host_only: 0,
        none: 0,
        unknown: 0,
      }};
      for (const n of (nodes || [])) {{
        const k = inferResidency(n);
        counts[k] = (counts[k] || 0) + 1;
      }}

      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      if (!total) {{
        el.style.display = "none";
        el.innerHTML = "";
        return;
      }}

      const ordered = ["gpu_only", "gpu_host", "host_only"];
      if (counts.none > 0) ordered.push("none");
      if (counts.unknown > 0) ordered.push("unknown");

      const parts = [];
      parts.push('<div class="legend-title">Cache Residency</div>');
      for (const k of ordered) {{
        const c = counts[k] || 0;
        if (!c) continue;
        const meta = RESIDENCY_META[k] || RESIDENCY_META.unknown;
        parts.push(
          '<span class="agent-chip"><span class="swatch" style="background:' +
            meta.color +
            '"></span>' +
            escapeHtml(meta.label) +
            ": " +
            c +
            "</span>"
        );
      }}
      el.innerHTML = parts.join("");
      el.style.display = "";
    }}

    function buildGraph(nodes, edges) {{
      const nodeMap = new Map();
      for (const n of (nodes || [])) {{
        nodeMap.set(n.nodeId, n);
      }}
      const children = new Map();
      for (const n of (nodes || [])) {{
        children.set(n.nodeId, []);
      }}
      for (const e of (edges || [])) {{
        if (!children.has(e.parentId)) children.set(e.parentId, []);
        children.get(e.parentId).push(e.childId);
      }}
      for (const [k, v] of children) {{
        v.sort((a, b) => a - b);
      }}
      const roots = [];
      for (const n of (nodes || [])) {{
        if (n.parentId === null || n.parentId === undefined) roots.push(n.nodeId);
      }}
      roots.sort((a, b) => a - b);
      return {{ nodeMap, children, roots }};
    }}

    function bfsLimit(roots, children, maxNodes) {{
      const out = new Set();
      const q = [];
      for (const r of roots) q.push(r);
      while (q.length && out.size < maxNodes) {{
        const id = q.shift();
        if (out.has(id)) continue;
        out.add(id);
        const kids = children.get(id) || [];
        for (const k of kids) {{
          if (out.size >= maxNodes) break;
          q.push(k);
        }}
      }}
      return out;
    }}

    function filterNodes(roots, nodeMap, children, filter) {{
      if (!filter) return null;
      const f = String(filter).toLowerCase();
      const keep = new Set();
      const parent = new Map();

      const q = [];
      for (const r of roots) {{
        q.push(r);
        parent.set(r, null);
      }}
      while (q.length) {{
        const id = q.shift();
        const kids = children.get(id) || [];
        for (const k of kids) {{
          if (!parent.has(k)) parent.set(k, id);
          q.push(k);
        }}
      }}

      for (const [id, n] of nodeMap.entries()) {{
        const t = nodeText(n).toLowerCase();
        if (t.includes(f)) {{
          let cur = id;
          while (cur !== null && cur !== undefined) {{
            keep.add(cur);
            cur = parent.get(cur);
          }}
        }}
      }}
      return keep;
    }}

    function buildHierarchy(roots, nodeMap, children, allowedSet) {{
      function build(id) {{
        const n = nodeMap.get(id) || {{}};
        const kids = children.get(id) || [];
        const nextKids = [];
        for (const k of kids) {{
          if (!allowedSet || allowedSet.has(k)) {{
            nextKids.push(build(k));
          }}
        }}
        const labelSrc = (n.segmentText || n.prefixText || "");
        const oneLine = String(labelSrc).replace(/\\s+/g, " ").slice(0, 60);
        return {{
          id: id,
          label: "node=" + id + " " + oneLine,
          raw: n,
          children: nextKids
        }};
      }}

      const rootChildren = [];
      for (const r of roots) {{
        if (!allowedSet || allowedSet.has(r)) {{
          rootChildren.push(build(r));
        }}
      }}
      // If there's a single root, use it directly (avoids an extra synthetic root node).
      if (rootChildren.length === 1) {{
        return rootChildren[0];
      }}
      return {{
        id: -1,
        label: "root",
        raw: null,
        children: rootChildren
      }};
    }}

    const workerSel = document.getElementById("worker");
    const dpSel = document.getElementById("dp");
    const metaEl = document.getElementById("meta");
    const selMeta = document.getElementById("selMeta");
    const selEl = document.getElementById("sel");

    let _selectedG = null;
    const hintBox = document.getElementById("hintBox");

    function setSelected(node) {{
      if (_selectedG) _selectedG.classList.remove("selected");
      _selectedG = null;

      if (!node) {{
        selMeta.innerHTML = "";
        selEl.style.display = "none";
        hintBox.style.display = "";
        return;
      }}
      hintBox.style.display = "none";
      const n = node.raw ? node.raw : null;
      if (!n) {{
        selMeta.innerHTML = "<span class=\\"badge\\">root</span>";
        selEl.style.display = "none";
        return;
      }}
      const parts = [];
      parts.push("<span class=\\"badge\\">node=" + n.nodeId + "</span>");
      parts.push("<span class=\\"badge\\">depth=" + n.depth + "</span>");
      parts.push("<span class=\\"badge\\">segLen=" + n.segmentLen + "</span>");
      parts.push("<span class=\\"badge\\">prefixLen=" + n.prefixLen + "</span>");
      parts.push("<span class=\\"badge\\">children=" + n.numChildren + "</span>");
      const residency = inferResidency(n);
      const residencyCol = residencyColor(residency);
      const residencyBg = hexToRgba(residencyCol, 0.12);
      const stats = readyTokenStats(n);
      parts.push(
        '<span class=\\"badge\\" style=\\"background:' +
          residencyBg +
          ";color:" +
          residencyCol +
          ";border:1px solid " +
          hexToRgba(residencyCol, 0.35) +
          '\\">cache=' +
          escapeHtml(residencyLabel(residency)) +
          "</span>"
      );
      parts.push("<span class=\\"badge\\">evicted=" + String(!!n.evicted) + "</span>");
      parts.push("<span class=\\"badge\\">backuped=" + String(!!n.backuped) + "</span>");
      parts.push("<span class=\\"badge\\">gpu_kv=" + String(stats.gpuReady) + "</span>");
      parts.push("<span class=\\"badge\\">cpu_kv=" + String(stats.cpuReady) + "</span>");
      if (stats.gpuRaw !== stats.gpuReady) {{
        parts.push("<span class=\\"badge\\">gpu_kv_raw=" + String(stats.gpuRaw) + "</span>");
      }}
      if (stats.cpuRaw !== stats.cpuReady) {{
        parts.push("<span class=\\"badge\\">cpu_kv_raw=" + String(stats.cpuRaw) + "</span>");
      }}
      if (stats.writePending) {{
        parts.push("<span class=\\"badge\\">write_pending=true</span>");
      }}
      if (stats.loadPending) {{
        parts.push("<span class=\\"badge\\">load_pending=true</span>");
      }}
      const agentHits = n.agentHits || {{}};
      const agentKeys = Object.keys(agentHits).sort();
      if (agentKeys.length > 0) {{
        for (const a of agentKeys) {{
          const col = _agentColors[a] || "#64748b";
          const bg = hexToRgba(col, 0.12);
          parts.push(
            '<span class=\\"badge\\" style=\\"background:' +
              bg +
              ";color:" +
              col +
              ";border:1px solid " +
              hexToRgba(col, 0.35) +
              '\\">' +
              escapeHtml(a) +
              ": " +
              agentHits[a] +
              "</span>"
          );
        }}
      }}
      selMeta.innerHTML = parts.join(" ");

      const sections = [];
      if (n.segmentText !== undefined) {{
        sections.push(
          '<details open class=\\"inspector-section inspector-seg\\"><summary>Segment Text</summary><pre class=\\"inspector-pre\\">' +
            escapeHtml(n.segmentText || "") +
            "</pre></details>"
        );
      }}
      if (n.prefixText !== undefined) {{
        sections.push(
          '<details class=\\"inspector-section inspector-pref\\"><summary>Prefix Text</summary><pre class=\\"inspector-pre\\">' +
            escapeHtml(n.prefixText || "") +
            "</pre></details>"
        );
      }}
      if (!sections.length) {{
        sections.push(
          '<div class=\\"hint\\">No decoded text found on this node. Re-run capture without <code>--no-text</code>.</div>'
        );
      }}
      selEl.innerHTML = sections.join("");
      selEl.style.display = "";
    }}

    const SVG_NS = "http://www.w3.org/2000/svg";
    const svg = document.getElementById("svg");
    const gZoom = document.createElementNS(SVG_NS, "g");
    const gLinks = document.createElementNS(SVG_NS, "g");
    const gNodes = document.createElementNS(SVG_NS, "g");
    gZoom.appendChild(gLinks);
    gZoom.appendChild(gNodes);
    svg.appendChild(gZoom);

    let _scale = 1.0;
    let _tx = 0.0;
    let _ty = 0.0;
    let _dragging = false;
    let _dragStart = null;

    function applyTransform() {{
      gZoom.setAttribute("transform", `translate(${_tx},${_ty}) scale(${_scale})`);
    }}

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    svg.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const rect = svg.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      const k = ev.deltaY < 0 ? 1.1 : 0.9;
      const newScale = clamp(_scale * k, 0.1, 5.0);
      const sx = (mx - _tx) / _scale;
      const sy = (my - _ty) / _scale;
      _tx = mx - sx * newScale;
      _ty = my - sy * newScale;
      _scale = newScale;
      applyTransform();
    }}, {{ passive: false }});

    svg.addEventListener("mousedown", (ev) => {{
      _dragging = true;
      _dragStart = {{ x: ev.clientX, y: ev.clientY, tx: _tx, ty: _ty }};
    }});

    window.addEventListener("mousemove", (ev) => {{
      if (!_dragging || !_dragStart) return;
      _tx = _dragStart.tx + (ev.clientX - _dragStart.x);
      _ty = _dragStart.ty + (ev.clientY - _dragStart.y);
      applyTransform();
    }});

    window.addEventListener("mouseup", () => {{
      _dragging = false;
      _dragStart = null;
    }});

    function clearSvg() {{
      while (gLinks.firstChild) gLinks.removeChild(gLinks.firstChild);
      while (gNodes.firstChild) gNodes.removeChild(gNodes.firstChild);
      setSelected(null);
    }}

    function fitToContent(padding = 20) {{
      const bbox = gZoom.getBBox();
      const w = svg.clientWidth;
      const h = svg.clientHeight;
      if (bbox.width === 0 || bbox.height === 0) return;
      if (!(w > 0 && h > 0)) return;
      const scale = Math.min((w - padding * 2) / bbox.width, (h - padding * 2) / bbox.height);
      const tx = (w / 2) - scale * (bbox.x + bbox.width / 2);
      const ty = (h / 2) - scale * (bbox.y + bbox.height / 2);
      if (!Number.isFinite(scale) || !Number.isFinite(tx) || !Number.isFinite(ty)) return;
      _scale = clamp(scale, 0.1, 5.0);
      _tx = tx;
      _ty = ty;
      applyTransform();
    }}

    function computeLayout(root, dx, dy) {{
      let nextX = 0;

      function post(node, depth) {{
        node.depth = depth;
        if (!node.children || node.children.length === 0) {{
          node._x = nextX;
          nextX += 1;
          return;
        }}
        for (const c of node.children) {{
          post(c, depth + 1);
        }}
        let sum = 0;
        for (const c of node.children) sum += c._x;
        node._x = sum / node.children.length;
      }}

      post(root, 0);

      const placed = [];
      const edges = [];

      function walk(node) {{
        const x = node._x * dx;
        const y = node.depth * dy;
        placed.push({{ ...node, x, y }});
        if (node.children) {{
          for (const c of node.children) {{
            edges.push({{ from: node.id, to: c.id }});
            walk(c);
          }}
        }}
      }}

      walk(root);
      return {{ nodes: placed, edges }};
    }}

    function computeMinDx(nodes, nodeR, hitDotR, hitX, hitCountX, padding, segPad) {{
      let maxCountChars = 0;
      let maxSegChars = 0;
      for (const n of (nodes || [])) {{
        const hits = (n && n.agentHits) ? n.agentHits : {{}};
        for (const k of Object.keys(hits)) {{
          const v = hits[k];
          if (!v) continue;
          const text = String(v > 999 ? "999+" : v);
          if (text.length > maxCountChars) maxCountChars = text.length;
        }}
        const segLen = n ? n.segmentLen : null;
        if (segLen !== null && segLen !== undefined) {{
          const segText = String(segLen);
          if (segText.length > maxSegChars) maxSegChars = segText.length;
        }}
      }}
      if (!maxCountChars && !maxSegChars) return 0;
      const charW = 7;
      const countTextW = maxCountChars * charW + 2;
      const rightExt = hitX + hitCountX + countTextW + hitDotR;
      const segTextW = maxSegChars * charW + 2;
      const extraLeft = maxSegChars ? (segTextW + segPad) : 0;
      return nodeR + rightExt + nodeR + padding + extraLeft;
    }}

    function makePath(p0, p1) {{
      const my = (p0.y + p1.y) / 2;
      return `M ${p0.x} ${p0.y} C ${p0.x} ${my}, ${p1.x} ${my}, ${p1.x} ${p1.y}`;
    }}

    function renderTree() {{
      clearSvg();
      try {{
        const workerUrl = workerSel.value;
        const dpIdx = Number(dpSel.value || 0);
        const filter = document.getElementById("filter").value || "";
        const maxNodes = Math.max(1, Number(document.getElementById("maxNodes").value || 2000));
	        const dx = Math.max(4, Number(document.getElementById("dx").value || 48));
        const dy = Math.max(40, Number(document.getElementById("dy").value || 180));
        const showLabels = !!document.getElementById("labels").checked;
	        const NODE_R = 11;
	        const HIT_DOT_R = 6;
	        const HIT_STEP = 16;
	        const HIT_X = NODE_R + HIT_DOT_R + 10;
	        const HIT_COUNT_X = HIT_DOT_R + 8;
	        const MAX_AGENT_BADGES = 6;
	        const HIT_PAD = 8;
	        const SEG_PAD = 8;

        const w = (DATA.workers || {{}})[workerUrl];
        if (!w || !w.success) {{
          metaEl.textContent = "strategy=" + (DATA.strategy || "") + "  |  worker=" + workerUrl + "  |  status=error";
          const txt = w && w.error ? String(w.error) : "unknown";
          const legendEl = document.getElementById("agentLegend");
          if (legendEl) legendEl.style.display = "none";
          const residencyLegendEl = document.getElementById("residencyLegend");
          if (residencyLegendEl) residencyLegendEl.style.display = "none";
          selMeta.innerHTML = "<span class=\\"err\\">error</span>";
          selEl.textContent = txt;
          return;
        }}
        const t = (w.dp_trees || [])[dpIdx] || {{}};
        const nodes = t.nodes || [];
        const edges = t.edges || [];
        metaEl.textContent =
          "strategy=" + (DATA.strategy || "") +
          "  |  worker=" + workerUrl +
          "  |  dp=" + dpIdx +
          "  |  nodes=" + nodes.length +
          "  |  truncated=" + String(!!t.truncated) +
          "  |  strict_sync=" + String(!!t.strictSyncRequested) +
          "  |  converged=" + String(!!t.strictSyncConverged) +
          "  |  gpu_kv=" + String(toInt(t.gpuKvTokens)) +
          "  |  cpu_kv=" + String(toInt(t.cpuKvTokens));

        computeAgentColors(nodes);
        renderAgentLegend();
        renderResidencyLegend(nodes);

        const {{ nodeMap, children, roots }} = buildGraph(nodes, edges);
        const limitSet = bfsLimit(roots, children, maxNodes);
        const filterSet = filterNodes(roots, nodeMap, children, filter);
        let allowed = limitSet;
        if (filterSet) {{
          const inter = new Set();
          for (const id of allowed) {{
            if (filterSet.has(id)) inter.add(id);
          }}
          allowed = inter;
        }}

        const treeData = buildHierarchy(roots, nodeMap, children, allowed);
        const minDx = computeMinDx(nodes, NODE_R, HIT_DOT_R, HIT_X, HIT_COUNT_X, HIT_PAD, SEG_PAD);
        const layout = computeLayout(treeData, Math.max(dx, minDx), dy);

        const pos = new Map();
        for (const n of layout.nodes) {{
          pos.set(n.id, n);
        }}

        for (const e of layout.edges) {{
          const p0 = pos.get(e.from);
          const p1 = pos.get(e.to);
          if (!p0 || !p1) continue;
          const path = document.createElementNS(SVG_NS, "path");
          path.setAttribute("class", "link");
          path.setAttribute("d", makePath(p0, p1));
          gLinks.appendChild(path);
        }}

        for (const n of layout.nodes) {{
          const g = document.createElementNS(SVG_NS, "g");
          g.setAttribute("class", "node");
          g.setAttribute("transform", `translate(${n.x},${n.y})`);
          g.addEventListener("click", (ev) => {{
            ev.stopPropagation();
            if (_selectedG) _selectedG.classList.remove("selected");
            g.classList.add("selected");
            _selectedG = g;
            setSelected(n);
          }});

          const raw = n.raw || {};
          const residency = inferResidency(raw);
          const residencyClass = residency.replace(/_/g, "-");
          const stats = readyTokenStats(raw);

          const tt = document.createElementNS(SVG_NS, "title");
          tt.textContent =
            (n.label || "") +
            " | cache=" + residencyLabel(residency) +
            " | gpu_kv=" + String(stats.gpuReady) +
            " | cpu_kv=" + String(stats.cpuReady);
          g.appendChild(tt);

          const c = document.createElementNS(SVG_NS, "circle");
          c.setAttribute("class", "node-dot node-dot-" + residencyClass);
          c.setAttribute("r", String(NODE_R));
          g.appendChild(c);

          const rawSegLen = (n.raw || {}).segmentLen;
          if (rawSegLen !== null && rawSegLen !== undefined) {{
            const sl = document.createElementNS(SVG_NS, "text");
            sl.setAttribute("class", "node-seglen");
            sl.setAttribute("text-anchor", "end");
            sl.setAttribute("dy", "0.32em");
            sl.setAttribute("x", String(-(NODE_R + SEG_PAD)));
            sl.textContent = String(rawSegLen);
            g.appendChild(sl);
          }}

          // Per-agent hit counts (compact colored badges)
          const agentHits = raw.agentHits || {{}};
          const agentKeys = Object.keys(agentHits).sort();
          const showKeys = agentKeys.slice(0, MAX_AGENT_BADGES);
          const startY = -((showKeys.length - 1) * HIT_STEP) / 2;
          for (let i = 0; i < showKeys.length; i++) {{
            const a = showKeys[i];
            const v = agentHits[a];
            if (!v) continue;
            const col = _agentColors[a] || "#64748b";

            const gg = document.createElementNS(SVG_NS, "g");
            gg.setAttribute("class", "agent-hit");
            gg.setAttribute("transform", `translate(${HIT_X},${startY + i * HIT_STEP})`);

	            const bc = document.createElementNS(SVG_NS, "circle");
	            bc.setAttribute("r", String(HIT_DOT_R));
	            bc.setAttribute("fill", col);
	            gg.appendChild(bc);

	            const ti = document.createElementNS(SVG_NS, "text");
	            ti.setAttribute("class", "hit-initial");
	            ti.setAttribute("text-anchor", "middle");
	            ti.setAttribute("dy", "0.35em");
	            ti.textContent = String(a || "?").trim().slice(0, 1).toUpperCase();
	            gg.appendChild(ti);

	            const tc = document.createElementNS(SVG_NS, "text");
	            tc.setAttribute("class", "hit-count");
	            tc.setAttribute("x", String(HIT_COUNT_X));
	            tc.setAttribute("dy", "0.35em");
	            tc.setAttribute("fill", col);
	            tc.textContent = String(v > 999 ? "999+" : v);
	            gg.appendChild(tc);

            const btt = document.createElementNS(SVG_NS, "title");
            btt.textContent = `${a}: ${v} hits`;
            gg.appendChild(btt);

            g.appendChild(gg);
          }}

          if (showLabels) {{
            const tx = document.createElementNS(SVG_NS, "text");
            tx.setAttribute("class", "node-label");
            tx.setAttribute("dy", "0.32em");
            tx.setAttribute("x", String(NODE_R + 10));
            tx.textContent = (n.label || "").slice(0, 40);
            g.appendChild(tx);
          }}

          gNodes.appendChild(g);
        }}

        fitToContent(30);
      }} catch (e) {{
        selMeta.innerHTML = "<span class=\\"err\\">render_error</span>";
        selEl.textContent = String(e && e.stack ? e.stack : e);
      }}
    }}

    function refreshDpOptions() {{
      const workerUrl = workerSel.value;
      const w = (DATA.workers || {{}})[workerUrl];
      const count = (w && w.success && Array.isArray(w.dp_trees)) ? w.dp_trees.length : 0;
      dpSel.innerHTML = "";
      for (let i = 0; i < Math.max(1, count); i++) {{
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = String(i);
        dpSel.appendChild(opt);
      }}
    }}

    function init() {{
      const workers = keys(DATA.workers || {{}});
      workerSel.innerHTML = "";
      for (const w of workers) {{
        const opt = document.createElement("option");
        opt.value = w;
        opt.textContent = w;
        workerSel.appendChild(opt);
      }}
      if (workers.length) {{
        workerSel.value = workers[0];
      }}
      refreshDpOptions();
      renderTree();
    }}

    document.getElementById("apply").addEventListener("click", renderTree);
    document.getElementById("fit").addEventListener("click", () => fitToContent(30));
    document.getElementById("resetZoom").addEventListener("click", () => {{
      _scale = 1.0;
      _tx = 0.0;
      _ty = 0.0;
      applyTransform();
    }});
    workerSel.addEventListener("change", () => {{
      refreshDpOptions();
      renderTree();
    }});
    dpSel.addEventListener("change", renderTree);
    svg.addEventListener("click", () => setSelected(null));
    init();
  </script>
</body>
</html>
"""
        html_template_fixed = html_template.replace("{{", "{").replace("}}", "}")
        html_content = html_template_fixed.replace(
            "__SGLANG_PROMPT_TREE_TITLE__", title_html
        ).replace("__SGLANG_PROMPT_TREE_DATA__", payload_js)
        path.write_text(html_content, encoding="utf-8")
        return str(path)
