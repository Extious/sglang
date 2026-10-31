"""AegisKV: agent-aware KV checkpointing and takeover for multi-worker serving.

The worker side lives here: it copies running requests' KV into backup arenas
owned by peer workers and imports a committed backup into the radix cache, so
a resubmitted request resumes with a prefix hit. Placement policy runs in the
external master (`benchmark/AegisKV/harness`).
"""
