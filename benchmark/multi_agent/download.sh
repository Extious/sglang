#!/usr/bin/bash
# Download datasets from Hugging Face (MAST-Data, PatronusAI/TRAIL)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [mast|trail|agentcode|parallel|router|system_prompts|all]"
    echo "  mast           - mcemri/MAST-Data"
    echo "  trail          - PatronusAI/TRAIL"
    echo "  agentcode      - AlignmentLab-AI/agentcode"
    echo "  parallel       - DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset"
    echo "  router         - bhaiyahnsingh45/multiagent-router-finetuning"
    echo "  system_prompts - kimcomehome/system-prompts-multi-agent-systems"
    echo "  all            - download all (default)"
    exit 1
}

download_mast() {
    if [ -d "MAST-Data" ] && [ -n "$(ls -A MAST-Data 2>/dev/null)" ]; then
        echo "MAST-Data already exists, skipping."
        return
    fi
    echo "Downloading mcemri/MAST-Data from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"mcemri/MAST-Data\", repo_type=\"dataset\", local_dir=\"MAST-Data\")
"
    echo "Done. MAST-Data is in $SCRIPT_DIR/MAST-Data"
}

download_trail() {
    if [ -d "TRAIL" ] && [ -n "$(ls -A TRAIL 2>/dev/null)" ]; then
        echo "TRAIL already exists, skipping."
        return
    fi
    echo "Downloading PatronusAI/TRAIL from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"PatronusAI/TRAIL\", repo_type=\"dataset\", local_dir=\"TRAIL\")
"
    echo "Done. TRAIL is in $SCRIPT_DIR/TRAIL"
}

download_agentcode() {
    if [ -d "agentcode" ] && [ -n "$(ls -A agentcode 2>/dev/null)" ]; then
        echo "agentcode already exists, skipping."
        return
    fi
    echo "Downloading AlignmentLab-AI/agentcode from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"AlignmentLab-AI/agentcode\", repo_type=\"dataset\", local_dir=\"agentcode\")
"
    echo "Done. agentcode is in $SCRIPT_DIR/agentcode"
}

download_parallel() {
    if [ -d "Multi-Agents-Parallel-Orchestration-Dataset" ] && [ -n "$(ls -A Multi-Agents-Parallel-Orchestration-Dataset 2>/dev/null)" ]; then
        echo "Multi-Agents-Parallel-Orchestration-Dataset already exists, skipping."
        return
    fi
    echo "Downloading DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset\", repo_type=\"dataset\", local_dir=\"Multi-Agents-Parallel-Orchestration-Dataset\")
"
    echo "Done. Multi-Agents-Parallel-Orchestration-Dataset is in $SCRIPT_DIR/Multi-Agents-Parallel-Orchestration-Dataset"
}

download_router() {
    if [ -d "multiagent-router-finetuning" ] && [ -n "$(ls -A multiagent-router-finetuning 2>/dev/null)" ]; then
        echo "multiagent-router-finetuning already exists, skipping."
        return
    fi
    echo "Downloading bhaiyahnsingh45/multiagent-router-finetuning from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"bhaiyahnsingh45/multiagent-router-finetuning\", repo_type=\"dataset\", local_dir=\"multiagent-router-finetuning\")
"
    echo "Done. multiagent-router-finetuning is in $SCRIPT_DIR/multiagent-router-finetuning"
}

download_system_prompts() {
    if [ -d "system-prompts-multi-agent-systems" ] && [ -n "$(ls -A system-prompts-multi-agent-systems 2>/dev/null)" ]; then
        echo "system-prompts-multi-agent-systems already exists, skipping."
        return
    fi
    echo "Downloading kimcomehome/system-prompts-multi-agent-systems from Hugging Face..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"kimcomehome/system-prompts-multi-agent-systems\", repo_type=\"dataset\", local_dir=\"system-prompts-multi-agent-systems\")
"
    echo "Done. system-prompts-multi-agent-systems is in $SCRIPT_DIR/system-prompts-multi-agent-systems"
}

case "${1:-all}" in
    mast)
        download_mast
        ;;
    trail)
        download_trail
        ;;
    agentcode)
        download_agentcode
        ;;
    parallel)
        download_parallel
        ;;
    router)
        download_router
        ;;
    system_prompts)
        download_system_prompts
        ;;
    all)
        download_mast
        download_trail
        download_agentcode
        download_parallel
        download_router
        download_system_prompts
        ;;
    *)
        usage
        ;;
esac
