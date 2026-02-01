#!/usr/bin/bash
# Download datasets from Hugging Face (MAST-Data, PatronusAI/TRAIL)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [mast|trail|all]"
    echo "  mast  - mcemri/MAST-Data"
    echo "  trail - PatronusAI/TRAIL"
    echo "  all   - download both (default)"
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

case "${1:-all}" in
    mast)
        download_mast
        ;;
    trail)
        download_trail
        ;;
    all)
        download_mast
        download_trail
        ;;
    *)
        usage
        ;;
esac
