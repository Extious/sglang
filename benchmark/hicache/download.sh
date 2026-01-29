#!/usr/bin/bash

# The usage function
usage() {
    echo "Usage: $0 {sharegpt|ultragpt|loogle|nextqa|all}"
    exit 1
}

# The download function
download() {
    case "$1" in
        sharegpt)
            echo $1
            wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
            ;;
        ultragpt)
            echo $1
            # Download UltraChat dataset from Hugging Face
            huggingface-cli download stingning/ultrachat --repo-type dataset --local-dir ultrachat
            ;;
        loogle)
            echo $1
            git lfs install
            git clone git@hf.co:datasets/bigainlco/LooGLE
            unzip LooGLE/data.zip
            ;;
        nextqa)
            echo $1
            git lfs install
            git clone https://huggingface.co/datasets/lmms-lab/NExTQA
            unzip NExTQA/videos.zip
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Arg check
if [ "$#" -ne 1 ]; then
    usage
fi

# Invoke

case "$1" in
    sharegpt|ultragpt|loogle|nextqa)
        download "$1"
        ;;
    all)
        download sharegpt
        download ultragpt
        download loogle
        download nextqa
        ;;
    *)
        usage
        ;;
esac
