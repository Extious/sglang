---
dataset_info:
  features:
  - name: title
    dtype: string
  - name: system_prompt
    dtype: string
  - name: domain
    dtype: string
  splits:
  - name: train
    num_bytes: 104461
    num_examples: 100
  download_size: 48475
  dataset_size: 104461
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
