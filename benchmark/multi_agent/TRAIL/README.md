---
license: mit
language:
- en
pretty_name: TRAIL
extra_gated_prompt: To avoid contamination and data leakage, you agree to not reshare
  this dataset outside of a gated or private repository on the HF hub.
extra_gated_fields:
  I agree to not reshare the submissions set according to the above conditions: checkbox
dataset_info:
  features:
  - name: trace
    dtype: string
  - name: labels
    dtype: string
  splits:
  - name: gaia
    num_bytes: 111823308
    num_examples: 117
  - name: swe_bench
    num_bytes: 74599554
    num_examples: 31
  download_size: 54643931
  dataset_size: 186422862
configs:
- config_name: default
  data_files:
  - split: gaia
    path: data/gaia-*
  - split: swe_bench
    path: data/swe_bench-*
---
# Trace Reasoning and Agentic Issue Localization (TRAIL)


<img src="https://i.imgur.com/BDk2QcM.jpeg" width="30%" height="30%" alt="TRAIL"/>

TRAIL is a benchmark dataset of 148 annotated AI agent execution traces containing 841 errors across reasoning, execution, and planning categories. Created from real-world software engineering and information retrieval tasks, it challenges even state-of-the-art LLMs, with the best model achieving only 11% accuracy, highlighting the difficulty of trace debugging for complex agent workflows.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
TRAIL (Trace Reasoning and Agentic Issue Localization) is a new benchmark dataset designed to evaluate how well large language models can debug and identify errors in complex AI agent workflows. 
The dataset contains 148 meticulously annotated agent execution traces with 841 unique errors across a taxonomy of error categories spanning reasoning errors (like hallucinations), system execution errors (like API issues), and planning/coordination errors. 
TRAIL is constructed from real-world applications using the GAIA and SWE-Bench datasets, featuring both single and multi-agent systems tackling tasks in software engineering and information retrieval. 
The paper demonstrates that even state-of-the-art LLMs perform poorly on TRAIL, with the best model (Gemini-2.5-Pro) achieving only 11% joint accuracy. 
The benchmark is particularly challenging because it requires processing extremely long contexts that often exceed model context windows and demands significant output generation, making it valuable for improving LLMs' ability to evaluate complex agentic systems.

- **Curated by:** Patronus AI
- **Language(s) (NLP):** English
- **License:** MIT License

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/patronus-ai/trail-benchmark
- **Paper:** https://arxiv.org/abs/2505.08638

### Out-of-Scope Use
You must not use this dataset for training systems (AI models or otherwise) that are intended to automate human evaluation. This dataset is only meant for evaluation and benchmarking of such systems.

## Model Performance on TRAIL
<img src="https://i.imgur.com/QeHGLAj.png" width="50%" height="50%" alt="TRAIL Results"/>


## Dataset Structure
<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
The dataset consists of 148 traces (118 from GAIA and 30 from SWE-Bench) totaling 1,987 OpenTelemetry spans, of which 575 exhibit at least one error. The dataset is structured with trace-level annotations showing span IDs, error category types, supporting evidence, descriptions, and impact levels (Low/Medium/High) for each identified error. The dataset is split between the GAIA benchmark (open-world search tasks) and SWE-Bench (software engineering bug fixing), ensuring ecological validity across different agent applications.

## Dataset Creation

### Curation Rationale
<!-- Motivation for the creation of this dataset. -->
The dataset was created to address the growing need for robust and dynamic evaluation methods for agentic workflow traces.
As agentic systems become increasingly complex and widely adopted across domains, existing evaluation methods that rely on manual, domain-specific analysis of traces do not scale well. 
TRAIL provides a structured way to evaluate traces with a comprehensive taxonomy, enabling more systematic debugging and error analysis of complex agent behavior.

### Source Data
<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

### Data Collection and Processing
<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->
The dataset was created using text-only data instances from GAIA (for open-world search tasks) and SWE-Bench Lite (for software engineering bug fixing tasks). 
For GAIA traces, we used the Hugging Face OpenDeepResearch agent with o3-mini-2025-01-31 as the backbone model. 
For SWE-Bench, we used a CodeAct agent with claude-3-7-sonnet-20250219 as the backbone model, with added instructional constraints to organically introduce errors. 
All traces were collected using OpenTelemetry, specifically the OpenInference standard, ensuring compatibility with real-world tracing and observability software.

### Who are the source data producers?
<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->
The source data was produced by AI agent systems based on OpenAI's o3-mini and Anthropic's Claude models, executing tasks from the GAIA and SWE-Bench datasets. The traces capture the execution flows of these agents attempting to solve information retrieval and software engineering tasks.

### Annotations
<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

### Annotation process
<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->
Four expert annotators with backgrounds in software engineering and log debugging annotated the agent traces. 
Due to the lengthy traces (often exceeding maximum LLM context lengths), four independent rounds of verification were performed by ML researchers to ensure high quality. 
Annotators iterated over each LLM and tool span individually and in context, marking span ID, error category, evidence, description, and impact level. 
They also rated overall traces based on instruction adherence, plan optimality, security, and reliability. 
Interannotator agreement was high, with only 5.63% of spans modified in SWE-Bench and 5.31% in GAIA during review.

### Who are the annotators?
<!-- This section describes the people or systems who created the annotations. -->
The annotations were created by four expert annotators with backgrounds in software engineering and log debugging, selected based on their age (18+) and expertise in computer science. 
The annotations were further verified by four industry ML researchers to ensure high quality.

### Personal and Sensitive Information
<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->
The dataset does not contain personal identifiable information (PII) or sensitive content. 
The traces were manually verified before being forwarded to annotators to ensure no explicit or biased content was included.

### Bias, Risks, and Limitations
<!-- This section is meant to convey both technical and sociotechnical limitations. -->
The TRAIL dataset has the following limitations:
- It is primarily focused on text-only inputs and outputs.
- There is an imbalance in error categories, with Output Generation errors (particularly Formatting Errors and Instruction Non-compliance) accounting for nearly 42% of all errors.
- 
## Citation
<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@misc{deshpande2025trail,
  title={TRAIL: Trace Reasoning and Agentic Issue Localization},
  author={Darshan Deshpande and Varun Gangal and Hersh Mehta and Jitin Krishnan and Anand Kannappan and Rebecca Qian},
  year={2025},
  eprint={2505.08638},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2505.08638}
}
```

**APA:**
```
Deshpande, D., Gangal, V., Mehta, H., Krishnan, J., Kannappan, A., & Qian, R. (2025). TRAIL: Trace Reasoning and Agentic Issue Localization. arXiv. https://arxiv.org/abs/2505.08638
```

## Dataset Card Authors
Darshan Deshpande

## Dataset Card Contact
darshan@patronus.ai