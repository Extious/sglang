---
language:
- en
license: apache-2.0
task_categories:
- text-classification
- question-answering
tags:
- function-calling
- multi-agent
- routing
- customer-support
- synthetic
pretty_name: Multi-Agent Router Fine-tuning Dataset
size_categories:
- n<1K
---

# Multi-Agent Router Fine-tuning Dataset

## Dataset Description

This dataset is designed for fine-tuning language models to perform intelligent routing in multi-agent customer support systems. The model learns to classify user queries and route them to the appropriate specialized agent with relevant parameters.

### Supported Tasks

- **Function Calling**: Route queries to appropriate agent functions
- **Intent Classification**: Identify the type of support needed
- **Parameter Extraction**: Extract relevant parameters from queries

## Dataset Structure

### Data Instances

Each instance contains:
- `query`: The user's question or request
- `agent_name`: The target agent to handle the query (technical_support_agent, billing_agent, or product_info_agent)
- `agent_arguments`: JSON object with parameters for the agent
- `system_message`: System prompt for the model

Example:
```json
{
  "query": "My app keeps crashing when I try to upload photos larger than 5MB",
  "agent_name": "technical_support_agent",
  "agent_arguments": {
    "issue_type": "crash",
    "priority": "high"
  },
  "system_message": "You are an intelligent routing agent..."
}
```

### Data Fields

- **query** (string): User's question or request
- **agent_name** (string): Target agent name
  - `technical_support_agent`: Technical issues, bugs, integration
  - `billing_agent`: Payments, subscriptions, invoices
  - `product_info_agent`: Features, plans, integrations
- **agent_arguments** (dict): Agent-specific parameters
  - Technical Support: `issue_type`, `priority`
  - Billing: `request_type`, `urgency`
  - Product Info: `query_type`, `category`
- **system_message** (string): System prompt

### Data Splits

| Split | Examples |
|-------|----------|
| train | 92 |
| test  | 23 |

## Dataset Creation

### Curation Rationale

This dataset was created to train routing models for multi-agent customer support systems. Real-world customer support requires:
- Accurate classification of query intent
- Extraction of priority/urgency levels
- Routing to specialized agents

### Source Data

#### Initial Data Collection and Normalization

The dataset consists of synthetic but realistic customer support queries covering:
- **Technical Support** (20 samples): App crashes, API errors, authentication issues, performance problems
- **Billing** (20 samples): Refunds, payment failures, subscription management, pricing inquiries
- **Product Information** (20 samples): Feature comparisons, integrations, compliance questions, platform capabilities
- **Edge Cases** (5 samples): Ambiguous queries to test robustness

Queries were designed to be:
- Realistic and varied
- Include specific details (error codes, product names, numeric values)
- Cover different priority/urgency levels
- Include edge cases and ambiguous requests

## Usage

### Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("bhaiyahnsingh45/multiagent-router-finetuning")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# Example usage
for example in train_data:
    print(f"Query: {example['query']}")
    print(f"Agent: {example['agent_name']}")
    print(f"Arguments: {example['agent_arguments']}")
```

### Fine-tuning Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

# Define your agent functions
def technical_support_agent(issue_type: str, priority: str) -> str:
    """Routes technical issues to specialized support team."""
    pass

def billing_agent(request_type: str, urgency: str) -> str:
    """Routes billing and payment queries."""
    pass

def product_info_agent(query_type: str, category: str) -> str:
    """Routes product information queries."""
    pass

# Get tool schemas
tools = [
    get_json_schema(technical_support_agent),
    get_json_schema(billing_agent),
    get_json_schema(product_info_agent)
]

# Format for training (example with FunctionGemma)
def create_conversation(sample):
    return {
        "messages": [
            {"role": "developer", "content": sample["system_message"]},
            {"role": "user", "content": sample["query"]},
            {"role": "assistant", "tool_calls": [{
                "type": "function",
                "function": {
                    "name": sample["agent_name"],
                    "arguments": sample["agent_arguments"]
                }
            }]}
        ],
        "tools": tools
    }

# Apply to dataset
dataset = dataset.map(create_conversation)
```

## Dataset Statistics

### Query Length Distribution

- **Min tokens**: ~5
- **Max tokens**: ~25
- **Average tokens**: ~12

### Agent Distribution

| Agent | Count | Percentage |
|-------|-------|------------|
| Technical Support | ~20 | ~33% |
| Billing | ~20 | ~33% |
| Product Info | ~20 | ~33% |
| Edge Cases | ~5 | ~8% |

### Parameter Distribution

**Technical Support - Priority Levels:**
- High: ~50%
- Medium: ~40%
- Low: ~10%

**Billing - Urgency Levels:**
- High: ~30%
- Medium: ~40%
- Low: ~30%

## Evaluation

Expected model performance after fine-tuning:
- **Baseline accuracy**: 10-30% (pre-trained model)
- **Target accuracy**: 70-95% (fine-tuned model)
- **Training time**: ~5-10 minutes on T4 GPU

## Considerations for Using the Data

### Social Impact

This dataset helps improve automated customer support systems by:
- Reducing wait times through accurate routing
- Improving first-contact resolution rates
- Enabling 24/7 support capabilities

### Limitations

- Synthetic data may not cover all real-world variations
- English language only
- Limited to three agent types
- May require domain adaptation for specific industries

## Additional Information

### Dataset Curators

Created for fine-tuning FunctionGemma and similar function-calling models.

### Licensing Information

Apache 2.0 License

### Citation Information

```bibtex
@dataset{multiagent_router_finetuning,
  author = {Your Name},
  title = {Multi-Agent Router Fine-tuning Dataset},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/bhaiyahnsingh45/multiagent-router-finetuning}
}
```

### Contributions

Contributions to expand this dataset are welcome! Areas for improvement:
- Additional languages
- More agent types (sales, feedback, onboarding)
- Domain-specific variations (healthcare, finance, e-commerce)
- Real user query examples (with proper anonymization)

### Contact

For questions or feedback, please open an issue on the dataset repository.

---

**Note**: This is a synthetic dataset created for training purposes. For production use, consider augmenting with real anonymized customer queries from your specific domain.
