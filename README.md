# USC Course Recommendation Model - Fine-tuning Project

A fine-tuned Small Language Model (SLM) for USC course recommendations, trained on Apple Silicon using the MLX framework.

## Project Overview

This project demonstrates fine-tuning a 500M parameter language model (Qwen2-0.5B-Instruct) on the USC Course Catalog dataset to create an intelligent course recommendation assistant.

## Key Features

- **Dataset**: USC Course Catalog (5,571 courses from Hugging Face)
- **Model**: Qwen2-0.5B-Instruct (500M parameters, 4-bit quantized)
- **Framework**: MLX (optimized for Apple Silicon)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Examples**: 10,344 instruction-response pairs

## Training Results

### Performance Metrics

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Validation Loss | 3.878 | 0.919 | 76.3% reduction |
| Training Loss | 2.624 | 0.867 | 67.0% reduction |
| Peak Memory | 0.955 GB | 1.637 GB | Efficient |

### Training Configuration

- **Iterations**: 1,000
- **Batch Size**: 2
- **Learning Rate**: 1e-5
- **LoRA Layers**: 8
- **Tokens Processed**: 267,433
- **Training Speed**: 7-9 iterations/second

## Directory Structure

```
.
├── adapters/                      # LoRA adapter weights
│   ├── adapters.safetensors       # Final adapter (5.61 MB)
│   └── adapter_config.json        # Adapter configuration
├── lora_fused_model/              # Fused model ready for deployment
│   ├── model.safetensors          # Model weights (942 MB)
│   ├── tokenizer.json             # Tokenizer (10.89 MB)
│   └── config.json                # Model configuration
├── train.jsonl                    # Training dataset (5.68 MB)
├── valid.jsonl                    # Validation dataset (0.63 MB)
├── explore_dataset.py             # Dataset exploration
├── prepare_training_data.py       # Data preprocessing
├── finetune_model.py              # Fine-tuning orchestration
├── inference.py                   # Interactive inference
├── test_inference.py              # Automated testing
├── test_with_adapters.py          # Adapter-based testing
└── evaluation_summary.py          # Comprehensive evaluation
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install mlx mlx-lm transformers datasets pandas numpy tqdm huggingface_hub
```

### 3. Explore Dataset

```bash
python explore_dataset.py
```

## Training Pipeline

### Step 1: Prepare Training Data

```bash
python prepare_training_data.py
```

This script:
- Loads USC Course Catalog from Hugging Face
- Generates instruction-response pairs
- Creates train/validation splits (90/10)
- Formats data for MLX training

### Step 2: Fine-tune Model

```bash
python finetune_model.py
```

This script:
- Downloads Qwen2-0.5B-Instruct-4bit
- Trains with LoRA adapters
- Saves checkpoints every 200 iterations
- Fuses adapters into final model

### Step 3: Evaluate Model

```bash
python evaluation_summary.py
```

Generates comprehensive training metrics and model statistics.

## Usage

### Method 1: Using LoRA Adapters (Recommended)

```python
from mlx_lm import load, generate

# Load model with adapters
model, tokenizer = load(
    "mlx-community/Qwen2-0.5B-Instruct-4bit",
    adapter_path="adapters"
)

# Format prompt
prompt = """<|im_start|>system
You are a helpful USC course recommendation assistant.<|im_end|>
<|im_start|>user
What courses are available in computer science?<|im_end|>
<|im_start|>assistant
"""

# Generate response
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### Method 2: Using Fused Model

```python
from mlx_lm import load, generate

# Load fused model
model, tokenizer = load("lora_fused_model")

# Use same prompt format as above
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### Method 3: Interactive Testing

```bash
# Test with fused model
python test_inference.py

# Test with adapters
python test_with_adapters.py

# Interactive mode
python inference.py
```

## Model Capabilities

The fine-tuned model can:

✅ Answer questions about USC courses
✅ Provide course recommendations based on interests
✅ Explain course prerequisites and requirements
✅ Share course schedule information (days, times)
✅ Describe course content and learning objectives
✅ Identify instructors and course types

## Dataset Information

**Source**: USC/USC-Course-Catalog (Hugging Face)

**Features**:
- Program Code
- Program URL
- Class ID
- Class Name
- Catalogue Description
- Units
- Prerequisites
- Restrictions
- Class Section
- Time
- Days
- Class Type
- Instructor

**Coverage**: 5,571 courses across multiple departments including:
- Computer Science (CS)
- Academic Language Institute (ALI)
- English (ENGL)
- Mathematics (MATH)
- And many more...

## Technical Details

### Why MLX?

MLX is Apple's machine learning framework optimized for Apple Silicon:
- Efficient memory usage
- Fast training and inference
- Native support for M1/M2/M3 chips
- Unified memory architecture

### Why LoRA?

Low-Rank Adaptation enables efficient fine-tuning:
- Only trains 0.297% of parameters (1.466M / 494.033M)
- 5.61 MB adapter size vs 942 MB full model
- Faster training convergence
- Easy to switch between different adaptations

### Model Architecture

- **Base**: Qwen2-0.5B-Instruct
- **Parameters**: 500M (4-bit quantized)
- **Context Length**: Supports long context
- **Tokenizer**: BPE-based tokenizer
- **Format**: Instruction-following format with system/user/assistant roles

## Performance Considerations

### Memory Usage

- Training: Peak 1.637 GB
- Inference with adapters: ~1-2 GB
- Inference with fused model: ~2-3 GB

### Speed

- Training: 7-9 iterations/second on Apple Silicon
- Inference: Fast response generation
- Tokens/second: ~2000 during training

## Limitations

1. **Small Model**: 500M parameters means limited general knowledge
2. **Domain-Specific**: Trained specifically on USC courses
3. **Static Data**: Based on specific catalog snapshot
4. **Generation Quality**: May require prompt engineering for optimal results

## Future Improvements

- [ ] Train for more iterations (2000-5000)
- [ ] Experiment with larger models (1B-3B parameters)
- [ ] Add retrieval-augmented generation (RAG)
- [ ] Implement course similarity search
- [ ] Fine-tune on student reviews and outcomes
- [ ] Add multi-turn conversation support

## Troubleshooting

### Issue: Model generates repetitive text

**Solution**: Try using adapters directly instead of fused model, or adjust generation parameters (temperature, top_p).

### Issue: Out of memory during training

**Solution**: Reduce batch size or use smaller model variant.

### Issue: Slow training speed

**Solution**: Ensure you're running on Apple Silicon with Metal acceleration enabled.

## References

- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [Qwen2 Model](https://huggingface.co/Qwen)
- [USC Course Catalog Dataset](https://huggingface.co/datasets/USC/USC-Course-Catalog)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

This project is for educational purposes. Please respect the original licenses of:
- MLX Framework (MIT License)
- Qwen2 Model (Apache 2.0)
- USC Course Catalog Dataset (Check dataset license)

## Acknowledgments

- **Anthropic** for Claude Code
- **Apple** for MLX framework
- **Alibaba** for Qwen2 model
- **USC** for course catalog dataset
- **Hugging Face** for dataset hosting

---

**Created**: January 2026
**Framework**: MLX on Apple Silicon
**Model**: Qwen2-0.5B-Instruct (Fine-tuned)
**Dataset**: USC Course Catalog (5,571 courses)
