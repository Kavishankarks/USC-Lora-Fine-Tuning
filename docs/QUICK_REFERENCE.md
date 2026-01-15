# Quick Reference Guide - USC Course Recommendation Model

## 🚀 Quick Start

### Using Adapters (Recommended for Development)

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/Qwen2-0.5B-Instruct-4bit",
    adapter_path="../adapters"
)

prompt = """<|im_start|>system
You are a helpful USC course recommendation assistant.<|im_end|>
<|im_start|>user
What computer science courses should I take?<|im_end|>
<|im_start|>assistant
"""

response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### Using Fused Model (Standalone)
```python
from mlx_lm import load, generate

model, tokenizer = load("lora_fused_model")

# Same prompt format as above
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
```

---

## 📊 At a Glance Comparison

| Aspect | Adapters | Fused Model |
|--------|----------|-------------|
| **Size** | 5.61 MB | 942 MB |
| **Speed** | 98% | 100% |
| **Flexibility** | High | Low |
| **Storage** | Efficient | Standard |
| **Use Case** | Multi-task/Dev | Single-task/Prod |

---

## 📁 Project Structure

```
PDF-Finetuning-Model/
├── adapters/                      # ⭐ LoRA adapters (5.61 MB)
│   ├── adapters.safetensors       # Final adapter weights
│   └── adapter_config.json        # Configuration
├── lora_fused_model/              # 📦 Fused model (942 MB)
│   ├── model.safetensors          # Merged weights
│   ├── tokenizer.json             # Tokenizer
│   └── config.json                # Config
├── train.jsonl                    # Training data (9,309 examples)
├── valid.jsonl                    # Validation data (1,035 examples)
├── README.md                      # Full documentation
├── ADAPTERS_VS_FUSED_MODELS.md   # Technical deep dive
└── QUICK_REFERENCE.md             # This file
```

---

## 🎯 When to Use What

### Use Adapters ✅
- Experimenting with multiple versions
- Limited storage space
- Need Git-friendly versioning
- Serving multiple tasks
- Rapid iteration

### Use Fused Model ✅
- Production deployment (single task)
- Simpler code requirements
- Maximum inference speed
- Standalone distribution
- No base model dependency

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| **Initial Val Loss** | 3.878 |
| **Final Val Loss** | 0.919 |
| **Improvement** | 76.3% |
| **Training Examples** | 9,309 |
| **Validation Examples** | 1,035 |
| **Training Time** | ~2-3 minutes |
| **Peak Memory** | 1.637 GB |
| **Trainable Params** | 0.297% |

---

## 🔧 Common Commands

### Explore Dataset
```bash
python explore_dataset.py
```

### Prepare Training Data
```bash
python prepare_training_data.py
```

### Fine-tune Model (Extended Training)
```bash
python finetune_model.py  # Now trains for 5000 iterations
```

### Test Inference
```bash
python test_inference.py          # Test fused model
python test_with_adapters.py      # Test with adapters
python inference.py               # Interactive mode
```

### View Training Summary
```bash
python evaluation_summary.py
```

---

## 💡 Prompt Template

Always use this format for best results:

```python
prompt = f"""<|im_start|>system
You are a helpful USC course recommendation assistant. Provide accurate information about courses, prerequisites, schedules, and recommendations based on the USC Course Catalog.<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""
```

### Example Questions

✅ "What courses are available in computer science?"
✅ "Tell me about Machine Learning courses."
✅ "What are the prerequisites for deep learning?"
✅ "Recommend courses for someone interested in AI."
✅ "When are artificial intelligence courses offered?"

---

## 🔢 Key Statistics

```
Dataset:           USC Course Catalog
Total Courses:     5,571
Programs:          Multiple (CS, MATH, ENGL, ALI, etc.)
Training Examples: 10,344
Base Model:        Qwen2-0.5B-Instruct-4bit
Parameters:        500M (494M base + 1.47M LoRA)
Framework:         MLX (Apple Silicon optimized)
```

---

## 📦 Model Files

### Adapters Directory (5.61 MB)
```
adapters/
├── adapters.safetensors           # Final weights
├── 0000200_adapters.safetensors   # Checkpoint @ 200
├── 0000400_adapters.safetensors   # Checkpoint @ 400
├── 0000600_adapters.safetensors   # Checkpoint @ 600
├── 0000800_adapters.safetensors   # Checkpoint @ 800
├── 0001000_adapters.safetensors   # Checkpoint @ 1000
└── adapter_config.json            # Configuration
```

### Fused Model Directory (942 MB)
```
lora_fused_model/
├── model.safetensors              # Merged weights
├── tokenizer.json                 # Tokenizer
├── config.json                    # Model config
├── tokenizer_config.json          # Tokenizer config
└── chat_template.jinja            # Chat template
```

---

## 🎨 Example Use Cases

### 1. Course Discovery
```python
prompt = "What introductory courses can I take with no prerequisites?"
```

### 2. Prerequisite Check
```python
prompt = "What are the prerequisites for Advanced Machine Learning?"
```

### 3. Schedule Information
```python
prompt = "What days and times is Database Systems offered?"
```

### 4. Program Exploration
```python
prompt = "Tell me about courses in the Computer Science program."
```

### 5. Personalized Recommendations
```python
prompt = "I'm interested in artificial intelligence and data science. What courses should I take?"
```

---

## ⚡ Performance Tips

### Loading Speed
- **Adapters**: ~2.8s (base + adapter)
- **Fused**: ~2.3s (single load)
- **Difference**: 0.5s (negligible)

### Inference Speed
- **Adapters**: 98% of fused speed
- **Overhead**: 2-5% (acceptable for most cases)

### Memory Usage
- **Training**: 1.637 GB peak
- **Inference**: 1.8-2.0 GB
- **Efficient**: 4-bit quantization helps!

---

## 🐛 Troubleshooting

### Repetitive Outputs
**Problem**: Model generates repetitive text
**Solution**: Use adapters directly instead of fused model

### Out of Memory
**Problem**: Training crashes
**Solution**: Reduce batch size or use smaller model

### Slow Training
**Problem**: Training is taking too long
**Solution**: Ensure running on Apple Silicon with Metal

### Import Errors
**Problem**: Cannot import mlx_lm
**Solution**: Activate virtual environment: `source venv/bin/activate`

---

## 📚 File Descriptions

| File | Purpose | When to Run |
|------|---------|-------------|
| `explore_dataset.py` | Inspect USC catalog | Before training |
| `prepare_training_data.py` | Create training data | Before training |
| `finetune_model.py` | Train the model | Main training |
| `inference.py` | Interactive testing | After training |
| `test_inference.py` | Automated tests | After training |
| `test_with_adapters.py` | Test adapters | After training |
| `evaluation_summary.py` | View metrics | After training |

---

## 🔄 Workflow

```
1. Setup
   └─→ Create venv
       └─→ Install packages

2. Data Preparation
   └─→ Run explore_dataset.py
       └─→ Run prepare_training_data.py

3. Training
   └─→ Run finetune_model.py
       └─→ Monitor loss curves

4. Evaluation
   └─→ Run evaluation_summary.py
       └─→ Check metrics

5. Testing
   └─→ Run test_inference.py
       └─→ Try interactive mode
       └─→ Test with adapters

6. Deployment
   └─→ Choose adapters OR fused
       └─→ Integrate into application
```

---

## 💾 Storage Requirements

### Development
```
venv:           ~500 MB
Base model:     ~290 MB (cached)
Training data:  ~6.3 MB
Adapters:       ~5.6 MB
Total:          ~802 MB
```

### With Fused Model
```
Everything above + Fused: ~942 MB
Total: ~1.7 GB
```

---

## 🌟 Key Advantages

### This Implementation
✅ Efficient: Only 0.297% parameters trained
✅ Fast: 7-9 iterations/second on Apple Silicon
✅ Accurate: 76.3% validation loss reduction
✅ Small: 5.61 MB adapter size
✅ Flexible: Switch between adapters easily
✅ Scalable: Can train on larger datasets

### MLX Framework
✅ Native Apple Silicon support
✅ Unified memory architecture
✅ Fast training and inference
✅ Low memory footprint
✅ Python-first API

---

## 📖 Learn More

- **Full Documentation**: `README.md`
- **Technical Details**: `ADAPTERS_VS_FUSED_MODELS.md`
- **MLX Docs**: https://ml-explore.github.io/mlx/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

## ✨ Quick Tips

1. **Always activate venv**: `source venv/bin/activate`
2. **Use adapters for dev**: Faster iteration
3. **Monitor loss curves**: Should decrease steadily
4. **Save checkpoints**: Every 200 iterations
5. **Test early**: Don't wait for full training
6. **Document changes**: Keep track of experiments
7. **Version adapters**: Use descriptive names
8. **Backup regularly**: Save adapter checkpoints

---

**Last Updated**: January 2026
**Project**: USC Course Recommendation Model
**Framework**: MLX on Apple Silicon
