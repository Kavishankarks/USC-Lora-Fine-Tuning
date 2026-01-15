"""
Fine-tune a small language model on USC Course Catalog using MLX
Uses Qwen2-0.5B-Instruct for efficient training on Apple Silicon
"""
import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f" Error: {description} failed")
        sys.exit(1)
    else:
        print(f"✓ {description} completed successfully")

    return result

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         USC Course Recommendation Model Fine-tuning               ║
    ║              Using MLX on Apple Silicon                           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    MODEL_NAME = "mlx-community/Qwen2-0.5B-Instruct-4bit"  # Small 4-bit quantized model
    OUTPUT_DIR = "lora_fused_model"
    ADAPTER_DIR = "adapters"

    print(f"\n📋 Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Training data: train.jsonl ({os.path.getsize('train.jsonl') / (1024*1024):.2f} MB)")
    print(f"   Validation data: valid.jsonl ({os.path.getsize('valid.jsonl') / (1024*1024):.2f} MB)")
    print(f"   Output directory: {OUTPUT_DIR}")

    # Check if data files exist
    if not os.path.exists('train.jsonl') or not os.path.exists('valid.jsonl'):
        print("❌ Error: Training data files not found. Run prepare_training_data.py first.")
        sys.exit(1)

    print("\n" + "="*80)
    print("STEP 1: Fine-tuning with LoRA")
    print("="*80)
    print("""
    This will train the model using Low-Rank Adaptation (LoRA):
    - Efficient: Only trains a small number of parameters
    - Fast: Optimized for Apple Silicon with MLX
    - Effective: Good performance with minimal resource usage

    Training parameters:
    - Learning rate: 1e-5
    - Batch size: 2
    - Iterations: 1000
    - LoRA rank: 8
    - LoRA alpha: 16
    - Validation every 100 steps
    """)

    # Fine-tune with LoRA
    finetune_cmd = [
        "mlx_lm.lora",
        "--model", MODEL_NAME,
        "--train",
        "--data", ".",  # Use current directory with train.jsonl and valid.jsonl
        "--iters", "5000",
        "--steps-per-eval", "100",
        "--learning-rate", "1e-5",
        "--batch-size", "2",
        "--num-layers", "8",  # Number of layers to apply LoRA to
        "--adapter-path", ADAPTER_DIR,
        "--save-every", "200"
    ]

    run_command(finetune_cmd, "LoRA Fine-tuning")

    print("\n" + "="*80)
    print("STEP 2: Fusing LoRA adapters with base model")
    print("="*80)
    print("""
    Merging the trained LoRA adapters back into the base model
    for easier deployment and inference.
    """)

    # Fuse the LoRA adapters
    fuse_cmd = [
        "mlx_lm.fuse",
        "--model", MODEL_NAME,
        "--adapter-path", ADAPTER_DIR,
        "--save-path", OUTPUT_DIR,
        "--dequantize"  # Convert back to full precision
    ]

    run_command(fuse_cmd, "Model Fusion")

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"""
    Your fine-tuned model has been saved to: {OUTPUT_DIR}/

    Model files:
    - {OUTPUT_DIR}/weights.safetensors (or .npz)
    - {OUTPUT_DIR}/config.json
    - {OUTPUT_DIR}/tokenizer.json

    You can now use this model for inference!
    """)

if __name__ == "__main__":
    main()
