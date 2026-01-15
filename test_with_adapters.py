"""
Test inference with LoRA adapters directly
"""
from mlx_lm import load, generate
import sys

def format_prompt(question):
    """Format user question into the instruction template"""
    return f"""<|im_start|>system
You are a helpful USC course recommendation assistant. Provide accurate information about courses, prerequisites, schedules, and recommendations based on the USC Course Catalog.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║      USC Course Recommendation Model - Testing with Adapters      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Load model with adapters
    base_model = "mlx-community/Qwen2-0.5B-Instruct-4bit"
    adapter_path = "adapters"

    print(f"Loading model: {base_model}")
    print(f"With adapters: {adapter_path}")

    try:
        model, tokenizer = load(base_model, adapter_path=adapter_path)
        print("✓ Model with adapters loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Test questions about real USC courses
    test_questions = [
        "Tell me about the Academic and Professional Speaking Skills II course.",
        "What ALI courses are available?",
        "Tell me about Literary Genres and Film course.",
        "What are some beginner courses I can take?",
    ]

    print("="*80)
    print("RUNNING TEST QUERIES")
    print("="*80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: {question}")
        print(f"{'─'*80}")

        prompt = format_prompt(question)

        # Generate response
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=250,
                verbose=False
            )

            # Extract only the assistant's response (remove prompt)
            answer = response[len(prompt):]

            # Stop at the end token
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0]

            print(f"\n🤖 Answer:\n{answer.strip()}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")

    print("\n" + "="*80)
    print("✅ INFERENCE TEST COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
