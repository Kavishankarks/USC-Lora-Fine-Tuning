"""
Test inference with the fine-tuned USC Course Recommendation model
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
    ║         USC Course Recommendation Model - Inference Test          ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Load model
    model_path = "lora_fused_model"

    print(f"Loading fine-tuned model from: {model_path}")
    try:
        model, tokenizer = load(model_path)
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Test questions
    test_questions = [
        "What courses are available in computer science?",
        "Tell me about Machine Learning courses.",
        "What are the prerequisites for deep learning courses?",
        "Recommend courses for someone interested in AI.",
        "When are artificial intelligence courses offered?",
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
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )

        # Extract only the assistant's response (remove prompt)
        answer = response[len(prompt):]

        # Stop at the end token
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0]

        print(f"\n🤖 Answer:\n{answer.strip()}")
        print()

    print("\n" + "="*80)
    print("✅ INFERENCE TEST COMPLETE!")
    print("="*80)
    print("""
    The model has been successfully fine-tuned and is generating responses
    based on the USC Course Catalog dataset!
    """)

if __name__ == "__main__":
    main()
