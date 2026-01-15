"""
Prepare USC Course Catalog dataset for fine-tuning
Converts raw course data into instruction-response format for course recommendations
"""
from datasets import load_dataset
import json
import random
from tqdm import tqdm

# Load the dataset
print("Loading USC Course Catalog dataset...")
ds = load_dataset("USC/USC-Course-Catalog")
courses = ds['train']

print(f"Total courses: {len(courses)}")

# Create instruction-response pairs
training_data = []

# Template variations for instruction diversity
question_templates = [
    "What courses are available in {program}?",
    "Tell me about {class_name} course.",
    "What are the prerequisites for {class_name}?",
    "Recommend courses for someone interested in {program}.",
    "What {program} courses should I take?",
    "Give me information about class {class_id}.",
    "What time is {class_name} offered?",
    "Who teaches {class_name}?",
    "I want to study {program}. What courses do you recommend?",
    "What are the requirements for {class_name}?"
]

print("\nGenerating training examples...")

# Create diverse training examples
for idx in tqdm(range(len(courses))):
    course = courses[idx]

    # Skip if essential fields are missing
    if not course['Class Name'] or course['Class Name'].strip() == '':
        continue

    # Clean the data
    program = course['Program Code'] if course['Program Code'] else 'N/A'
    class_id = course['Class ID'].strip() if course['Class ID'] else 'N/A'
    class_name = course['Class Name'].strip() if course['Class Name'] else 'N/A'
    description = course['Catalogue'].strip() if course['Catalogue'] else 'No description available'
    units = course['Units'] if course['Units'] else 'N/A'
    prereqs = course['Prerequisites'] if course['Prerequisites'] and course['Prerequisites'] != 'Not available' else 'None'
    restrictions = course['Restrictions'] if course['Restrictions'] and course['Restrictions'] != 'Not available' else 'None'
    time = course['Time'] if course['Time'] else 'N/A'
    days = course['Days'] if course['Days'] else 'N/A'
    class_type = course['Class Type'] if course['Class Type'] else 'N/A'
    instructor = course['Instructor'] if course['Instructor'] and course['Instructor'] != 'None' else 'TBA'

    # Generate multiple question types per course

    # Type 1: General program inquiry
    if random.random() > 0.7:  # 30% of examples
        question = f"What courses are available in {program}?"
        answer = f"I recommend {class_name} ({class_id}). {description} This course is {units}, offered {days} at {time}."

        training_data.append({
            "instruction": question,
            "response": answer,
            "metadata": {"type": "program_inquiry", "program": program}
        })

    # Type 2: Specific course details
    question = f"Tell me about {class_name}."
    answer = f"{class_name} ({class_id}) is a {program} course. {description}\n\nDetails:\n- Units: {units}\n- Prerequisites: {prereqs}\n- Restrictions: {restrictions}\n- Schedule: {days} at {time}\n- Type: {class_type}\n- Instructor: {instructor}"

    training_data.append({
        "instruction": question,
        "response": answer,
        "metadata": {"type": "course_details", "class_id": class_id}
    })

    # Type 3: Prerequisites inquiry
    if prereqs != 'None' and random.random() > 0.5:
        question = f"What are the prerequisites for {class_name}?"
        answer = f"The prerequisites for {class_name} ({class_id}) are: {prereqs}"

        training_data.append({
            "instruction": question,
            "response": answer,
            "metadata": {"type": "prerequisites", "class_id": class_id}
        })

    # Type 4: Schedule inquiry
    if random.random() > 0.6:
        question = f"When is {class_name} offered?"
        answer = f"{class_name} ({class_id}) is offered {days} at {time}. The class type is {class_type} and it's taught by {instructor}."

        training_data.append({
            "instruction": question,
            "response": answer,
            "metadata": {"type": "schedule", "class_id": class_id}
        })

print(f"\nGenerated {len(training_data)} training examples")

# Shuffle and split
random.shuffle(training_data)
split_idx = int(len(training_data) * 0.9)  # 90% train, 10% validation
train_data = training_data[:split_idx]
val_data = training_data[split_idx:]

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")

# Convert to instruction format for MLX
def format_for_mlx(examples):
    formatted = []
    for ex in examples:
        # Format for instruction following (Qwen2 format)
        text = f"<|im_start|>system\nYou are a helpful USC course recommendation assistant. Provide accurate information about courses, prerequisites, schedules, and recommendations based on the USC Course Catalog.<|im_end|>\n<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
        formatted.append({"text": text})
    return formatted

train_formatted = format_for_mlx(train_data)
val_formatted = format_for_mlx(val_data)

# Save to JSONL format (MLX-LM compatible)
print("\nSaving formatted data...")
with open('train.jsonl', 'w') as f:
    for item in train_formatted:
        f.write(json.dumps(item) + '\n')

with open('valid.jsonl', 'w') as f:
    for item in val_formatted:
        f.write(json.dumps(item) + '\n')

print("✓ Training data saved to train.jsonl")
print("✓ Validation data saved to valid.jsonl")

# Show sample
print("\n" + "="*80)
print("SAMPLE TRAINING EXAMPLE")
print("="*80)
print(train_formatted[0]['text'])
print("="*80)
