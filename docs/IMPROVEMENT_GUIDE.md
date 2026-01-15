# Model Improvement Guide
## Enhancing the USC Course Recommendation System

A comprehensive guide to improving model accuracy, performance, and capabilities.

---

## Table of Contents

1. [Current Baseline](#current-baseline)
2. [Training Improvements](#training-improvements)
3. [Data Quality Enhancements](#data-quality-enhancements)
4. [Model Architecture Upgrades](#model-architecture-upgrades)
5. [Inference Optimization](#inference-optimization)
6. [Advanced Techniques](#advanced-techniques)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Current Baseline

### Existing Performance

| Metric | Current Value | Target |
|--------|---------------|--------|
| Validation Loss | 0.919 | < 0.5 |
| Training Iterations | 1,000 | 5,000-10,000 |
| Training Examples | 10,344 | 20,000+ |
| Model Size | 500M params | 1-3B params |
| Trainable Params | 0.297% | 0.5-1% |
| Response Quality | Moderate | High |

### Current Limitations

❌ **Short Training Duration**: Only 1,000 iterations
❌ **Limited Data Diversity**: 4 question types per course
❌ **Small Model**: 500M parameters may limit capability
❌ **Basic Prompting**: Simple instruction format
❌ **No Context**: Doesn't remember conversation history
❌ **Repetition Issues**: Sometimes generates repetitive text
❌ **No Retrieval**: Pure generation without knowledge lookup

---

## Training Improvements

### 1. Extended Training Duration

**Current**: 1,000 iterations
**Recommended**: 5,000-10,000 iterations

#### Implementation

```python
# finetune_model.py - Updated configuration
finetune_cmd = [
    "mlx_lm.lora",
    "--model", MODEL_NAME,
    "--train",
    "--data", ".",
    "--iters", "5000",  # ← Increased from 1000
    "--steps-per-eval", "100",
    "--learning-rate", "1e-5",
    "--batch-size", "2",
    "--num-layers", "16",  # ← Apply to more layers
    "--adapter-path", ADAPTER_DIR,
    "--save-every", "500"
]
```

**Expected Improvements**:
- ✅ Lower validation loss (target: 0.5-0.6)
- ✅ Better convergence
- ✅ More stable predictions
- ✅ Reduced repetition

**Time**: ~10-15 minutes (5x longer)

---

### 2. Learning Rate Scheduling

**Current**: Fixed learning rate (1e-5)
**Improvement**: Warm-up + Cosine decay

#### Implementation

```python
# Create custom training config
config = {
    "learning_rate": 5e-5,  # Start higher
    "warmup_steps": 100,     # Gradual warm-up
    "lr_schedule": "cosine", # Decay to 1e-6
    "min_lr": 1e-6,
}
```

**Benefits**:
- 🚀 Faster initial convergence
- 🎯 Better final convergence
- 📉 Smoother training curves
- ⚡ Escape local minima

---

### 3. Increased LoRA Rank

**Current**: Default rank (likely 8)
**Recommended**: 16-32

#### Implementation

```python
# In adapter configuration
lora_config = {
    "r": 32,        # ← Increased rank
    "lora_alpha": 64,  # Scale with rank (2x)
    "lora_dropout": 0.05,  # Prevent overfitting
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

**Trade-offs**:
- ✅ Higher capacity (better learning)
- ✅ More expressiveness
- ❌ Larger adapter size (32x)
- ❌ Slightly slower training

**Adapter Size**:
- Rank 8: ~5.6 MB
- Rank 16: ~11 MB
- Rank 32: ~22 MB

---

### 4. Gradient Accumulation

**Purpose**: Simulate larger batch sizes with limited memory

#### Implementation

```python
finetune_cmd = [
    # ... other args
    "--grad-accumulation-steps", "4",  # Effective batch size = 2 × 4 = 8
]
```

**Benefits**:
- 📊 More stable gradients
- 🎯 Better convergence
- 💾 Same memory usage
- ⚡ Slightly slower (4x steps per update)

---

### 5. Mixed Precision Training

**Current**: Full precision
**Improvement**: Use mixed precision (if supported)

```python
# MLX automatically handles this, but can optimize
model_config = {
    "dtype": "bfloat16",  # More stable than float16
}
```

**Benefits**:
- ⚡ 2x faster training
- 💾 50% less memory
- 🎯 Similar accuracy

---

## Data Quality Enhancements

### 1. Expand Training Data

**Current**: 10,344 examples (4 types per course)
**Target**: 25,000+ examples (10+ types per course)

#### New Question Types

```python
# prepare_training_data.py - Add more question types

question_types = {
    # Existing
    "program_inquiry": "What courses are available in {program}?",
    "course_details": "Tell me about {class_name}.",
    "prerequisites": "What are the prerequisites for {class_name}?",
    "schedule": "When is {class_name} offered?",

    # NEW: Add these
    "instructor_query": "Who teaches {class_name}?",
    "difficulty": "How difficult is {class_name}?",
    "workload": "What's the workload like for {class_name}?",
    "comparison": "Compare {class_name} with similar courses.",
    "learning_outcomes": "What will I learn in {class_name}?",
    "career_relevance": "How does {class_name} help my career?",
    "project_based": "Does {class_name} have projects or exams?",
    "follow_up": "What courses should I take after {class_name}?",
    "corequisites": "What courses pair well with {class_name}?",
    "units_explanation": "How many units is {class_name} and what does that mean?",
}
```

#### Enhanced Data Generation

```python
def generate_enhanced_training_data(courses):
    training_data = []

    for course in courses:
        # Extract course info
        program = course['Program Code']
        class_id = course['Class ID']
        class_name = course['Class Name']
        description = course['Catalogue']
        prereqs = course['Prerequisites']
        units = course['Units']
        time = course['Time']
        days = course['Days']
        instructor = course['Instructor']

        # 1. Basic course inquiry
        training_data.append({
            "instruction": f"Tell me about {class_name}.",
            "response": f"{class_name} ({class_id}) is offered by the {program} program.\n\n{description}\n\nKey Details:\n- Units: {units}\n- Prerequisites: {prereqs}\n- Schedule: {days} at {time}\n- Instructor: {instructor}"
        })

        # 2. Prerequisite chain
        training_data.append({
            "instruction": f"What do I need to take before {class_name}?",
            "response": f"To enroll in {class_name}, you need: {prereqs}. Make sure you've completed these requirements before registering."
        })

        # 3. Schedule planning
        training_data.append({
            "instruction": f"Can you help me fit {class_name} into my schedule?",
            "response": f"{class_name} is offered {days} at {time}. This is a {units} course. Plan your week accordingly to avoid conflicts."
        })

        # 4. Instructor information
        if instructor and instructor != "None":
            training_data.append({
                "instruction": f"Who is teaching {class_name}?",
                "response": f"{class_name} is taught by {instructor}. You can find more information about the instructor and their teaching style in the course reviews."
            })

        # 5. Multi-turn conversation simulation
        training_data.append({
            "instruction": f"I'm interested in {program} courses.",
            "response": f"Great! {program} offers many interesting courses. One option is {class_name} ({class_id}). {description}"
        })

        # 6. Comparison questions
        training_data.append({
            "instruction": f"How does {class_name} compare to other {program} courses?",
            "response": f"{class_name} is a {units} course in the {program} program. {description} It's scheduled for {days} at {time}."
        })

        # 7. Units explanation
        units_num = extract_units(units)  # Extract numeric value
        training_data.append({
            "instruction": f"How many units is {class_name}?",
            "response": f"{class_name} is {units}. This typically means you'll spend approximately {units_num * 3} hours per week on lectures, assignments, and studying."
        })

        # 8. Career relevance (for CS/technical courses)
        if program in ['CS', 'CSCI', 'EE', 'DATA']:
            training_data.append({
                "instruction": f"How will {class_name} help my career?",
                "response": f"{class_name} provides essential skills in {program}. The concepts covered include: {description} These are highly valued in industry and research."
            })

        # 9. Follow-up courses
        training_data.append({
            "instruction": f"What should I take after completing {class_name}?",
            "response": f"After {class_name}, you can explore advanced courses that list it as a prerequisite. Check courses in {program} that build on these foundations."
        })

        # 10. Time commitment
        training_data.append({
            "instruction": f"How much time per week does {class_name} require?",
            "response": f"{class_name} is {units}, which typically requires {units_num * 3}-{units_num * 4} hours per week including class time, homework, and studying. Classes meet {days} at {time}."
        })

    return training_data
```

**Expected Improvements**:
- ✅ 2.5x more training data
- ✅ Better question diversity
- ✅ More natural responses
- ✅ Better generalization

---

### 2. Data Augmentation

#### Add Paraphrasing

```python
import random

def paraphrase_question(question):
    """Generate variations of the same question"""
    templates = {
        "Tell me about": [
            "What can you tell me about",
            "I'd like to know about",
            "Give me information on",
            "Describe",
            "Explain"
        ],
        "What are the prerequisites": [
            "What do I need to take before",
            "What courses are required before",
            "What's needed as a prerequisite for",
            "Requirements for"
        ],
        "When is": [
            "What time is",
            "What days does",
            "What's the schedule for",
            "When does"
        ]
    }

    for pattern, variations in templates.items():
        if pattern in question:
            new_pattern = random.choice(variations)
            return question.replace(pattern, new_pattern)

    return question
```

#### Synthetic Negative Examples

```python
def generate_negative_examples():
    """Generate examples of what NOT to do"""
    return [
        {
            "instruction": "Write my essay for CS101",
            "response": "I can't write assignments for you, but I can help you understand CS101 course content, prerequisites, and schedule. Would you like information about the course?"
        },
        {
            "instruction": "Hack into the registration system",
            "response": "I'm a course recommendation assistant. I can only provide information about USC courses, prerequisites, and schedules. For registration help, please contact the registrar's office."
        },
        {
            "instruction": "Tell me about courses at UCLA",
            "response": "I specialize in USC courses only. I don't have information about courses at other universities. Is there a USC course you'd like to know about?"
        }
    ]
```

---

### 3. Add Contextual Information

#### Enrich Course Descriptions

```python
def enrich_course_data(course):
    """Add contextual information"""

    # Extract difficulty level from course number
    course_num = extract_course_number(course['Class ID'])

    if course_num < 200:
        difficulty = "Introductory"
        audience = "This course is suitable for beginners."
    elif course_num < 400:
        difficulty = "Intermediate"
        audience = "This course assumes some foundational knowledge."
    elif course_num < 600:
        difficulty = "Advanced"
        audience = "This course is for advanced students."
    else:
        difficulty = "Graduate-level"
        audience = "This course is designed for graduate students."

    # Add typical workload
    units_num = extract_units(course['Units'])
    workload = f"{units_num * 3}-{units_num * 4} hours per week"

    # Add semester recommendation
    if "Fall" in course['Class Section'] or "Spring" in course['Class Section']:
        semester_info = "typically offered in Fall/Spring"
    else:
        semester_info = "check availability each semester"

    course['enriched'] = {
        'difficulty': difficulty,
        'audience': audience,
        'workload': workload,
        'semester_info': semester_info
    }

    return course
```

---

## Model Architecture Upgrades

### 1. Use Larger Base Model

**Current**: Qwen2-0.5B (500M parameters)
**Recommended**: Qwen2-1.5B or Qwen2-3B

#### Trade-offs

| Model | Size | Speed | Quality | Memory | Recommendation |
|-------|------|-------|---------|--------|----------------|
| Qwen2-0.5B | 500MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 1.6GB | Current |
| Qwen2-1.5B | 1.5GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 3.0GB | **Best Balance** |
| Qwen2-3B | 3.0GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 5.0GB | Maximum Quality |
| Qwen2-7B | 7.0GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 12GB | Overkill |

#### Implementation

```python
# finetune_model.py
# Change model selection
MODEL_NAME = "mlx-community/Qwen2-1.5B-Instruct-4bit"  # ← Upgrade

# Adjust training parameters for larger model
finetune_cmd = [
    "mlx_lm.lora",
    "--model", MODEL_NAME,
    "--train",
    "--data", ".",
    "--iters", "3000",  # May need fewer iterations
    "--batch-size", "1",  # Reduce for memory
    "--learning-rate", "5e-6",  # Lower for stability
    # ... rest of config
]
```

**Expected Improvements**:
- ✅ Better understanding
- ✅ More coherent responses
- ✅ Less repetition
- ✅ Better reasoning
- ❌ Slower inference
- ❌ More memory

---

### 2. Increase LoRA Coverage

**Current**: 8 layers
**Recommended**: All layers (16-32)

```python
finetune_cmd = [
    # ... other args
    "--num-layers", "-1",  # -1 means all layers
]
```

**Benefits**:
- ✅ More thorough adaptation
- ✅ Better task-specific learning
- ❌ Slightly larger adapters

---

### 3. Multi-Adapter Strategy

Train specialized adapters for different aspects:

```python
# Train separate adapters
adapters = {
    "course_details": "adapters/course_details",
    "recommendations": "adapters/recommendations",
    "scheduling": "adapters/scheduling",
    "prerequisites": "adapters/prerequisites",
}

# Route queries to appropriate adapter
def route_query(question):
    if "recommend" in question.lower():
        return adapters["recommendations"]
    elif "prerequisite" in question.lower():
        return adapters["prerequisites"]
    elif "when" in question.lower() or "time" in question.lower():
        return adapters["scheduling"]
    else:
        return adapters["course_details"]
```

---

## Inference Optimization

### 1. Implement Retrieval-Augmented Generation (RAG)

**Problem**: Model may hallucinate or generate incorrect info
**Solution**: Retrieve actual course data before generating

#### Implementation

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class CourseRAG:
    def __init__(self, course_data):
        self.courses = course_data
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Create course embeddings
        course_texts = [
            f"{c['Class Name']} {c['Catalogue']}"
            for c in self.courses
        ]
        self.course_embeddings = self.embedder.encode(course_texts)

    def retrieve_relevant_courses(self, query, top_k=3):
        """Retrieve most relevant courses for query"""
        query_embedding = self.embedder.encode([query])[0]

        # Compute similarities
        similarities = np.dot(self.course_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.courses[i] for i in top_indices]

    def generate_with_rag(self, question):
        """Generate response with retrieved context"""
        # 1. Retrieve relevant courses
        relevant_courses = self.retrieve_relevant_courses(question)

        # 2. Build context
        context = "Relevant courses:\n"
        for course in relevant_courses:
            context += f"- {course['Class Name']}: {course['Catalogue']}\n"

        # 3. Create enhanced prompt
        prompt = f"""<|im_start|>system
You are a helpful USC course recommendation assistant. Use the following course information to answer accurately.<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""

        # 4. Generate with model
        response = generate(model, tokenizer, prompt=prompt)
        return response
```

**Benefits**:
- ✅ Factually accurate responses
- ✅ No hallucination
- ✅ Always up-to-date
- ✅ Verifiable information

---

### 2. Add Response Filtering

```python
def filter_response(response):
    """Remove repetition and clean output"""

    # Remove repetitive patterns
    lines = response.split('\n')
    seen = set()
    filtered_lines = []

    for line in lines:
        if line.strip() and line not in seen:
            seen.add(line)
            filtered_lines.append(line)

    # Remove incomplete sentences at the end
    cleaned = '\n'.join(filtered_lines)
    if cleaned and not cleaned[-1] in '.!?':
        cleaned = cleaned.rsplit('.', 1)[0] + '.'

    return cleaned
```

---

### 3. Temperature and Sampling Tuning

```python
# Experiment with generation parameters
generation_configs = {
    "factual": {
        "temperature": 0.3,  # Low for factual responses
        "top_p": 0.8,
        "repetition_penalty": 1.2
    },
    "creative": {
        "temperature": 0.7,  # Higher for recommendations
        "top_p": 0.9,
        "repetition_penalty": 1.1
    },
    "conservative": {
        "temperature": 0.1,  # Very factual
        "top_p": 0.95,
        "repetition_penalty": 1.3
    }
}

# Use appropriate config based on query type
config = generation_configs["factual"]
response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=200,
    temperature=config["temperature"],
    top_p=config["top_p"],
    repetition_penalty=config["repetition_penalty"]
)
```

---

## Advanced Techniques

### 1. Multi-Task Training

Train on multiple related tasks:

```python
training_tasks = {
    "course_qa": {
        "weight": 0.4,
        "data": "course_qa.jsonl"
    },
    "recommendation": {
        "weight": 0.3,
        "data": "recommendations.jsonl"
    },
    "prerequisite_planning": {
        "weight": 0.2,
        "data": "prerequisites.jsonl"
    },
    "schedule_optimization": {
        "weight": 0.1,
        "data": "scheduling.jsonl"
    }
}
```

---

### 2. Curriculum Learning

Train on increasingly difficult examples:

```python
def curriculum_training():
    """Train with progressive difficulty"""

    stages = [
        {
            "name": "Basic QA",
            "iterations": 1000,
            "data": "easy_qa.jsonl",
            "learning_rate": 1e-5
        },
        {
            "name": "Complex QA",
            "iterations": 2000,
            "data": "medium_qa.jsonl",
            "learning_rate": 5e-6
        },
        {
            "name": "Multi-hop Reasoning",
            "iterations": 2000,
            "data": "hard_qa.jsonl",
            "learning_rate": 1e-6
        }
    ]

    for stage in stages:
        print(f"Training stage: {stage['name']}")
        train_model(
            data=stage['data'],
            iterations=stage['iterations'],
            learning_rate=stage['learning_rate']
        )
```

---

### 3. Ensemble Methods

Combine multiple models:

```python
class EnsembleModel:
    def __init__(self, model_paths):
        self.models = [load(path) for path in model_paths]

    def generate_ensemble(self, prompt):
        """Generate from multiple models and combine"""
        responses = []

        for model, tokenizer in self.models:
            response = generate(model, tokenizer, prompt=prompt)
            responses.append(response)

        # Vote or combine responses
        best_response = self.select_best(responses)
        return best_response

    def select_best(self, responses):
        """Select best response using heuristics"""
        # Score each response
        scores = []
        for resp in responses:
            score = 0
            score += len(resp)  # Prefer longer responses
            score -= resp.count("..") * 10  # Penalize ellipsis
            score -= self.count_repetition(resp) * 20
            scores.append(score)

        return responses[np.argmax(scores)]
```

---

### 4. Active Learning

Identify weak areas and generate targeted training data:

```python
def identify_weak_areas(model, test_questions):
    """Find question types with poor performance"""

    weak_categories = {}

    for question, expected_answer in test_questions:
        response = generate(model, tokenizer, prompt=question)

        # Evaluate response quality
        score = evaluate_response(response, expected_answer)

        category = categorize_question(question)

        if category not in weak_categories:
            weak_categories[category] = []

        weak_categories[category].append(score)

    # Find lowest scoring categories
    avg_scores = {
        cat: np.mean(scores)
        for cat, scores in weak_categories.items()
    }

    return sorted(avg_scores.items(), key=lambda x: x[1])[:3]

# Generate more training data for weak areas
weak_areas = identify_weak_areas(model, test_set)
for category, score in weak_areas:
    print(f"Generating more data for: {category} (score: {score})")
    additional_data = generate_targeted_data(category, n=1000)
    training_data.extend(additional_data)
```

---

## Evaluation Metrics

### 1. Quantitative Metrics

```python
def evaluate_model_comprehensive(model, tokenizer, test_set):
    """Comprehensive evaluation"""

    metrics = {
        "perplexity": [],
        "bleu_scores": [],
        "rouge_scores": [],
        "response_length": [],
        "factual_accuracy": [],
        "relevance": []
    }

    for example in test_set:
        question = example['question']
        expected = example['answer']

        response = generate(model, tokenizer, prompt=question)

        # Compute metrics
        metrics['perplexity'].append(compute_perplexity(response))
        metrics['bleu_scores'].append(compute_bleu(response, expected))
        metrics['rouge_scores'].append(compute_rouge(response, expected))
        metrics['response_length'].append(len(response.split()))
        metrics['factual_accuracy'].append(check_facts(response, example))
        metrics['relevance'].append(compute_relevance(response, question))

    # Aggregate
    results = {
        metric: np.mean(values)
        for metric, values in metrics.items()
    }

    return results
```

### 2. Qualitative Evaluation

```python
def human_evaluation_template():
    """Template for human evaluation"""

    criteria = {
        "relevance": "Does the response answer the question? (1-5)",
        "accuracy": "Is the information factually correct? (1-5)",
        "completeness": "Is the response complete? (1-5)",
        "clarity": "Is the response clear and well-structured? (1-5)",
        "helpfulness": "Is the response helpful to the user? (1-5)"
    }

    return criteria

# Create evaluation dataset
def create_evaluation_set():
    test_questions = [
        {
            "question": "What computer science courses should I take for AI?",
            "category": "recommendation",
            "difficulty": "medium"
        },
        {
            "question": "Tell me about CS 101",
            "category": "course_details",
            "difficulty": "easy"
        },
        # ... more test cases
    ]

    return test_questions
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

**Priority: High | Effort: Low | Impact: Medium**

1. ✅ Extended training (5000 iterations)
   ```bash
   # Already configured in finetune_model.py
   python finetune_model.py  # Now trains for 5000 iterations
   ```

2. ✅ Increase LoRA rank to 16
   ```python
   # Add to config
   "--lora-rank", "16"
   ```

3. ✅ Add more question types (10 → 15 types)
   ```python
   # Update prepare_training_data.py
   python prepare_training_data.py
   ```

4. ✅ Implement response filtering
   ```python
   # Add to inference.py
   response = filter_response(raw_response)
   ```

**Expected Improvement**: 15-20% better response quality

---

### Phase 2: Data Enhancement (3-5 days)

**Priority: High | Effort: Medium | Impact: High**

1. ✅ Generate 25K training examples
   - Add more question types
   - Add paraphrasing
   - Add negative examples

2. ✅ Enrich course data
   - Add difficulty levels
   - Add workload estimates
   - Add career relevance

3. ✅ Create validation test set
   - 500 human-annotated examples
   - Cover all question types
   - Include edge cases

**Expected Improvement**: 25-35% better response quality

---

### Phase 3: Model Upgrade (2-3 days)

**Priority: Medium | Effort: Medium | Impact: High**

1. ✅ Upgrade to Qwen2-1.5B
   ```python
   MODEL_NAME = "mlx-community/Qwen2-1.5B-Instruct-4bit"
   ```

2. ✅ Train with all layers
   ```python
   "--num-layers", "-1"
   ```

3. ✅ Use gradient accumulation
   ```python
   "--grad-accumulation-steps", "4"
   ```

**Expected Improvement**: 30-40% better response quality

---

### Phase 4: Advanced Techniques (5-7 days)

**Priority: Medium | Effort: High | Impact: Very High**

1. ✅ Implement RAG system
   - Course embedding database
   - Semantic search
   - Context injection

2. ✅ Multi-adapter routing
   - Train specialized adapters
   - Implement routing logic
   - A/B test performance

3. ✅ Add conversation memory
   - Track conversation history
   - Maintain context
   - Handle follow-ups

**Expected Improvement**: 50-60% better response quality

---

### Phase 5: Production Optimization (3-5 days)

**Priority: Low | Effort: High | Impact: Medium**

1. ✅ Quantization optimization
2. ✅ Caching layer
3. ✅ Load balancing
4. ✅ Monitoring and logging
5. ✅ A/B testing infrastructure

---

## Expected Results

### Performance Trajectory

| Phase | Val Loss | Quality Score | Response Time |
|-------|----------|---------------|---------------|
| Baseline | 0.919 | 60% | 1.2s |
| Phase 1 | 0.650 | 70% | 1.2s |
| Phase 2 | 0.550 | 80% | 1.3s |
| Phase 3 | 0.450 | 85% | 1.8s |
| Phase 4 | 0.400 | 92% | 2.0s |
| Phase 5 | 0.400 | 92% | 1.5s |

### Final Target Metrics

- ✅ Validation Loss: < 0.5
- ✅ Response Accuracy: > 90%
- ✅ User Satisfaction: > 4.5/5
- ✅ Response Time: < 2 seconds
- ✅ Factual Accuracy: > 95%

---

## Monitoring and Iteration

### 1. Continuous Evaluation

```python
import wandb

# Log to Weights & Biases
wandb.init(project="usc-course-recommender")

def log_training_metrics(iteration, metrics):
    wandb.log({
        "iteration": iteration,
        "train_loss": metrics['train_loss'],
        "val_loss": metrics['val_loss'],
        "learning_rate": metrics['lr'],
    })
```

### 2. User Feedback Loop

```python
def collect_feedback(response, user_rating):
    """Collect user feedback for improvement"""

    feedback_data = {
        "response": response,
        "rating": user_rating,
        "timestamp": datetime.now(),
        "question_type": categorize_question(response)
    }

    # Save for retraining
    save_feedback(feedback_data)

    # If rating < 3, flag for review
    if user_rating < 3:
        flag_for_review(feedback_data)
```

---

## Conclusion

### Recommended Priority Order

1. **Start with Phase 1** (Quick wins) - Immediate 15-20% improvement
2. **Move to Phase 2** (Data) - Foundation for long-term quality
3. **Then Phase 3** (Model) - Significant capability boost
4. **Consider Phase 4** (Advanced) - If maximum quality needed
5. **Finally Phase 5** (Production) - When deploying at scale

### Next Steps

```bash
# 1. Run extended training (already configured)
python finetune_model.py

# 2. Evaluate results
python evaluation_summary.py

# 3. Test improvements
python test_inference.py

# 4. Implement Phase 2 enhancements
# Update prepare_training_data.py with new question types

# 5. Re-train with enhanced data
python prepare_training_data.py
python finetune_model.py
```

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Estimated Total Improvement**: 50-60% better response quality
**Timeline**: 2-4 weeks for full implementation
