# Commonsense Validation and Explanation (ComVE) - SubTasks A, B, and C

This repository contains implementations for the **ComVE shared task** (SemEval-2020) focused on commonsense validation and explanation. The task consists of three subtasks:

1. **SubTask A:** Given two statements, identify which one does not make sense.
2. **SubTask B:** Provide a reason why a given nonsensical statement is against commonsense, choosing from three options.
3. **SubTask C:** Generate a reason why a given nonsensical statement does not make sense.

The repository uses pre-trained language models like **RoBERTa** and **BART** from the Hugging Face Transformers library.

---

## SubTask A: Identifying Nonsensical Statements
### Problem
Given two statements, identify the nonsensical one.
- Example:
  - **Statement 1:** He put a turkey into the fridge.
  - **Statement 2:** He put an elephant into the fridge.

  The nonsensical statement is **Statement 2**.

### Approach
We framed this as a binary text classification problem using the **RoBERTa** model:
- Input: Two statements.
- Output: A label (0 or 1) indicating which statement is nonsensical.

### Implementation
1. **Pre-trained Model:** `roberta-base` or `roberta-large`.
2. **Tokenizer:** Tokenizes input pairs and truncates sequences.
3. **Trainer API:** Used for fine-tuning the model with `Trainer` from Hugging Face.

### Evaluation
The performance is evaluated using **accuracy**:
- Partial training: ~49% accuracy.
- Full training: ~92.9% accuracy.

---

## SubTask B: Reason Selection
### Problem
Given a nonsensical statement and three potential reasons, select the correct one.
- Example:
  - **Statement:** He put an elephant into the fridge.
  - **Options:**
    - Reason A: An elephant is much bigger than a fridge.
    - Reason B: Elephants are usually white while fridges are usually white.
    - Reason C: An elephant cannot eat a fridge.
  - The correct answer is **Reason A**.

### Approach
We modeled this as a multiple-choice problem using **RoBERTa**:
- Input: Nonsensical statement + candidate reasons.
- Output: The index of the correct reason.

### Implementation
1. **Pre-trained Model:** `roberta-base` or `roberta-large`.
2. **Data Collator:** Custom data collator for multiple-choice problems.
3. **Trainer API:** Fine-tuned using the Hugging Face `Trainer`.

### Evaluation
The performance is evaluated using **accuracy**:
- Partial training: ~52% accuracy.
- Full training: ~92.8% accuracy.

---

## SubTask C: Reason Generation
### Problem
Generate a valid explanation for why a given nonsensical statement does not make sense.
- Example:
  - **Statement:** He put an elephant into the fridge.
  - **Generated Reason:** An elephant is much bigger than a fridge.

### Approach
We modeled this as a sequence-to-sequence problem using **BART**:
- Input: Nonsensical statement.
- Output: Generated reason.

### Implementation
1. **Pre-trained Model:** `facebook/bart-base` or `facebook/bart-large`.
2. **Tokenizer:** Tokenizes the input and target sequences.
3. **Trainer API:** Fine-tuned using `Seq2SeqTrainer` from Hugging Face.

### Evaluation
The performance is evaluated using **BLEU** and **ROUGE** metrics:
- Partial training: BLEU ~0.216, ROUGE ~0.446.
- Full training: BLEU ~0.228, ROUGE ~0.461.

---

## Setup and Usage

### Prerequisites
- Python 3.7+
- Required libraries:
  ```bash
  pip install transformers datasets evaluate sacrebleu rouge
