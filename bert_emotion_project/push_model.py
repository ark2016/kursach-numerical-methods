import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi

# --- НАСТРОЙКИ ---
USER = "Ark2016"
REPO_NAME = "ruBert-large-emotions-lora"
REPO_ID = f"{USER}/{REPO_NAME}"
CHECKPOINT_DIR = "./rubert_large_lora_results/checkpoint-10185"
print(f"Loading adapter from {CHECKPOINT_DIR}...")
base_model_name = "ai-forever/ruBert-large"
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=28,
    problem_type="multi_label_classification"
)
# Загружаем обученный LoRA адаптер
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
# Загружаем токенизатор из той же папки (Trainer его туда сохранил)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

print(f"Pushing to Hugging Face: {REPO_ID} ...")
model.push_to_hub(REPO_ID, token=True)
tokenizer.push_to_hub(REPO_ID, token=True)
print("Done! Check your profile: https://huggingface.co/" + REPO_ID)
print("Generating Model Card...")
readme_text = f"""---
language: 
- ru
pipeline_tag: text-classification
tags:
- emotion
- multi-label
- bert
- lora
- rubert-large
license: mit
datasets:
- seara/ru_go_emotions
metrics:
- f1
- accuracy
library_name: peft
---

# RuBERT-large Emotion Detection (LoRA Fine-tuned)

Model trained by **{USER}**.
Base model: [ai-forever/ruBert-large](https://huggingface.co/ai-forever/ruBert-large).

## Metrics
- **F1 Micro:** 0.57
- **Accuracy:** 0.45

## Usage
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo_id = "{REPO_ID}"
model = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained("ai-forever/ruBert-large", num_labels=28, problem_type="multi_label_classification"), repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```
"""

api = HfApi()
try:
    api.upload_file(
        path_or_fileobj=readme_text.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Add model card"
    )
    print("README pushed.")
except Exception as e:
    print(f"Could not push README: {e}")