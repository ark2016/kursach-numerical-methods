import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

# --- НАСТРОЙКИ (КОНФИГУРАЦИЯ) ---
CONFIG = {
    "model_name": "ai-forever/ruBert-large",
    "dataset_name": "seara/ru_go_emotions",
    "output_dir": "./rubert_large_lora_results",
    "hf_repo_id": "Ark2016/ruBert-large-emotions-lora", 
    
    "batch_size": 64,
    "grad_accum": 1,
    "lr": 3e-4,
    "epochs": 15,
    "max_len": 128,
    # Настройки MLFlow и S3 (должны совпадать с docker-compose/.env)
    "mlflow_tracking_uri": "http://localhost:5000",
    "experiment_name": "ruBert_Large_Emotions_LoRA_All_Linear",
    "s3_endpoint": "http://localhost:9000",
    "s3_access_key": "minio_root",
    "s3_secret_key": "minio_password"
}

# --- ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ ---
os.environ["MLFLOW_TRACKING_URI"] = CONFIG["mlflow_tracking_uri"]
os.environ["MLFLOW_EXPERIMENT_NAME"] = CONFIG["experiment_name"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = CONFIG["s3_endpoint"]
os.environ["AWS_ACCESS_KEY_ID"] = CONFIG["s3_access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG["s3_secret_key"]
os.environ["AWS_DEFAULT_REGION"] = "us-east-1" # Заглушка для boto3
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"    # Разрешаем http
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading dataset...")
    ds = load_dataset(CONFIG["dataset_name"], "simplified")
    all_labels = set()

    for split in ds.keys():
        for labels in ds[split]['labels']:
            all_labels.update(labels)
            
    label_list = sorted(list(all_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    print(f"Detected {len(label_list)} unique emotions.")
    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit([label_list])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def preprocess(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=CONFIG["max_len"])
        labels_matrix = mlb.transform(examples["labels"])
        tokenized["labels"] = labels_matrix.astype("float32").tolist()
        return tokenized

    tokenized_ds = ds.map(
        preprocess, 
        batched=True, 
        remove_columns=ds["train"].column_names,
        load_from_cache_file=False
    )
    print("Loading Model & LoRA Config...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(label_list),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"] 
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        results = {
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds)
        }
        return results

    def custom_collator(features):
        batch = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
        return batch

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        learning_rate=CONFIG["lr"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        save_total_limit=2,
        logging_steps=50,
        report_to="mlflow",
        run_name="full_lora_run_v2",
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=custom_collator, 
        compute_metrics=compute_metrics
    )
    print("Starting training...")
    trainer.train()
    print("Evaluating on Test...")
    eval_results = trainer.evaluate()
    print(f"Final Metrics: {eval_results}")
    print(f"Pushing to HF Hub: {CONFIG['hf_repo_id']}")
    model.push_to_hub(CONFIG['hf_repo_id'], use_auth_token=True)
    tokenizer.push_to_hub(CONFIG['hf_repo_id'], use_auth_token=True)
    print("Success! Training complete.")

if __name__ == "__main__":
    main()