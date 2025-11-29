import os
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    "dataset_name": "seara/ru_go_emotions",
    "batch_size_embed": 64, 
    "batch_size_svm": 256,
    "max_len": 128,
    "svm_lr": 0.005,
    "svm_epochs": 20,
    "C_reg": 0.01,
    "C_neg_base": 2.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "./embeddings_cache", 
    
    # MLFLOW Settings
    "mlflow_tracking_uri": "http://localhost:5000",
    "experiment_name": "Encoder_Comparison_CS_SVM",
    "s3_endpoint": "http://localhost:9000",
    "s3_access_key": "minio_root",
    "s3_secret_key": "minio_password"
}

# СПИСОК МОДЕЛЕЙ ДЛЯ СРАВНЕНИЯ
MODELS = {
    "Baseline_ruBert": "ai-forever/ruBert-large",
    "Foreign_ruRoberta": "fyaronskiy/ruRoberta-large-ru-go-emotions",
    "My_LoRA_Ark2016": "Ark2016/ruBert-large-emotions-lora",
    "fyaronskiy/ruRoberta-large-ru-go-emotions": "fyaronskiy/ruRoberta-large-ru-go-emotions"
}

# Настройка окружения для MLFlow/Boto3
os.environ["MLFLOW_TRACKING_URI"] = CONFIG["mlflow_tracking_uri"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = CONFIG["s3_endpoint"]
os.environ["AWS_ACCESS_KEY_ID"] = CONFIG["s3_access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG["s3_secret_key"]
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# Создаем папку для кэша
os.makedirs(CONFIG["cache_dir"], exist_ok=True)

class MultilabelCSSVM(nn.Module):
    def __init__(self, input_dim, num_classes, class_counts, total_samples, C_reg, C_neg_base=2.0):
        super().__init__()
        self.C_reg = C_reg
        self.w = nn.Parameter(torch.randn(input_dim, num_classes) * np.sqrt(2.0 / (input_dim + num_classes)))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.C_neg = C_neg_base 
        self.kappa = 1.0 / (2 * self.C_neg - 1)
        neg_counts = total_samples - class_counts
        imbalance_ratios = neg_counts / (class_counts + 1e-6)
        c_pos_calculated = imbalance_ratios * self.C_neg
        min_c_pos = 2 * self.C_neg - 1
        self.C_pos_per_class = torch.clamp(c_pos_calculated, min=min_c_pos, max=50.0).float().to(CONFIG['device'])
        self.weight_pos_term = self.C_reg * self.C_pos_per_class
        self.weight_neg_term = self.C_reg * (1.0 / self.kappa)

    def forward(self, x):
        return x @ self.w + self.b

    def compute_loss(self, x, y_target):
        f_x = self.forward(x)
        reg_loss = 0.5 * torch.sum(self.w ** 2)
        is_pos = (y_target == 1).float()
        is_neg = (y_target == 0).float()
        raw_pos_loss = torch.clamp(1 - f_x, min=0)      
        raw_neg_loss = torch.clamp(f_x + self.kappa, min=0) 
        sum_pos_errors = torch.sum(is_pos * raw_pos_loss, dim=0)
        sum_neg_errors = torch.sum(is_neg * raw_neg_loss, dim=0)
        term_pos = torch.sum(self.weight_pos_term * sum_pos_errors)
        term_neg = self.weight_neg_term * torch.sum(sum_neg_errors)
        return reg_loss + term_pos + term_neg

def load_encoder(model_path):
    print(f"Loading encoder: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=28,
            output_hidden_states=True,
            problem_type="multi_label_classification"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print(" -> Detected PEFT/LoRA adapter (Loaded as SeqCls).")
    except Exception:
        # Для обычных моделей (Baseline, Foreign) грузим просто AutoModel
        model = AutoModel.from_pretrained(model_path)
        print(f" -> Detected Standard Transformer (Base/Full).")
    model.to(CONFIG['device'])
    model.eval()
    return model, tokenizer

def get_embeddings(model_key, model_path, dataset, mlb):
    safe_name = model_key.replace("/", "_")
    cache_path = os.path.join(CONFIG["cache_dir"], f"{safe_name}.pt")
    
    if os.path.exists(cache_path):
        print(f"Found cached embeddings for {model_key}. Loading...")
        data = torch.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    
    print(f"Cache not found. Extracting embeddings for {model_key}...")
    model, tokenizer = load_encoder(model_path)
    
    def _extract(split_name):
        texts = dataset[split_name]['text']
        labels = mlb.transform(dataset[split_name]['labels'])
        emb_list = []
        lbl_list = []
        
        for i in tqdm(range(0, len(texts), CONFIG['batch_size_embed']), desc=f"Extr {split_name}"):
            batch_text = texts[i : i + CONFIG['batch_size_embed']]
            batch_lbl = labels[i : i + CONFIG['batch_size_embed']]
            inp = tokenizer(batch_text, padding=True, truncation=True, max_length=CONFIG['max_len'], return_tensors="pt").to(CONFIG['device'])
            with torch.no_grad():
                out = model(**inp)
                if hasattr(out, "last_hidden_state"): # Это AutoModel (Baseline, Foreign)
                    emb = out.last_hidden_state[:, 0, :]
                elif hasattr(out, "hidden_states"):# Это AutoModelForSequenceClassification (LoRA)
                    emb = out.hidden_states[-1][:, 0, :]
                else:
                    raise ValueError("Unknown model output format")
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb_list.append(emb.cpu())
            lbl_list.append(torch.tensor(batch_lbl, dtype=torch.float32))
            
        return torch.cat(emb_list), torch.cat(lbl_list)

    X_train, y_train = _extract('train')
    X_test, y_test = _extract('test')
    print(f"Saving embeddings to {cache_path}...")
    torch.save({
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test
    }, cache_path)
    del model, tokenizer
    torch.cuda.empty_cache()
    return X_train, y_train, X_test, y_test

def main():
    print(f"Start Benchmark. Device: {CONFIG['device']}")
    ds = load_dataset(CONFIG['dataset_name'], "simplified")
    all_labels = set()
    for split in ds.keys():
        for labels in ds[split]['labels']:
            all_labels.update(labels)
    label_list = sorted(list(all_labels))
    try:
        class_names = ds['train'].features['labels'].feature.names
        target_names_str = [class_names[i] for i in label_list]
        print(f"Class names loaded: {target_names_str[:3]}...")
    except:
        print("Class names not found, using IDs.")
        target_names_str = [str(l) for l in label_list]

    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit([label_list])
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    for model_friendly_name, model_path in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Processing: {model_friendly_name}")
        print(f"{'='*40}")
        X_train, y_train, X_test, y_test = get_embeddings(model_friendly_name, model_path, ds, mlb)
        input_dim = X_train.shape[1]
        with mlflow.start_run(run_name=f"SVM_on_{model_friendly_name}"):
            mlflow.log_param("encoder_model", model_path)
            mlflow.log_param("svm_c_reg", CONFIG["C_reg"])
            class_counts = y_train.sum(dim=0).to(CONFIG['device'])
            svm = MultilabelCSSVM(
                input_dim=input_dim,
                num_classes=len(label_list),
                class_counts=class_counts,
                total_samples=len(y_train),
                C_reg=CONFIG['C_reg'],
                C_neg_base=CONFIG['C_neg_base']
            ).to(CONFIG['device'])
            optimizer = torch.optim.AdamW(svm.parameters(), lr=CONFIG['svm_lr'])
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size_svm'], shuffle=True)
            print("Training SVM...")
            svm.train()
            for epoch in range(CONFIG["svm_epochs"]):
                total_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
                    optimizer.zero_grad()
                    loss = svm.compute_loss(bx, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                if (epoch+1) % 5 == 0:
                    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
            print("Evaluating...")
            svm.eval()
            with torch.no_grad():
                test_logits = svm(X_test.to(CONFIG['device'])).cpu()
                pred_bin = (test_logits > 0).float().numpy()
                y_true = y_test.numpy()
            metrics = {
                "f1_micro": f1_score(y_true, pred_bin, average='micro'),
                "f1_macro": f1_score(y_true, pred_bin, average='macro'),
                "accuracy": accuracy_score(y_true, pred_bin)
            }
            print(f"Results for {model_friendly_name}: {metrics}")
            mlflow.log_metrics(metrics)
            torch.save(svm.state_dict(), "svm_model.pt")
            mlflow.log_artifact("svm_model.pt")
            report = classification_report(y_true, pred_bin, target_names=target_names_str, zero_division=0)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")
            print(f"Run for {model_friendly_name} completed.")

if __name__ == "__main__":
    main()