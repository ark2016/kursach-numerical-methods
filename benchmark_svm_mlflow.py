import os
import sys
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# Добавляем путь для импорта локальных модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import compute_all_metrics_at_k
from utils.optimizers import AdamW as CustomAdamW

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

    # Optimizer settings
    "use_custom_adamw": True,  # True = кастомный AdamW, False = torch.optim.AdamW
    "adamw_weight_decay": 0.01,
    "adamw_betas": (0.9, 0.999),

    # Baseline sklearn SVM
    "run_sklearn_baseline": True,
    "sklearn_svm_C_values": [0.1, 1.0, 10.0],  # Разные значения C для LinearSVC
    "use_scaler": True,  # StandardScaler перед sklearn SVM

    # Metrics @k settings
    "k_values": [1, 3, 5, 10],

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

def train_sklearn_baselines(X_train, y_train, X_test, y_test, target_names_str, encoder_name):
    """
    Обучает и оценивает несколько sklearn SVM baselines:
    - LinearSVC с разными значениями C
    - SGDClassifier с hinge loss (линейный SVM)
    - SGDClassifier с log_loss (логистическая регрессия)
    """
    print(f"\n--- Training sklearn baselines for {encoder_name} ---")

    # Конвертируем в numpy
    X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
    y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

    # Масштабирование признаков (опционально)
    if CONFIG["use_scaler"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)
    else:
        X_train_scaled = X_train_np
        X_test_scaled = X_test_np

    # Список классификаторов для сравнения
    classifiers = []

    # LinearSVC с разными C
    for C in CONFIG["sklearn_svm_C_values"]:
        classifiers.append((
            f"LinearSVC_C{C}",
            LinearSVC(C=C, max_iter=10000, dual='auto')
        ))

    # SGD классификаторы
    classifiers.extend([
        ("SGD_hinge", SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, n_jobs=-1, random_state=42)),
        ("SGD_log", SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, n_jobs=-1, random_state=42)),
    ])

    all_baseline_results = {}

    for clf_name, base_clf in classifiers:
        print(f"\n  Training {clf_name}...")

        with mlflow.start_run(run_name=f"sklearn_{clf_name}_on_{encoder_name}"):
            mlflow.log_param("encoder_model", encoder_name)
            mlflow.log_param("classifier", clf_name)
            mlflow.log_param("use_scaler", CONFIG["use_scaler"])

            # OneVsRest wrapper для multi-label
            clf = OneVsRestClassifier(base_clf, n_jobs=-1)

            try:
                clf.fit(X_train_scaled, y_train_np)
                y_pred = clf.predict(X_test_scaled)

                # Decision function для метрик @k
                try:
                    y_scores = clf.decision_function(X_test_scaled)
                except AttributeError:
                    try:
                        y_scores = clf.predict_proba(X_test_scaled)
                    except AttributeError:
                        y_scores = y_pred.astype(float)

                # Базовые метрики
                metrics = {
                    "f1_micro": f1_score(y_test_np, y_pred, average='micro'),
                    "f1_macro": f1_score(y_test_np, y_pred, average='macro'),
                    "f1_weighted": f1_score(y_test_np, y_pred, average='weighted'),
                    "accuracy": accuracy_score(y_test_np, y_pred)
                }

                # Метрики @k
                metrics_at_k = compute_all_metrics_at_k(y_test_np, y_scores, k_values=CONFIG["k_values"])
                metrics.update(metrics_at_k)

                print(f"    {clf_name}: F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}, MAP@5={metrics.get('map_at_5', 0):.4f}")

                mlflow.log_metrics(metrics)

                # Classification report
                report = classification_report(y_test_np, y_pred, target_names=target_names_str, zero_division=0)
                report_filename = f"sklearn_{clf_name}_report.txt"
                with open(report_filename, "w") as f:
                    f.write(f"Encoder: {encoder_name}\nClassifier: {clf_name}\n\n")
                    f.write(report)
                mlflow.log_artifact(report_filename)

                all_baseline_results[f"sklearn_{clf_name}"] = metrics

            except Exception as e:
                print(f"    Error training {clf_name}: {e}")
                continue

    return all_baseline_results


def main():
    print(f"Start Benchmark. Device: {CONFIG['device']}")
    print(f"Using custom AdamW: {CONFIG['use_custom_adamw']}")

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

    all_results = {}

    for model_friendly_name, model_path in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Processing: {model_friendly_name}")
        print(f"{'='*40}")
        X_train, y_train, X_test, y_test = get_embeddings(model_friendly_name, model_path, ds, mlb)
        input_dim = X_train.shape[1]

        # --- CS-SVM с кастомным или стандартным AdamW ---
        with mlflow.start_run(run_name=f"CS_SVM_on_{model_friendly_name}"):
            mlflow.log_param("encoder_model", model_path)
            mlflow.log_param("svm_c_reg", CONFIG["C_reg"])
            mlflow.log_param("optimizer", "custom_AdamW" if CONFIG["use_custom_adamw"] else "torch_AdamW")
            mlflow.log_param("lr", CONFIG["svm_lr"])
            mlflow.log_param("weight_decay", CONFIG["adamw_weight_decay"])

            class_counts = y_train.sum(dim=0).to(CONFIG['device'])
            svm = MultilabelCSSVM(
                input_dim=input_dim,
                num_classes=len(label_list),
                class_counts=class_counts,
                total_samples=len(y_train),
                C_reg=CONFIG['C_reg'],
                C_neg_base=CONFIG['C_neg_base']
            ).to(CONFIG['device'])

            # Выбор оптимизатора
            if CONFIG["use_custom_adamw"]:
                optimizer = CustomAdamW(
                    svm.parameters(),
                    lr=CONFIG['svm_lr'],
                    weight_decay=CONFIG["adamw_weight_decay"],
                    betas=CONFIG["adamw_betas"]
                )
            else:
                optimizer = torch.optim.AdamW(
                    svm.parameters(),
                    lr=CONFIG['svm_lr'],
                    weight_decay=CONFIG["adamw_weight_decay"],
                    betas=CONFIG["adamw_betas"]
                )

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size_svm'], shuffle=True)
            print("Training CS-SVM...")
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

            print("Evaluating CS-SVM...")
            svm.eval()
            with torch.no_grad():
                test_logits = svm(X_test.to(CONFIG['device'])).cpu()
                pred_bin = (test_logits > 0).float().numpy()
                y_scores = test_logits.numpy()  # Для метрик @k
                y_true = y_test.numpy()

            # Базовые метрики
            metrics = {
                "f1_micro": f1_score(y_true, pred_bin, average='micro'),
                "f1_macro": f1_score(y_true, pred_bin, average='macro'),
                "accuracy": accuracy_score(y_true, pred_bin)
            }

            # Метрики @k
            metrics_at_k = compute_all_metrics_at_k(y_true, y_scores, k_values=CONFIG["k_values"])
            metrics.update(metrics_at_k)

            print(f"CS-SVM Results for {model_friendly_name}:")
            print(f"  F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
            print(f"  Precision@5={metrics.get('precision_at_5', 0):.4f}, MAP@5={metrics.get('map_at_5', 0):.4f}")
            print(f"  NDCG@5={metrics.get('ndcg_at_5', 0):.4f}, Hit_Rate@5={metrics.get('hit_rate_at_5', 0):.4f}")

            mlflow.log_metrics(metrics)
            torch.save(svm.state_dict(), "svm_model.pt")
            mlflow.log_artifact("svm_model.pt")

            report = classification_report(y_true, pred_bin, target_names=target_names_str, zero_division=0)
            with open("classification_report.txt", "w") as f:
                f.write(f"Encoder: {model_friendly_name}\nOptimizer: {'custom' if CONFIG['use_custom_adamw'] else 'torch'} AdamW\n\n")
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            all_results[f"CS_SVM_{model_friendly_name}"] = metrics
            print(f"CS-SVM run for {model_friendly_name} completed.")

        # --- sklearn Baseline SVMs ---
        if CONFIG["run_sklearn_baseline"]:
            sklearn_results = train_sklearn_baselines(
                X_train, y_train, X_test, y_test,
                target_names_str, model_friendly_name
            )
            # Добавляем все sklearn результаты с префиксом encoder
            for clf_name, metrics in sklearn_results.items():
                all_results[f"{clf_name}_{model_friendly_name}"] = metrics

    # Итоговое сравнение
    print("\n" + "="*80)
    print("SUMMARY - All Results")
    print("="*80)
    print(f"{'Model':<40} {'F1_micro':<10} {'F1_macro':<10} {'MAP@5':<10} {'NDCG@5':<10}")
    print("-"*80)
    for name, m in all_results.items():
        print(f"{name:<40} {m['f1_micro']:<10.4f} {m['f1_macro']:<10.4f} {m.get('map_at_5', 0):<10.4f} {m.get('ndcg_at_5', 0):<10.4f}")

if __name__ == "__main__":
    main()