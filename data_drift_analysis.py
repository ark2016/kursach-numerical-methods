import os
import torch
import torch.nn as nn
import numpy as np
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from scipy.stats import wasserstein_distance
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.label_mapping import (
    GO_EMOTIONS_LABELS, IZARD_EMOTIONS_LABELS,
    GO_TO_IZARD_MAPPING, convert_go_emotions_binary_to_izard, get_mapping_matrix
)
from utils.metrics import compute_all_metrics_at_k
from utils.optimizers import AdamW as CustomAdamW

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    "source_dataset": "seara/ru_go_emotions",
    "target_dataset": "Djacon/ru-izard-emotions",
    "batch_size": 64,
    "batch_size_svm": 256,
    "max_len": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "./embeddings_cache",

    # CS-SVM settings (как в benchmark_svm_mlflow.py)
    "svm_lr": 0.005,
    "svm_epochs": 20,
    "C_reg": 0.01,
    "C_neg_base": 2.0,

    # Sklearn baseline settings
    "sklearn_svm_C_values": [0.1, 1.0],
    "use_scaler": True,

    # Metrics @k settings
    "k_values": [1, 3, 5, 10],

    # MLFlow Settings
    "mlflow_tracking_uri": "http://localhost:5000",
    "experiment_name": "Data_Drift_Analysis",
    "s3_endpoint": "http://localhost:9000",
    "s3_access_key": "minio_root",
    "s3_secret_key": "minio_password"
}

MODELS = {
    "Baseline_ruBert": "ai-forever/ruBert-large",
    "Foreign_ruRoberta": "fyaronskiy/ruRoberta-large-ru-go-emotions",
    "My_LoRA_Ark2016": "Ark2016/ruBert-large-emotions-lora",
}

# Настройка окружения для MLFlow
os.environ["MLFLOW_TRACKING_URI"] = CONFIG["mlflow_tracking_uri"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = CONFIG["s3_endpoint"]
os.environ["AWS_ACCESS_KEY_ID"] = CONFIG["s3_access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG["s3_secret_key"]
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

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


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        XY = np.vstack([X[:1000], Y[:1000]])
        distances = np.linalg.norm(XY[:, None] - XY[None, :], axis=2)
        gamma = 1.0 / (2 * np.median(distances[distances > 0]) ** 2)

    def rbf_kernel(A, B, gamma):
        dist = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
        return np.exp(-gamma * dist)

    max_samples = 2000
    if len(X) > max_samples:
        X = X[np.random.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    n, m = len(X), len(Y)
    mmd2 = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
    mmd2 += (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
    mmd2 -= 2 * np.mean(K_XY)

    return max(0, mmd2)


def compute_wasserstein_per_dim(X: np.ndarray, Y: np.ndarray) -> dict:
    distances = []
    n_dims = min(X.shape[1], 100)

    for i in range(n_dims):
        dist = wasserstein_distance(X[:, i], Y[:, i])
        distances.append(dist)

    return {
        "wasserstein_mean": np.mean(distances),
        "wasserstein_std": np.std(distances),
        "wasserstein_max": np.max(distances),
    }


def compute_cosine_similarity_stats(X: np.ndarray, Y: np.ndarray, n_samples: int = 1000) -> dict:
    if len(X) > n_samples:
        X = X[np.random.choice(len(X), n_samples, replace=False)]
    if len(Y) > n_samples:
        Y = Y[np.random.choice(len(Y), n_samples, replace=False)]

    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cross_sim = X_norm @ Y_norm.T
    within_X = X_norm @ X_norm.T
    within_Y = Y_norm @ Y_norm.T

    return {
        "cross_cosine_mean": np.mean(cross_sim),
        "cross_cosine_std": np.std(cross_sim),
        "within_X_cosine_mean": np.mean(within_X[np.triu_indices(len(X), k=1)]),
        "within_Y_cosine_mean": np.mean(within_Y[np.triu_indices(len(Y), k=1)]),
    }


def load_encoder(model_path: str):
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
        print(" -> Detected PEFT/LoRA adapter")
        is_lora = True
    except Exception:
        model = AutoModel.from_pretrained(model_path)
        print(" -> Detected Standard Transformer")
        is_lora = False
    model.to(CONFIG['device'])
    model.eval()
    return model, tokenizer, is_lora


def extract_embeddings(model, tokenizer, texts: list[str], is_lora: bool = False) -> np.ndarray:
    embeddings = []

    for i in tqdm(range(0, len(texts), CONFIG['batch_size']), desc="Extracting"):
        batch_texts = texts[i:i + CONFIG['batch_size']]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=CONFIG['max_len'],
            return_tensors="pt"
        ).to(CONFIG['device'])
        with torch.no_grad():
            outputs = model(**inputs)
            if is_lora:
                emb = outputs.hidden_states[-1][:, 0, :]
            else:
                emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def load_izard_dataset():
    ds = load_dataset("Djacon/ru-izard-emotions")
    def convert_to_label_list(example):
        labels = []
        for i, label_name in enumerate(IZARD_EMOTIONS_LABELS):
            if example[label_name] == 1:
                labels.append(i)
        example['labels'] = labels

        return example
    ds = ds.map(convert_to_label_list)

    return ds


def evaluate_with_label_mapping(model, tokenizer, is_lora: bool, target_dataset, source_mlb, run_name: str ) -> dict:
    X_test = extract_embeddings(
        model, tokenizer,
        target_dataset['test']['text'],
        is_lora
    )
    izard_mlb = MultiLabelBinarizer(classes=list(range(len(IZARD_EMOTIONS_LABELS))))
    izard_mlb.fit([list(range(len(IZARD_EMOTIONS_LABELS)))])
    y_true_izard = izard_mlb.transform(target_dataset['test']['labels'])
    if is_lora:
        all_logits = []
        for i in tqdm(range(0, len(target_dataset['test']['text']), CONFIG['batch_size']), desc="Predicting"):
            batch_texts = target_dataset['test']['text'][i:i + CONFIG['batch_size']]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=CONFIG['max_len'],
                return_tensors="pt"
            ).to(CONFIG['device'])
            with torch.no_grad():
                outputs = model(**inputs)
                all_logits.append(outputs.logits.cpu())
        logits_go = torch.cat(all_logits).numpy()
        probs_go = 1 / (1 + np.exp(-logits_go))
        mapping_matrix = get_mapping_matrix()
        probs_izard = np.zeros((len(probs_go), len(IZARD_EMOTIONS_LABELS)))
        for i in range(len(IZARD_EMOTIONS_LABELS)):
            go_indices = np.where(mapping_matrix[:, i] == 1)[0]
            if len(go_indices) > 0:
                probs_izard[:, i] = np.max(probs_go[:, go_indices], axis=1)
        y_pred = (probs_izard > 0.5).astype(int)
        y_scores = probs_izard
    else:
        return {"error": "Base model requires classifier head"}
    metrics = {
        "f1_micro": f1_score(y_true_izard, y_pred, average='micro'),
        "f1_macro": f1_score(y_true_izard, y_pred, average='macro'),
        "accuracy": accuracy_score(y_true_izard, y_pred),
    }
    metrics_at_k = compute_all_metrics_at_k(y_true_izard, y_scores, k_values=[1, 3, 5])
    metrics.update(metrics_at_k)
    return metrics


def train_and_evaluate_cs_svm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, encoder_name: str) -> dict:
    print(f"\n  Training CS-SVM for {encoder_name}...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    class_counts = y_train_t.sum(dim=0).to(CONFIG['device'])
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    svm = MultilabelCSSVM(
        input_dim=input_dim,
        num_classes=num_classes,
        class_counts=class_counts,
        total_samples=len(y_train),
        C_reg=CONFIG['C_reg'],
        C_neg_base=CONFIG['C_neg_base']
    ).to(CONFIG['device'])
    optimizer = CustomAdamW(svm.parameters(), lr=CONFIG['svm_lr'], weight_decay=0.01)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=CONFIG['batch_size_svm'],
        shuffle=True
    )
    svm.train()
    for epoch in range(CONFIG['svm_epochs']):
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
            optimizer.zero_grad()
            loss = svm.compute_loss(bx, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    svm.eval()
    with torch.no_grad():
        logits = svm(X_test_t.to(CONFIG['device'])).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
    mapping_matrix = get_mapping_matrix()
    probs_izard = np.zeros((len(probs), len(IZARD_EMOTIONS_LABELS)))
    for i in range(len(IZARD_EMOTIONS_LABELS)):
        go_indices = np.where(mapping_matrix[:, i] == 1)[0]
        if len(go_indices) > 0:
            probs_izard[:, i] = np.max(probs[:, go_indices], axis=1)
    y_pred = (probs_izard > 0.5).astype(int)
    metrics = {
        "f1_micro": f1_score(y_test, y_pred, average='micro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    metrics_at_k = compute_all_metrics_at_k(y_test, probs_izard, k_values=CONFIG["k_values"])
    metrics.update(metrics_at_k)
    print(f"    CS-SVM: F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}")

    return metrics

def train_and_evaluate_sklearn_baselines(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, encoder_name: str ) -> dict:
    print(f"\n  Training sklearn baselines for {encoder_name}...")
    if CONFIG["use_scaler"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    classifiers = []
    for C in CONFIG["sklearn_svm_C_values"]:
        classifiers.append((f"LinearSVC_C{C}", LinearSVC(C=C, max_iter=10000, dual='auto')))
    classifiers.extend([
        ("SGD_hinge", SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, n_jobs=-1, random_state=42)),
        ("SGD_log", SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, n_jobs=-1, random_state=42)),
    ])
    all_results = {}
    mapping_matrix = get_mapping_matrix()
    for clf_name, base_clf in classifiers:
        try:
            clf = OneVsRestClassifier(base_clf, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)
            try:
                scores = clf.decision_function(X_test_scaled)
            except AttributeError:
                try:
                    scores = clf.predict_proba(X_test_scaled)
                except AttributeError:
                    scores = clf.predict(X_test_scaled).astype(float)
            scores_izard = np.zeros((len(scores), len(IZARD_EMOTIONS_LABELS)))
            for i in range(len(IZARD_EMOTIONS_LABELS)):
                go_indices = np.where(mapping_matrix[:, i] == 1)[0]
                if len(go_indices) > 0:
                    scores_izard[:, i] = np.max(scores[:, go_indices], axis=1)
            y_pred = (scores_izard > 0).astype(int)
            metrics = {
                "f1_micro": f1_score(y_test, y_pred, average='micro'),
                "f1_macro": f1_score(y_test, y_pred, average='macro'),
                "accuracy": accuracy_score(y_test, y_pred),
            }
            metrics_at_k = compute_all_metrics_at_k(y_test, scores_izard, k_values=CONFIG["k_values"])
            metrics.update(metrics_at_k)
            all_results[clf_name] = metrics
            print(f"    {clf_name}: F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
        except Exception as e:
            print(f"    Error training {clf_name}: {e}")
            continue

    return all_results


def analyze_drift_for_model(model_name: str, model_path: str, source_ds, target_ds, source_mlb, izard_mlb ):
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    model, tokenizer, is_lora = load_encoder(model_path)
    with mlflow.start_run(run_name=f"drift_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("source_dataset", CONFIG["source_dataset"])
        mlflow.log_param("target_dataset", CONFIG["target_dataset"])
        mlflow.log_param("is_lora", is_lora)
        print("\nExtracting source train embeddings...")
        X_source_train = extract_embeddings(model, tokenizer, source_ds['train']['text'], is_lora)
        y_source_train = source_mlb.transform(source_ds['train']['labels'])
        print("\nExtracting target train embeddings (for drift analysis)...")
        X_target_train = extract_embeddings(model, tokenizer, target_ds['train']['text'], is_lora)
        print("\nExtracting target test embeddings...")
        X_target_test = extract_embeddings(model, tokenizer, target_ds['test']['text'], is_lora)
        y_target_test = izard_mlb.transform(target_ds['test']['labels'])
        print("\nComputing distribution metrics...")
        mmd = compute_mmd(X_source_train, X_target_train)
        mlflow.log_metric("mmd", mmd)
        print(f"  MMD: {mmd:.6f}")
        wasserstein_stats = compute_wasserstein_per_dim(X_source_train, X_target_train)
        mlflow.log_metrics(wasserstein_stats)
        print(f"  Wasserstein (mean): {wasserstein_stats['wasserstein_mean']:.6f}")
        cosine_stats = compute_cosine_similarity_stats(X_source_train, X_target_train)
        mlflow.log_metrics(cosine_stats)
        print(f"  Cross-dataset cosine sim: {cosine_stats['cross_cosine_mean']:.4f}")
        source_centroid = np.mean(X_source_train, axis=0)
        target_centroid = np.mean(X_target_train, axis=0)
        centroid_distance = np.linalg.norm(source_centroid - target_centroid)
        mlflow.log_metric("centroid_distance", centroid_distance)
        print(f"  Centroid distance: {centroid_distance:.4f}")
        drift_report = {
            "model_name": model_name,
            "mmd": float(mmd),
            "wasserstein": wasserstein_stats,
            "cosine_similarity": cosine_stats,
            "centroid_distance": float(centroid_distance),
            "source_samples": len(X_source_train),
            "target_samples": len(X_target_train),
            "svm_results": {}
        }
        if is_lora:
            print("\nEvaluating LoRA model directly on target dataset...")
            eval_metrics = evaluate_with_label_mapping(
                model, tokenizer, is_lora,
                target_ds, source_mlb, model_name
            )
            if "error" not in eval_metrics:
                for k, v in eval_metrics.items():
                    mlflow.log_metric(f"lora_direct_{k}", v)
                drift_report["lora_direct_metrics"] = eval_metrics
                print(f"  LoRA Direct: F1_micro={eval_metrics['f1_micro']:.4f}, F1_macro={eval_metrics['f1_macro']:.4f}")
        print("\n--- Training CS-SVM on source, evaluating on target ---")
        cs_svm_metrics = train_and_evaluate_cs_svm(
            X_source_train, y_source_train,
            X_target_test, y_target_test,
            model_name
        )
        for k, v in cs_svm_metrics.items():
            mlflow.log_metric(f"cs_svm_{k}", v)
        drift_report["svm_results"]["CS_SVM"] = cs_svm_metrics
        print("\n--- Training sklearn baselines on source, evaluating on target ---")
        sklearn_results = train_and_evaluate_sklearn_baselines(
            X_source_train, y_source_train,
            X_target_test, y_target_test,
            model_name
        )
        for clf_name, metrics in sklearn_results.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{clf_name}_{k}", v)
            drift_report["svm_results"][clf_name] = metrics
        report_path = f"drift_report_{model_name}.json"
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=2, default=str)
        mlflow.log_artifact(report_path)
        print(f"\nDrift analysis for {model_name} completed.")
    del model, tokenizer
    torch.cuda.empty_cache()

    return drift_report


def main():
    print(f"Starting Data Drift Analysis")
    print(f"Device: {CONFIG['device']}")
    print(f"Source: {CONFIG['source_dataset']}")
    print(f"Target: {CONFIG['target_dataset']}")
    print(f"Models to analyze: {list(MODELS.keys())}")
    print("\nLoading datasets...")
    source_ds = load_dataset(CONFIG["source_dataset"], "simplified")
    target_ds = load_izard_dataset()
    print(f"Source dataset size: train={len(source_ds['train'])}, test={len(source_ds['test'])}")
    print(f"Target dataset size: train={len(target_ds['train'])}, test={len(target_ds['test'])}")
    all_labels = set()
    for labels in source_ds['train']['labels']:
        all_labels.update(labels)
    label_list = sorted(list(all_labels))
    source_mlb = MultiLabelBinarizer(classes=label_list)
    source_mlb.fit([label_list])
    print(f"Source labels: {len(label_list)} classes")
    izard_mlb = MultiLabelBinarizer(classes=list(range(len(IZARD_EMOTIONS_LABELS))))
    izard_mlb.fit([list(range(len(IZARD_EMOTIONS_LABELS)))])
    print(f"Target labels: {len(IZARD_EMOTIONS_LABELS)} classes ({IZARD_EMOTIONS_LABELS})")
    mlflow.set_experiment(CONFIG["experiment_name"])
    all_reports = {}
    for model_name, model_path in MODELS.items():
        report = analyze_drift_for_model(
            model_name, model_path,
            source_ds, target_ds, source_mlb, izard_mlb
        )
        all_reports[model_name] = report
    print("\n" + "="*100)
    print("SUMMARY - Data Drift Analysis")
    print("="*100)
    print("\n--- Distribution Drift Metrics ---")
    print(f"{'Model':<25} {'MMD':<12} {'Wasserstein':<12} {'Centroid Dist':<12}")
    print("-"*60)
    for model_name, report in all_reports.items():
        print(f"{model_name:<25} {report['mmd']:<12.6f} {report['wasserstein']['wasserstein_mean']:<12.6f} {report['centroid_distance']:<12.4f}")
    print("\n--- SVM Performance on Target Dataset (with label mapping) ---")
    print(f"{'Model + Classifier':<45} {'F1_micro':<10} {'F1_macro':<10} {'MAP@5':<10}")
    print("-"*80)
    for model_name, report in all_reports.items():
        if "svm_results" in report:
            for clf_name, metrics in report["svm_results"].items():
                full_name = f"{model_name}_{clf_name}"
                print(f"{full_name:<45} {metrics['f1_micro']:<10.4f} {metrics['f1_macro']:<10.4f} {metrics.get('map_at_5', 0):<10.4f}")
    with open("drift_analysis_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print("\nSummary saved to drift_analysis_summary.json")


if __name__ == "__main__":
    main()
