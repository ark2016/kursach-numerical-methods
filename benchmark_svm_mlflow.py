import os
import sys
import torch
import numpy as np
import mlflow
import warnings
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
import cvxopt
from cvxopt import matrix, solvers
import osqp
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import compute_all_metrics_at_k
from cvxopt import spmatrix

# Отключаем вывод cvxopt
solvers.options['show_progress'] = False

# --- КОНФИГУРАЦИЯ ---
CONFIG = {
    "dataset_name": "seara/ru_go_emotions",
    "batch_size_embed": 64,
    "max_len": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "./embeddings_cache",

    # CS-SVM параметры согласно статье (уравнения 49-51)
    # C - параметр регуляризации (slack penalty)
    # C_1 - стоимость ошибки на положительном классе (false negative)
    # C_{-1} - стоимость ошибки на отрицательном классе (false positive)
    # κ = 1/(2*C_{-1} - 1), условие: C_{-1} >= 1, C_1 >= 2*C_{-1} - 1
    "C_slack": 1.0,       # Параметр C (slack penalty)
    "C_pos": 3.0,         # C_1: стоимость FN (должно быть >= 2*C_neg - 1)
    "C_neg": 2.0,         # C_{-1}: стоимость FP (должно быть >= 1)

    # Baseline sklearn SVM
    "run_sklearn_baseline": False,  # Отключено - уже протестировано
    "sklearn_svm_C_values": [0.1, 1.0],  # Разные значения C для LinearSVC
    # ВАЖНО: Эмбеддинги уже нормализованы (L2 norm) на строке 1148!
    # StandardScaler поверх нормализованных эмбеддингов может навредить вычислению bias.
    # Попробуйте сначала с False, если получаете все F1=0.
    "use_scaler": False,  # StandardScaler перед SVM (было True - вызывало проблемы!)

    # Metrics @k settings
    "k_values": [1, 3, 5, 10],

    # MLFLOW Settings
    "mlflow_tracking_uri": "http://localhost:5000",
    "experiment_name": "Encoder_Comparison_CS_SVM_Primal",
    "s3_endpoint": "http://localhost:9000",
    "s3_access_key": "minio_root",
    "s3_secret_key": "minio_password"
}

# СПИСОК МОДЕЛЕЙ ДЛЯ СРАВНЕНИЯ
MODELS = {
    "Baseline_ruBert": "ai-forever/ruBert-large",
    "Foreign_ruRoberta": "fyaronskiy/ruRoberta-large-ru-go-emotions",
    "My_LoRA_Ark2016": "Ark2016/ruBert-large-emotions-lora",
    # Повторяющаяся модель (дубликат Foreign_ruRoberta)
    # "fyaronskiy/ruRoberta-large-ru-go-emotions": "fyaronskiy/ruRoberta-large-ru-go-emotions"
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

class BinaryCSSVM_Primal_Torch(torch.nn.Module):
    """
    Решает ПРЯМУЮ задачу CS-SVM (Equation 49 из статьи) через SGD/Adam на GPU.
    """

    def __init__(self, n_features, C_slack=1.0, C_pos=3.0, C_neg=2.0, 
                 device="cuda", epochs=500, lr=0.01):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        
        # Параметры из статьи
        self.kappa = 1.0 / (2 * C_neg - 1)
        
        # Коэффициенты штрафа (Equation 49)
        self.weight_pos = C_slack * C_pos
        self.weight_neg = C_slack / self.kappa
        
        # Параметры модели (w и b)
        # Инициализируем маленькими случайными числами
        self.w = torch.nn.Parameter(torch.randn(n_features, 1, device=device) * 0.01)
        self.b = torch.nn.Parameter(torch.zeros(1, device=device))

    def forward(self, x):
        return x @ self.w + self.b

    def fit(self, X, y):
        # Конвертация данных на GPU
        # Если пришел numpy array, конвертируем в тензор
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.to(dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.to(dtype=torch.float32)
            
        X = X.to(self.device)
        y = y.to(self.device).reshape(-1, 1) # {-1, 1}
        
        # Маски для векторизованного расчета Loss
        pos_mask = (y > 0)
        neg_mask = (y < 0)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(X)
            
            # --- Расчет Loss (строго по статье ур. 49) ---
            
            # 1. Hinge Loss для позитивных примеров (y=+1)
            # Constraint: ξ >= 1 - (w*x + b)
            loss_pos = torch.tensor(0.0, device=self.device)
            if pos_mask.any():
                out_pos = outputs[pos_mask]
                loss_pos = self.weight_pos * torch.sum(torch.relu(1.0 - out_pos))
            
            # 2. Hinge Loss для негативных примеров (y=-1)
            # Constraint: ξ >= (w*x + b) + kappa
            loss_neg = torch.tensor(0.0, device=self.device)
            if neg_mask.any():
                out_neg = outputs[neg_mask]
                loss_neg = self.weight_neg * torch.sum(torch.relu(out_neg + self.kappa))
            
            # 3. Регуляризация ||w||^2
            l2_reg = 0.5 * torch.sum(self.w ** 2)
            
            # Итоговый лосс
            loss = l2_reg + loss_pos + loss_neg
            
            loss.backward()
            optimizer.step()

        return self

    # --- Свойства для безопасного извлечения весов в NumPy ---
    @property
    def w_cpu(self):
        """Возвращает веса w как одномерный numpy array на CPU"""
        return self.w.detach().cpu().numpy().flatten()
    
    @property
    def b_cpu(self):
        """Возвращает смещение b как скаляр на CPU"""
        return self.b.detach().cpu().numpy().item()


class BinaryCSSVM_QP:
    """
    Бинарный Cost-Sensitive SVM с оптимизацией через двойственную задачу (QP).

    Реализация строго по статье "Cost-sensitive Support Vector Machines"
    (Masnadi-Shirazi et al., arXiv:1212.0975v2)

    Прималь (уравнение 49):
        min_{w,b,ξ} 1/2 ||w||² + C[C₁ Σ_{y_i=1} ξ_i + (1/κ) Σ_{y_i=-1} ξ_i]
        s.t. (w^T x_i + b) ≥ 1 - ξ_i    ; y_i = +1
             (w^T x_i + b) ≤ -κ + ξ_i   ; y_i = -1

    Дуаль (уравнение 51):
        max_α Σ_i α_i[(y_i+1)/2 - κ(y_i-1)/2] - 1/2 Σ_i Σ_j α_i α_j y_i y_j K(x_i,x_j)
        s.t. Σ_i α_i y_i = 0
             0 ≤ α_i ≤ C·C₁   ; y_i = +1
             0 ≤ α_i ≤ C/κ    ; y_i = -1

    где κ = 1/(2C_{-1} - 1), условия: C_{-1} ≥ 1, C₁ ≥ 2C_{-1} - 1
    """

    def __init__(self, C_slack=1.0, C_pos=3.0, C_neg=2.0):
        """
        Args:
            C_slack: Параметр регуляризации C (slack penalty)
            C_pos: C₁ - стоимость ошибки на положительном классе (false negative)
            C_neg: C_{-1} - стоимость ошибки на отрицательном классе (false positive)
        """
        # Проверка условий из статьи (уравнение 50)
        assert C_neg >= 1.0, f"C_neg должно быть >= 1, получено {C_neg}"
        min_c_pos = 2 * C_neg - 1
        assert C_pos >= min_c_pos, f"C_pos должно быть >= {min_c_pos}, получено {C_pos}"

        self.C = C_slack
        self.C_pos = C_pos  # C₁
        self.C_neg = C_neg  # C_{-1}

        # κ = 1/(2C_{-1} - 1) (уравнение 50)
        self.kappa = 1.0 / (2 * C_neg - 1)

        # Верхние границы для двойственных переменных (уравнение 51)
        self.alpha_upper_pos = C_slack * C_pos      # C·C₁ для y_i = +1
        self.alpha_upper_neg = C_slack / self.kappa  # C/κ для y_i = -1

        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None

    def fit(self, X, y):
        """
        Обучение CS-SVM через решение двойственной задачи квадратичного программирования.
        ИСПРАВЛЕНО: Использование разреженных матриц для G.
        """
        n_samples, n_features = X.shape

        # Конвертируем метки в {-1, +1}
        y = np.where(y > 0, 1, -1).astype(np.float64)

        # ----------------------------------------------------------------------
        # 1. Матрица P (Kernel Matrix * Labels)
        # P остается плотной. Для 43k сэмплов она займет 64+ ГБ RAM.
        # Если упадет с MemoryError, придется переходить на Primal SGD.
        # ----------------------------------------------------------------------
        K = np.dot(X, X.T)
        YY = np.outer(y, y)
        P = YY * K
        P = P + 1e-8 * np.eye(n_samples) # Численная стабильность
        
        P_cvx = matrix(P.astype(np.float64))
        
        # ----------------------------------------------------------------------
        # 2. Вектор q
        # ----------------------------------------------------------------------
        q = np.where(y > 0, 1.0, self.kappa)
        q_cvx = matrix(-q.astype(np.float64))

        # ----------------------------------------------------------------------
        # 3. Ограничения Gx <= h (Box constraints) через Sparse Matrix
        #
        # Нам нужно закодировать:
        #  -α_i <= 0      (для всех i) -> индексы 0..N-1
        #   α_i <= upper  (для всех i) -> индексы N..2N-1
        #
        # Матрица G (2N x N) будет иметь вид:
        # [-I ]
        # [ I ]
        # ----------------------------------------------------------------------
        
        # Значения: N раз по -1.0, затем N раз по 1.0
        values = [-1.0] * n_samples + [1.0] * n_samples
        
        # Индексы строк (I): 0, 1, ..., 2N-1
        rows = list(range(2 * n_samples))
        
        # Индексы столбцов (J): 0, 1, ..., N-1, затем снова 0, 1, ..., N-1
        cols = list(range(n_samples)) * 2
        
        # Создаем разреженную матрицу G
        G_cvx = spmatrix(values, rows, cols, (2 * n_samples, n_samples))

        # Вектор h (правая часть неравенств)
        h_lower = np.zeros(n_samples)
        h_upper = np.where(y > 0, self.alpha_upper_pos, self.alpha_upper_neg)
        h_combined = np.hstack([h_lower, h_upper])
        
        h_cvx = matrix(h_combined.astype(np.float64))

        # ----------------------------------------------------------------------
        # 4. Равенство A x = b (сумма alpha * y = 0)
        # ----------------------------------------------------------------------
        A_eq = matrix(y.reshape(1, -1).astype(np.float64))
        b_eq = matrix(np.zeros(1).astype(np.float64))

        # ----------------------------------------------------------------------
        # 5. Решение
        # ----------------------------------------------------------------------
        # Чистим память перед запуском солвера (удаляем тяжелые numpy массивы) иначе падает с превышением INT64
        del K, YY, P, values, rows, cols
        import gc
        gc.collect()
        
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_eq, b_eq)

        if solution['status'] != 'optimal':
            print(f"Warning: QP solver status = {solution['status']}")

        alphas = np.array(solution['x']).flatten()

        # Находим опорные векторы
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold

        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        # Вычисление w
        self.w = np.sum(
            (alphas * y).reshape(-1, 1) * X,
            axis=0
        )

        # Вычисление b (усреднение по свободным SV)
        b_values = []
        for i in range(n_samples):
            if sv_threshold < alphas[i]:
                upper_bound = self.alpha_upper_pos if y[i] > 0 else self.alpha_upper_neg
                if alphas[i] < upper_bound - sv_threshold:
                    wx = np.dot(self.w, X[i])
                    if y[i] > 0:
                        b_values.append(1.0 - wx)
                    else:
                        b_values.append(-self.kappa - wx)

        if len(b_values) > 0:
            self.b = np.mean(b_values)
        else:
            b_all = []
            for i in np.where(sv_indices)[0]:
                wx = np.dot(self.w, X[i])
                if y[i] > 0:
                    b_all.append(1.0 - wx)
                else:
                    b_all.append(-self.kappa - wx)
            self.b = np.mean(b_all) if b_all else 0.0

        return self


    def decision_function(self, X):
        """Вычисление значения решающей функции f(x) = w^T x + b"""
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """Предсказание класса: sign(f(x))"""
        return np.sign(self.decision_function(X))


class BinaryCSSVM_WSS:
    """
    Оптимизированная версия Binary CS-SVM с WSS.

    Улучшения:
    1. Векторизованный выбор working set
    2. Кэширование kernel computations
    3. Adaptive working set size
    4. Улучшенная стратегия shrinking
    5. Early stopping с patience
    """

    def __init__(self, C_slack=1.0, C_pos=3.0, C_neg=2.0,
                 working_set_size=200, max_iter=1000, tol=1e-3,
                 shrinking=True, verbose=False,
                 adaptive_ws=True, patience=10,
                 max_cache_size_mb=100):
        """
        Args:
            adaptive_ws: Адаптивно изменять размер working set
            patience: Количество итераций без улучшения для early stopping
            max_cache_size_mb: Максимальный размер кэша в мегабайтах
        """
        assert C_neg >= 1.0, f"C_neg должно быть >= 1, получено {C_neg}"
        min_c_pos = 2 * C_neg - 1
        assert C_pos >= min_c_pos, f"C_pos должно быть >= {min_c_pos}, получено {C_pos}"

        self.C = C_slack
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.kappa = 1.0 / (2 * C_neg - 1)

        self.alpha_upper_pos = C_slack * C_pos
        self.alpha_upper_neg = C_slack / self.kappa

        self.q = working_set_size
        self.max_iter = max_iter
        self.tol = tol
        self.shrinking = shrinking
        self.verbose = verbose
        self.adaptive_ws = adaptive_ws
        self.patience = patience
        self.max_cache_size_mb = max_cache_size_mb

        # Результаты
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None

        # Кэш для kernel вычислений
        self._kernel_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._current_cache_size_bytes = 0
        
    def fit(self, X, y):
        """Обучение через Working Set Selection с оптимизациями."""
        n_samples, n_features = X.shape
        
        # ДОБАВИТЬ: Проверка на вырожденные случаи
        if n_samples < 10:
            warnings.warn("Too few samples, falling back to simple QP solve")
            # Используем простое QP решение
            return self._fit_simple_qp(X, y)
        
        # Проверка на чистую линейную разделимость
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            raise ValueError("All samples have the same label!")
        
        y = np.where(y > 0, 1, -1).astype(np.float64)

        # Инициализация
        alphas = np.zeros(n_samples)
        C_upper = np.where(y > 0, self.alpha_upper_pos, self.alpha_upper_neg)
        q_vec = np.where(y > 0, 1.0, self.kappa)
        gradients = q_vec.copy()

        # Active set для shrinking
        active_set = np.arange(n_samples)

        # Early stopping
        best_violation = float('inf')
        patience_counter = 0

        # Adaptive working set size
        current_q = self.q

        if self.verbose:
            iterator = tqdm(range(self.max_iter), desc="WSS Optimization")
        else:
            iterator = range(self.max_iter)

        for iteration in iterator:
            # Shrinking с адаптивной частотой
            shrink_freq = 50 if iteration < 100 else 100
            if self.shrinking and iteration > 0 and iteration % shrink_freq == 0:
                old_size = len(active_set)
                active_set = self._apply_shrinking_vectorized(
                    alphas, gradients, y, C_upper
                )
                if self.verbose and old_size != len(active_set):
                    print(f"\nShrinking: {old_size} -> {len(active_set)} active")

            # Выбираем working set (векторизованная версия)
            B = self._select_working_set_vectorized(
                alphas, gradients, y, C_upper, active_set, current_q
            )

            if len(B) == 0:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

            # Решаем подзадачу
            alpha_B_old = alphas[B].copy()
            alphas[B] = self._solve_subproblem_cached(
                X, y, alphas, B, C_upper, gradients
            )

            # Обновляем градиенты (оптимизированная версия)
            delta_alpha = alphas[B] - alpha_B_old
            if np.max(np.abs(delta_alpha)) > 1e-10:
                self._update_gradients_efficient(
                    X, y, delta_alpha, B, active_set, gradients
                )

            # Проверка сходимости
            max_violation = self._compute_max_violation_vectorized(
                alphas[active_set], gradients[active_set],
                y[active_set], C_upper[active_set]
            )

            # Early stopping check - менее агрессивный
            # Требуем значительное улучшение (не просто любое)
            if max_violation < best_violation * 0.95:  # 5% улучшение
                best_violation = max_violation
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping только если действительно застряли И violation уже маленький
            if patience_counter >= self.patience and max_violation < self.tol * 10:
                if self.verbose:
                    print(f"\nEarly stopping at iteration {iteration} (violation={max_violation:.6f})")
                break

            # Adaptive working set size
            if self.adaptive_ws and iteration % 20 == 0:
                current_q = self._adapt_working_set_size(
                    max_violation, len(B), current_q
                )

            if self.verbose and iteration % 100 == 0:
                cache_rate = (self._cache_hits /
                            (self._cache_hits + self._cache_misses + 1e-8))
                print(f"Iter {iteration}: violation={max_violation:.6f}, "
                      f"|B|={len(B)}, q={current_q}, "
                      f"cache_hit={cache_rate:.2%}")

            if max_violation < self.tol:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

        # Финальная проверка (если использовали shrinking)
        if self.shrinking and len(active_set) < n_samples:
            self._final_check(X, y, alphas, gradients, q_vec, C_upper, active_set)

        # Сохраняем результаты
        self._finalize_solution(X, y, alphas, C_upper)

        if self.verbose:
            print(f"\nTraining complete:")
            print(f"  Support vectors: {len(self.alphas)} / {n_samples}")
            print(f"  Cache statistics: hits={self._cache_hits}, "
                  f"misses={self._cache_misses}")

        return self
    
    def _select_working_set_vectorized(self, alphas, gradients, y,
                                      C_upper, candidate_set, q):
        """
        Векторизованный выбор working set.

        ВАЖНО: Гарантируем наличие примеров обоих классов в working set,
        иначе equality constraint sum(alpha*y)=0 вынудит alpha=0!
        """
        if len(candidate_set) == 0:
            return np.array([], dtype=int)

        eps = 1e-8
        candidates = candidate_set

        # Векторизованное вычисление нарушений
        alpha_c = alphas[candidates]
        grad_c = gradients[candidates]
        y_c = y[candidates]
        C_c = C_upper[candidates]

        yg = y_c * grad_c

        # Маски для трёх случаев
        at_lower = alpha_c < eps
        at_upper = alpha_c > C_c - eps
        free = ~at_lower & ~at_upper

        # Вычисляем нарушения
        violations = np.zeros(len(candidates))
        violations[at_lower] = np.maximum(0, 1.0 - yg[at_lower])
        violations[at_upper] = np.maximum(0, yg[at_upper] - 1.0)
        violations[free] = np.abs(yg[free] - 1.0)

        # ИСПРАВЛЕНИЕ: Выбираем working set с балансом классов
        # Разделяем кандидатов на положительные и отрицательные
        pos_mask = y_c > 0
        neg_mask = y_c < 0

        pos_candidates = candidates[pos_mask]
        neg_candidates = candidates[neg_mask]
        pos_violations = violations[pos_mask]
        neg_violations = violations[neg_mask]

        # Выбираем top-q/2 из каждого класса (или все, если меньше)
        half_q = max(q // 2, 10)  # Минимум 10 из каждого класса

        selected = []

        # Выбираем из положительного класса
        if len(pos_candidates) > 0:
            n_pos = min(half_q, len(pos_candidates))
            if len(pos_candidates) <= n_pos:
                pos_selected_mask = pos_violations > self.tol / 10
                selected.append(pos_candidates[pos_selected_mask])
            else:
                threshold_idx = max(0, len(pos_violations) - n_pos)
                top_pos = np.argpartition(pos_violations, threshold_idx)[threshold_idx:]
                pos_selected_mask = pos_violations[top_pos] > self.tol / 10
                selected.append(pos_candidates[top_pos[pos_selected_mask]])

        # Выбираем из отрицательного класса
        if len(neg_candidates) > 0:
            n_neg = min(half_q, len(neg_candidates))
            if len(neg_candidates) <= n_neg:
                neg_selected_mask = neg_violations > self.tol / 10
                selected.append(neg_candidates[neg_selected_mask])
            else:
                threshold_idx = max(0, len(neg_violations) - n_neg)
                top_neg = np.argpartition(neg_violations, threshold_idx)[threshold_idx:]
                neg_selected_mask = neg_violations[top_neg] > self.tol / 10
                selected.append(neg_candidates[top_neg[neg_selected_mask]])

        if len(selected) == 0:
            return np.array([], dtype=int)

        return np.concatenate(selected)
    
    def _solve_subproblem_cached(self, X, y, alphas, B, C_upper, gradients):
        """Решение подзадачи с кэшированием kernel matrix."""
        q = len(B)
        if q == 0:
            return alphas[B]
        
        # Для q=1 тоже используем полное QP решение, т.к. аналитическое
        # решение не учитывает equality constraint корректно
        
        # Проверяем кэш для kernel matrix
        B_tuple = tuple(sorted(B))
        if B_tuple in self._kernel_cache:
            K_BB = self._kernel_cache[B_tuple]
            self._cache_hits += 1
        else:
            X_B = X[B]
            K_BB = X_B @ X_B.T
            
            # Вычисляем размер в байтах
            matrix_size_bytes = K_BB.nbytes
            max_cache_bytes = self.max_cache_size_mb * 1024 * 1024
            
            # Освобождаем память если нужно (FIFO)
            while (self._current_cache_size_bytes + matrix_size_bytes > max_cache_bytes 
                   and len(self._kernel_cache) > 0):
                # Удаляем первый (старейший) элемент
                old_key = next(iter(self._kernel_cache))
                old_matrix = self._kernel_cache.pop(old_key)
                self._current_cache_size_bytes -= old_matrix.nbytes
            
            # Добавляем новый
            if matrix_size_bytes < max_cache_bytes:
                self._kernel_cache[B_tuple] = K_BB
                self._current_cache_size_bytes += matrix_size_bytes
            
            self._cache_misses += 1

        y_B = y[B]
        Q_BB = (y_B.reshape(-1, 1) * y_B) * K_BB

        # Улучшенная регуляризация для численной стабильности
        min_eig = np.min(np.linalg.eigvalsh(Q_BB))
        reg_eps = max(1e-8, -min_eig + 1e-6) if min_eig < 0 else 1e-8
        Q_BB_reg = Q_BB + reg_eps * np.eye(q)

        # ИСПРАВЛЕНО: Правильный линейный коэффициент для QP подзадачи
        #
        # Двойственная задача: min_α 1/2 α^T Q α - q^T α
        # где Q_ij = y_i y_j K_ij, q_i = 1 (y=+1) или κ (y=-1)
        #
        # Градиент: ∇f(α) = Q α - q
        #
        # Код хранит: gradients[i] = q_i - (Q @ α)_i
        # Следовательно: (Q @ α)_i = q_i - gradients_i  (БЕЗ умножения на y!)
        #
        # Для подзадачи с фиксированным α_notB:
        # p = Q_{B,notB} @ α_notB - q_B
        #   = (Q @ α)_B - Q_BB @ α_B - q_B
        #   = (q_B - gradients_B) - Q_BB @ α_B - q_B
        #   = -gradients_B - Q_BB @ α_B

        p = -gradients[B] - Q_BB @ alphas[B]

        # ВАЖНО: правильное вычисление c для equality constraint
        # Constraint: sum(alpha[B] * y[B]) = -sum(alpha[not B] * y[not B])
        all_indices = np.arange(len(alphas))
        not_B = np.setdiff1d(all_indices, B)
        c = -(alphas[not_B] @ y[not_B]) if len(not_B) > 0 else 0.0

        l_box = np.zeros(q)
        u_box = C_upper[B]

        # OSQP setup с улучшенными настройками
        # ИСПРАВЛЕНО: A должна включать equality constraint И box constraints!
        # Формулировка OSQP: l <= A*x <= u
        # A = [y_B^T]  - equality constraint (1 row)
        #     [I_q]    - identity для box constraints (q rows)
        try:
            P_sparse = sparse.csc_matrix(Q_BB_reg)

            # Строим матрицу A: сначала equality constraint, потом identity для box
            A_eq = sparse.csc_matrix(y_B.reshape(1, -1))  # (1, q)
            A_box = sparse.eye(q, format='csc')            # (q, q)
            A_sparse = sparse.vstack([A_eq, A_box], format='csc')  # (1+q, q)

            # Границы: [c] для equality, [0, C_upper] для box
            l_constraints = np.concatenate([[c], l_box])   # (1+q,)
            u_constraints = np.concatenate([[c], u_box])   # (1+q,)
            
            m = osqp.OSQP()
            m.setup(
                P=P_sparse, q=p, A=A_sparse,
                l=l_constraints, u=u_constraints,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                polish=True,
                adaptive_rho=True,  # Улучшает сходимость
                check_termination=25  # Проверяем сходимость чаще
            )
            
            results = m.solve()
            
            if results.info.status != 'solved' and results.info.status != 'solved_inaccurate':
                if self.verbose:
                    warnings.warn(f"OSQP status: {results.info.status} for |B|={q}")
                return alphas[B]  # Возвращаем старые значения
            
            alpha_B_new = np.clip(results.x, 0, C_upper[B])

            # Проверка на NaN/inf
            if not np.all(np.isfinite(alpha_B_new)):
                if self.verbose:
                    warnings.warn(f"OSQP returned NaN/inf, returning old alphas")
                return alphas[B]

            # Проверка: насколько изменились alphas
            change = np.linalg.norm(alpha_B_new - alphas[B])
            if change < 1e-10:
                return alphas[B]  # Не обновляем если изменения минимальны

            return alpha_B_new
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"OSQP error: {e}, returning old alphas")
            return alphas[B]
    
    def _apply_shrinking_vectorized(self, alphas, gradients, y, C_upper):
        """Векторизованная версия shrinking."""
        eps = 1e-6
        margin = 0.1

        yg = y * gradients

        # Маски для трёх случаев
        at_lower = alphas < eps
        at_upper = alphas > C_upper - eps
        free = ~at_lower & ~at_upper

        # Условия активности
        active_lower = at_lower & (yg < 1.0 + margin)
        active_upper = at_upper & (yg > 1.0 - margin)

        # Объединяем
        active_mask = active_lower | active_upper | free
        active_indices = np.where(active_mask)[0]

        return active_indices if len(active_indices) > 0 else np.arange(len(alphas))
    
    def _compute_max_violation_vectorized(self, alphas, gradients, y, C_upper):
        """Векторизованное вычисление максимального нарушения KKT."""
        if len(alphas) == 0:
            return 0.0

        eps = 1e-8
        yg = y * gradients

        # Маски
        at_lower = alphas < eps
        at_upper = alphas > C_upper - eps
        free = ~at_lower & ~at_upper

        # Нарушения
        violations = np.zeros(len(alphas))
        violations[at_lower] = np.maximum(0, 1.0 - yg[at_lower])
        violations[at_upper] = np.maximum(0, yg[at_upper] - 1.0)
        violations[free] = np.abs(yg[free] - 1.0)

        return np.max(violations)

    def _update_gradients_efficient(self, X, y, delta_alpha, B,
                                   active_set, gradients):
        """Эффективное обновление градиентов."""
        # Вычисляем только для активных переменных
        # gradients[active] -= X[active] @ X[B].T @ (delta_alpha * y[B]) * y[active]

        # Оптимизация: сначала X[B].T @ (delta_alpha * y[B])
        temp = X[B].T @ (delta_alpha * y[B])  # (n_features,)

        # Затем X[active] @ temp
        kernel_delta = X[active_set] @ temp  # (n_active,)

        # Обновляем градиенты
        gradients[active_set] -= kernel_delta * y[active_set]

    def _adapt_working_set_size(self, max_violation, current_B_size, current_q):
        """Адаптивно изменяет размер working set с более плавными изменениями."""
        
        # Более плавная адаптация на основе violation
        if max_violation > 0.1:
            # Большие нарушения - агрессивно увеличиваем
            new_q = min(int(current_q * 1.25), 500)
        elif max_violation > 0.05:
            # Средние нарушения - умеренное увеличение
            new_q = min(current_q + 25, 500)
        elif max_violation < 0.005:
            # Очень малые нарушения - агрессивно уменьшаем
            new_q = max(int(current_q * 0.75), 50)
        elif max_violation < 0.01:
            # Малые нарушения - умеренное уменьшение
            new_q = max(current_q - 25, 100)
        else:
            new_q = current_q
        
        # Проверка эффективности: если working set недоиспользован
        if current_B_size < current_q * 0.2 and current_q > 100:
            # Слишком много переменных не нарушают KKT
            new_q = max(int(current_q * 0.8), 100)
        
        # Ограничиваем скорость изменения
        max_change = 100
        if abs(new_q - current_q) > max_change:
            new_q = current_q + max_change if new_q > current_q else current_q - max_change
        
        return new_q

    def _final_check(self, X, y, alphas, gradients, q_vec, C_upper, active_set):
        """Финальная проверка всех переменных после shrinking."""
        n_samples = len(alphas)

        # Пересчитываем градиенты для всех
        K_alpha_y = X @ (X.T @ (alphas * y))
        gradients[:] = q_vec - K_alpha_y * y

        # Проверяем неактивные переменные
        inactive_set = np.setdiff1d(np.arange(n_samples), active_set)

        if len(inactive_set) > 0:
            for _ in range(10):
                B = self._select_working_set_vectorized(
                    alphas, gradients, y, C_upper, inactive_set, self.q
                )
                if len(B) == 0:
                    break

                alpha_B_old = alphas[B].copy()
                alphas[B] = self._solve_subproblem_cached(
                    X, y, alphas, B, C_upper, gradients
                )

                delta_alpha = alphas[B] - alpha_B_old
                if np.max(np.abs(delta_alpha)) > 1e-10:
                    temp = X[B].T @ (delta_alpha * y[B])
                    kernel_delta = X @ temp
                    gradients -= kernel_delta * y

    def _fit_simple_qp(self, X, y):
        """Простое QP решение для малых наборов данных."""
        # Используем стандартный QP solver для малых задач
        simple_svm = BinaryCSSVM_QP(
            C_slack=self.C,
            C_pos=self.C_pos,
            C_neg=self.C_neg
        )
        simple_svm.fit(X, y)
        
        # Копируем результаты
        self.w = simple_svm.w
        self.b = simple_svm.b
        self.support_vectors = simple_svm.support_vectors
        self.support_vector_labels = simple_svm.support_vector_labels
        self.alphas = simple_svm.alphas
        
        return self

    def _finalize_solution(self, X, y, alphas, C_upper):
        """Сохраняет финальное решение с валидацией."""
        sv_threshold = 1e-5
        sv_indices = alphas > sv_threshold

        # ДОБАВИТЬ: Проверка количества SV
        n_sv = np.sum(sv_indices)
        if n_sv == 0:
            warnings.warn("No support vectors found! Model may be degenerate.")
            # Fallback: используем все точки
            sv_indices = np.ones(len(alphas), dtype=bool)

        if self.verbose:
            print(f"  Support vectors: {n_sv} / {len(alphas)} ({100*n_sv/len(alphas):.1f}%)")

        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        # Вычисляем w
        self.w = np.sum((alphas * y).reshape(-1, 1) * X, axis=0)

        # ДОБАВИТЬ: Проверка нормы w
        w_norm = np.linalg.norm(self.w)
        if w_norm < 1e-10:
            warnings.warn("Weight vector has near-zero norm! Model may be degenerate.")

        # Вычисляем b
        self.b = self._compute_bias(X, y, alphas, C_upper)
        
        # ДОБАВИТЬ: Финальная проверка KKT на train set
        if self.verbose:
            train_violations = self._compute_kkt_violations(X, y, alphas, C_upper)
            print(f"  Final KKT violations: max={np.max(train_violations):.6f}, "
                  f"mean={np.mean(train_violations):.6f}")

    def _compute_kkt_violations(self, X, y, alphas, C_upper):
        """Вычисляет нарушения KKT для диагностики."""
        eps = 1e-8
        
        # Вычисляем градиенты
        q_vec = np.where(y > 0, 1.0, self.kappa)
        K_alpha_y = X @ (X.T @ (alphas * y))
        gradients = q_vec - K_alpha_y * y
        
        yg = y * gradients
        
        violations = np.zeros(len(alphas))
        at_lower = alphas < eps
        at_upper = alphas > C_upper - eps
        free = ~at_lower & ~at_upper
        
        violations[at_lower] = np.maximum(0, 1.0 - yg[at_lower])
        violations[at_upper] = np.maximum(0, yg[at_upper] - 1.0)
        violations[free] = np.abs(yg[free] - 1.0)
        
        return violations

    def _compute_bias(self, X, y, alphas, C_upper):
        """
        Вычисляет bias используя free support vectors.

        Упрощенная версия с явным циклом (более надежная чем векторизация с np.nan).
        Для free SV: y_i * (w^T x_i + b) = 1
        => b = y_i - w^T x_i  (если y_i = +1)
        => b = -κ - w^T x_i   (если y_i = -1, так как w^T x_i + b = -κ)
        """
        b_estimates = []
        eps = 1e-5

        # Сначала пробуем free support vectors
        free_sv_mask = (alphas > eps) & (alphas < C_upper - eps)
        free_sv_indices = np.where(free_sv_mask)[0]

        if len(free_sv_indices) > 0:
            for i in free_sv_indices:
                wx = np.dot(self.w, X[i])
                if y[i] > 0:
                    b_estimates.append(1.0 - wx)
                else:  # y[i] < 0
                    b_estimates.append(-self.kappa - wx)

            return np.mean(b_estimates)

        # Fallback: используем все support vectors (boundary SVs)
        sv_indices = np.where(alphas > eps)[0]
        if len(sv_indices) == 0:
            if self.verbose:
                warnings.warn("No support vectors found! Bias set to 0.0")
            return 0.0

        for i in sv_indices:
            wx = np.dot(self.w, X[i])
            if y[i] > 0:
                b_estimates.append(1.0 - wx)
            else:  # y[i] < 0
                b_estimates.append(-self.kappa - wx)

        return np.mean(b_estimates) if len(b_estimates) > 0 else 0.0
    
    def decision_function(self, X):
        """Вычисление решающей функции f(x) = w^T x + b"""
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """Предсказание класса: sign(f(x))"""
        return np.sign(self.decision_function(X))


class MultilabelCSSVM_WSS:
    """Оптимизированная multilabel версия с улучшениями."""

    def __init__(self, num_classes, class_counts, total_samples,
                 C_slack=1.0, C_pos_base=3.0, C_neg_base=2.0,
                 working_set_size=200, max_iter=1000, verbose=False,
                 adaptive_ws=True, patience=10, max_cache_size_mb=100):
        """
        Args:
            adaptive_ws: Адаптивно изменять размер working set
            patience: Количество итераций без улучшения для early stopping
            max_cache_size_mb: Максимальный размер кэша в мегабайтах
        """
        self.num_classes = num_classes
        self.C_slack = C_slack
        self.C_neg_base = C_neg_base

        # κ для CS-SVM
        self.kappa = 1.0 / (2 * C_neg_base - 1)
        # Порог для предсказания: середина между +1 и -κ
        self.threshold = (1.0 - self.kappa) / 2.0
        self.working_set_size = working_set_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.adaptive_ws = adaptive_ws
        self.patience = patience
        self.max_cache_size_mb = max_cache_size_mb

        # Адаптивное вычисление C_pos
        neg_counts = total_samples - class_counts
        imbalance_ratios = neg_counts / (class_counts + 1e-6)
        c_pos_calculated = imbalance_ratios * C_neg_base
        min_c_pos = 2 * C_neg_base - 1
        self.C_pos_per_class = np.clip(c_pos_calculated, min_c_pos, 50.0)

        self.classifiers = []
        self.w = None
        self.b = None

    def fit(self, X, Y):
        """Обучение One-vs-Rest классификаторов."""
        n_samples, n_features = X.shape

        self.w = np.zeros((n_features, self.num_classes))
        self.b = np.zeros(self.num_classes)
        self.classifiers = []

        # Обучаем классификаторы
        for c in tqdm(range(self.num_classes), desc="Training CS-SVM (WSS Optimized)"):
            y_binary = np.where(Y[:, c] > 0, 1, -1)

            clf = BinaryCSSVM_WSS(
                C_slack=self.C_slack,
                C_pos=self.C_pos_per_class[c],
                C_neg=self.C_neg_base,
                working_set_size=self.working_set_size,
                max_iter=self.max_iter,
                tol=5e-3,  # Ослаблено с 1e-3 для более быстрой и надежной сходимости
                shrinking=False,  # Отключено для надежной сходимости
                verbose=self.verbose,
                adaptive_ws=False,  # Отключено для стабильности
                patience=self.patience,
                max_cache_size_mb=self.max_cache_size_mb
            )

            clf.fit(X, y_binary)
            self.classifiers.append(clf)

            self.w[:, c] = clf.w
            self.b[c] = clf.b

            # ДИАГНОСТИКА: логируем параметры каждого класса
            if self.verbose:
                n_pos = np.sum(y_binary > 0)
                n_neg = np.sum(y_binary < 0)
                w_norm = np.linalg.norm(clf.w)
                print(f"  Class {c}: pos={n_pos}, neg={n_neg}, b={clf.b:.6f}, ||w||={w_norm:.6f}, n_sv={len(clf.alphas)}")

        return self
    
    def decision_function(self, X):
        """Вычисление решающей функции для всех классов"""
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        Бинарное предсказание с правильным порогом для CS-SVM.

        Для CS-SVM:
        - Положительный класс: w^T x + b >= 1
        - Отрицательный класс: w^T x + b <= -κ
        - Порог: (1 - κ) / 2
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(np.float32)

    def predict_proba(self, X):
        """Возвращает scores"""
        return self.decision_function(X)


# class MultilabelCSSVM_QP:
#     """
#     Multilabel CS-SVM с One-vs-Rest стратегией.
#     Использует Primal-форму на GPU для скорости.
#     """
#
#     def __init__(self, num_classes, class_counts, total_samples,
#                  C_slack=1.0, C_pos_base=3.0, C_neg_base=2.0):
#         self.num_classes = num_classes
#         self.C_slack = C_slack
#         self.C_neg_base = C_neg_base
#
#         # Адаптивное вычисление C_pos для каждого класса на основе дисбаланса
#         neg_counts = total_samples - class_counts
#         imbalance_ratios = neg_counts / (class_counts + 1e-6)
#         c_pos_calculated = imbalance_ratios * C_neg_base
#         min_c_pos = 2 * C_neg_base - 1  # Ограничение из статьи
#         self.C_pos_per_class = np.clip(c_pos_calculated, min_c_pos, 50.0)
#
#         self.classifiers = []
#         self.w = None
#         self.b = None
#
#     def fit(self, X, Y):
#         """
#         Обучение One-vs-Rest классификаторов.
#         """
#         n_samples, n_features = X.shape
#         # Инициализируем массивы весов (NumPy)
#         self.w = np.zeros((n_features, self.num_classes))
#         self.b = np.zeros(self.num_classes)
#         self.classifiers = []
#
#         # Progress bar для отслеживания обучения классов
#         for c in tqdm(range(self.num_classes), desc="Training CS-SVM Primal (GPU)"):
#             # Преобразуем метки: 1 -> +1, 0 -> -1
#             y_binary = np.where(Y[:, c] > 0, 1, -1)
#
#             # Используем Torch реализацию (Primal form)
#             clf = BinaryCSSVM_Primal_Torch(
#                 n_features=n_features,
#                 C_slack=self.C_slack,
#                 C_pos=self.C_pos_per_class[c],
#                 C_neg=self.C_neg_base,
#                 device=CONFIG["device"], 
#                 epochs=1000,             
#                 lr=0.005
#             )
#             # Создаём классификатор с адаптивной стоимостью для этого класса
#             # clf = BinaryCSSVM_QP( # работает слишком долго и требует 64+ Gb RAM
#             #     C_slack=self.C_slack,
#             #     C_pos=self.C_pos_per_class[c],
#             #     C_neg=self.C_neg_base
#             # )
#
#             clf.fit(X, y_binary)
#             self.classifiers.append(clf)
#
#             # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
#             # Берем веса через свойство w_cpu, которое делает .detach().cpu().numpy()
#             self.w[:, c] = clf.w_cpu
#             self.b[c] = clf.b_cpu
#
#         return self
#
#     def decision_function(self, X):
#         """Вычисление решающей функции для всех классов (на CPU)"""
#         return np.dot(X, self.w) + self.b
#
#     def predict(self, X):
#         """Бинарное предсказание (threshold = 0)"""
#         scores = self.decision_function(X)
#         return (scores > 0).astype(np.float32)
#
#     def predict_proba(self, X):
#         """Возвращает scores"""
#         return self.decision_function(X)


def load_encoder(model_path):
    """
    Загружает encoder модель и tokenizer.
    Обрабатывает LoRA модели и обычные модели.
    """
    try:
        # Пытаемся загрузить как LoRA модель
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=28,  # ru_go_emotions classes
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        print(f"Loaded LoRA model: {model_path}")
    except Exception as e:
        # Загружаем как обычную модель
        try:
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Loaded AutoModel: {model_path}")
        except Exception as e2:
            print(f"Error loading model: {e}, {e2}")
            raise
    
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
    Обучает sklearn baseline классификаторы.

    ВАЖНО: X_train и X_test должны быть уже отмасштабированы (если use_scaler=True),
    чтобы все эксперименты шли на равных условиях с CS-SVM.
    """
    print(f"\n--- Training sklearn baselines for {encoder_name} ---")
    X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
    y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

    # Данные уже отмасштабированы в main(), используем как есть
    X_train_scaled = X_train_np
    X_test_scaled = X_test_np

    classifiers = []

    for C in CONFIG["sklearn_svm_C_values"]:
        classifiers.append((
            f"LinearSVC_C{C}",
            LinearSVC(C=C, max_iter=10000, dual='auto')
        ))

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
            clf = OneVsRestClassifier(base_clf, n_jobs=-1)
            try:
                clf.fit(X_train_scaled, y_train_np)
                y_pred = clf.predict(X_test_scaled)
                try:
                    y_scores = clf.decision_function(X_test_scaled)
                except AttributeError:
                    try:
                        y_scores = clf.predict_proba(X_test_scaled)
                    except AttributeError:
                        y_scores = y_pred.astype(float)
                metrics = {
                    "f1_micro": f1_score(y_test_np, y_pred, average='micro'),
                    "f1_macro": f1_score(y_test_np, y_pred, average='macro'),
                    "f1_weighted": f1_score(y_test_np, y_pred, average='weighted'),
                    "accuracy": accuracy_score(y_test_np, y_pred)
                }
                metrics_at_k = compute_all_metrics_at_k(y_test_np, y_scores, k_values=CONFIG["k_values"])
                metrics.update(metrics_at_k)
                print(f"    {clf_name}: F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}, MAP@5={metrics.get('map_at_5', 0):.4f}")
                mlflow.log_metrics(metrics)
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
    """
    Главная функция бенчмарка.

    CS-SVM реализован строго по статье "Cost-sensitive Support Vector Machines"
    (Masnadi-Shirazi et al., arXiv:1212.0975v2) с использованием оптимизированного
    QP-solver через Working Set Selection (WSS) для решения двойственной задачи.
    """
    print(f"Start Benchmark. Device: {CONFIG['device']}")
    print("=" * 60)
    print("CS-SVM: Dual QP Optimization (Lagrange multipliers)")
    print(f"  C_slack = {CONFIG['C_slack']}")
    print(f"  C_pos (base) = {CONFIG['C_pos']}")
    print(f"  C_neg = {CONFIG['C_neg']}")
    kappa = 1.0 / (2 * CONFIG['C_neg'] - 1)
    print(f"  κ = 1/(2·C_neg - 1) = {kappa:.4f}")
    print("=" * 60)

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
        
        try:
            X_train, y_train, X_test, y_test = get_embeddings(model_friendly_name, model_path, ds, mlb)

            # Конвертируем в numpy
            X_train_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
            X_test_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
            y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
            y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

            # Применяем StandardScaler
            scaler = None
            if CONFIG["use_scaler"]:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_np)
                X_test_scaled = scaler.transform(X_test_np)
            else:
                X_train_scaled = X_train_np
                X_test_scaled = X_test_np

            with mlflow.start_run(run_name=f"CS_SVM_WSS_Optimized_on_{model_friendly_name}"):
                # Логируем параметры CS-SVM по статье
                mlflow.log_param("encoder_model", model_path)
                mlflow.log_param("optimization_method", "Dual_WSS_Optimized_Vectorized_Cached")
                mlflow.log_param("C_slack", CONFIG["C_slack"])
                mlflow.log_param("C_pos_base", CONFIG["C_pos"])
                mlflow.log_param("C_neg", CONFIG["C_neg"])
                mlflow.log_param("kappa", kappa)
                mlflow.log_param("use_scaler", CONFIG["use_scaler"])
                mlflow.log_param("adaptive_ws", True)
                mlflow.log_param("patience", 10)

                # Подсчёт примеров каждого класса для адаптивных весов
                class_counts = y_train_np.sum(axis=0)
                total_samples = len(y_train_np)

                print("Training CS-SVM via Dual WSS (Working Set Selection)...")
                print(f"  Dataset size: {total_samples} samples, {len(label_list)} classes")

                # Создаём и обучаем CS-SVM через WSS (Working Set Selection) с оптимизациями
                # ПАРАМЕТРЫ ДЛЯ ОТЛАДКИ:
                # - Если F1=0: попробуйте max_iter=2000-5000, tol=1e-2
                # - Если не сходится: попробуйте отключить shrinking=False, adaptive_ws=False
                # - Увеличьте patience=20 для более тщательной оптимизации
                svm = MultilabelCSSVM_WSS(
                    num_classes=len(label_list),
                    class_counts=class_counts,
                    total_samples=total_samples,
                    C_slack=CONFIG['C_slack'],
                    C_pos_base=CONFIG['C_pos'],
                    C_neg_base=CONFIG['C_neg'],
                    working_set_size=300,  # Увеличено для лучшего покрытия классов
                    max_iter=2000,
                    verbose=True,
                    adaptive_ws=True,
                    patience=50,  # Увеличено - теперь early stopping менее агрессивный
                    max_cache_size_mb=100
                )

                svm.fit(X_train_scaled, y_train_np)

                print("Evaluating CS-SVM...")
                # Предсказания
                y_scores = svm.decision_function(X_test_scaled)
                pred_bin = svm.predict(X_test_scaled)

                # ДИАГНОСТИКА: Проверяем распределение скоров
                print(f"  CS-SVM threshold: {svm.threshold:.4f} (κ={svm.kappa:.4f})")
                print(f"  Decision function distribution:")
                print(f"    Min: {np.min(y_scores):.6f}, Max: {np.max(y_scores):.6f}")
                print(f"    Mean: {np.mean(y_scores):.6f}, Std: {np.std(y_scores):.6f}")
                print(f"    Median: {np.median(y_scores):.6f}")
                print(f"    % scores > 0: {100 * np.mean(y_scores > 0):.2f}%")
                print(f"    % scores > threshold ({svm.threshold:.4f}): {100 * np.mean(y_scores > svm.threshold):.2f}%")
                print(f"  Bias (b) distribution:")
                print(f"    Min: {np.min(svm.b):.6f}, Max: {np.max(svm.b):.6f}")
                print(f"    Mean: {np.mean(svm.b):.6f}, Median: {np.median(svm.b):.6f}")
                print(f"  Predicted labels sum: {pred_bin.sum():.0f} (out of {pred_bin.size} total predictions)")

                # Метрики
                metrics = {
                    "f1_micro": f1_score(y_test_np, pred_bin, average='micro'),
                    "f1_macro": f1_score(y_test_np, pred_bin, average='macro'),
                    "f1_weighted": f1_score(y_test_np, pred_bin, average='weighted'),
                    "accuracy": accuracy_score(y_test_np, pred_bin)
                }
                metrics_at_k = compute_all_metrics_at_k(y_test_np, y_scores, k_values=CONFIG["k_values"])
                metrics.update(metrics_at_k)

                print(f"CS-SVM WSS Results for {model_friendly_name}:")
                print(f"  F1_micro={metrics['f1_micro']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
                print(f"  Precision@5={metrics.get('precision_at_5', 0):.4f}, MAP@5={metrics.get('map_at_5', 0):.4f}")
                print(f"  NDCG@5={metrics.get('ndcg_at_5', 0):.4f}, Hit_Rate@5={metrics.get('hit_rate_at_5', 0):.4f}")

                mlflow.log_metrics(metrics)

                # Сохраняем веса модели
                np.savez("svm_model_qp.npz", w=svm.w, b=svm.b)
                mlflow.log_artifact("svm_model_qp.npz")

                # Сохраняем отчёт классификации
                report = classification_report(y_test_np, pred_bin, target_names=target_names_str, zero_division=0)
                with open("classification_report.txt", "w", encoding="utf-8") as f:
                    f.write(f"Encoder: {model_friendly_name}\n")
                    f.write("Optimization: Dual WSS Optimized (Vectorized + Cached + Adaptive)\n")
                    f.write(f"Paper: Cost-sensitive Support Vector Machines (arXiv:1212.0975v2)\n\n")
                    f.write(f"Parameters:\n")
                    f.write(f"  C_slack = {CONFIG['C_slack']}\n")
                    f.write(f"  C_neg = {CONFIG['C_neg']}\n")
                    f.write(f"  κ = {kappa:.4f}\n")
                    f.write(f"  adaptive_ws = True\n")
                    f.write(f"  patience = 10\n\n")
                    f.write(report)
                mlflow.log_artifact("classification_report.txt")

                all_results[f"CS_SVM_WSS_{model_friendly_name}"] = metrics
                print(f"CS-SVM WSS run for {model_friendly_name} completed.")

            if CONFIG["run_sklearn_baseline"]:
                # ВАЖНО: Передаём УЖЕ отмасштабированные данные (X_train_scaled, X_test_scaled)
                # чтобы все эксперименты шли на равных условиях
                sklearn_results = train_sklearn_baselines(
                    X_train_scaled, y_train, X_test_scaled, y_test,
                    target_names_str, model_friendly_name
                )
                for clf_name, m in sklearn_results.items():
                    all_results[f"{clf_name}_{model_friendly_name}"] = m

        except Exception as e:
            print(f"Error processing {model_friendly_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("SUMMARY - All Results")
    print("="*80)
    print(f"{'Model':<45} {'F1_micro':<10} {'F1_macro':<10} {'MAP@5':<10} {'NDCG@5':<10}")
    print("-"*85)
    for name, m in all_results.items():
        print(f"{name:<45} {m['f1_micro']:<10.4f} {m['f1_macro']:<10.4f} {m.get('map_at_5', 0):<10.4f} {m.get('ndcg_at_5', 0):<10.4f}")

if __name__ == "__main__":
    main()