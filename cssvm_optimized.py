"""
Оптимизированная реализация Cost-Sensitive SVM с правильной dual оптимизацией.

Реализация строго по статье "Cost-sensitive Support Vector Machines"
(Masnadi-Shirazi et al., arXiv:1212.0975v2)

Основные улучшения:
1. Правильная dual оптимизация через QP-solver
2. Эффективное использование множителей Лагранжа
3. Оптимизированные вычисления
4. Правильная обработка KKT условий
"""

import numpy as np
import cvxopt
from cvxopt import matrix, solvers, spmatrix
from scipy import sparse
import warnings
from tqdm.auto import tqdm
import gc


class OptimizedCSSVM:
    """
    Оптимизированный Cost-Sensitive SVM с правильной dual оптимизацией.
    
    Реализует dual задачу (уравнение 51 из статьи) через QP-solver
    с учётом множителей Лагранжа.
    """
    
    def __init__(self, C_slack=1.0, C_pos=3.0, C_neg=2.0, 
                 use_wss=False, working_set_size=200, max_iter=1000, 
                 tol=1e-3, verbose=False):
        """
        Args:
            C_slack: Параметр регуляризации C (slack penalty)
            C_pos: C₁ - стоимость ошибки на положительном классе (false negative)
            C_neg: C_{-1} - стоимость ошибки на отрицательном классе (false positive)
            use_wss: Использовать Working Set Selection оптимизацию
            working_set_size: Размер рабочего набора для WSS
            max_iter: Максимальное количество итераций
            tol: Толерантность для сходимости
            verbose: Выводить информацию о процессе обучения
        """
        # Проверка условий из статьи (уравнение 50)
        assert C_neg >= 1.0, f"C_neg должно быть >= 1, получено {C_neg}"
        min_c_pos = 2 * C_neg - 1
        assert C_pos >= min_c_pos, f"C_pos должно быть >= {min_c_pos}, получено {C_pos}"

        self.C = C_slack
        self.C_pos = C_pos  # C₁
        self.C_neg = C_neg  # C_{-1}
        self.use_wss = use_wss
        self.q = working_set_size
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # κ = 1/(2C_{-1} - 1) (уравнение 50)
        self.kappa = 1.0 / (2 * C_neg - 1)

        # Верхние границы для двойственных переменных (уравнение 51)
        self.alpha_upper_pos = C_slack * C_pos      # C·C₁ для y_i = +1
        self.alpha_upper_neg = C_slack / self.kappa  # C/κ для y_i = -1

        # Результаты
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        
        # Статистика
        self.iterations = 0
        self.convergence_info = {}

    def fit(self, X, y):
        """
        Обучение CS-SVM через решение двойственной задачи.
        
        Args:
            X: Матрица признаков (n_samples, n_features)
            y: Метки классов (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Конвертируем метки в {-1, +1}
        y = np.where(y > 0, 1, -1).astype(np.float64)
        
        if self.use_wss:
            return self._fit_wss(X, y)
        else:
            return self._fit_qp(X, y)

    def _fit_qp(self, X, y):
        """
        Решение dual задачи через QP-solver (полная оптимизация).
        """
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Solving dual QP problem for {n_samples} samples...")
        
        # ----------------------------------------------------------------------
        # 1. Матрица P (Kernel Matrix * Labels)
        # P = YY * K, где YY = y * y^T, K = X * X^T
        # ----------------------------------------------------------------------
        K = np.dot(X, X.T)
        YY = np.outer(y, y)
        P = YY * K
        P = P + 1e-8 * np.eye(n_samples)  # Численная стабильность
        
        P_cvx = matrix(P.astype(np.float64))
        
        # ----------------------------------------------------------------------
        # 2. Вектор q (линейный член)
        # q_i = (y_i + 1)/2 - κ(y_i - 1)/2
        # ----------------------------------------------------------------------
        q = np.where(y > 0, 1.0, self.kappa)
        q_cvx = matrix(-q.astype(np.float64))  # Minimize 1/2 α^T P α - q^T α
        
        # ----------------------------------------------------------------------
        # 3. Ограничения Gx <= h (Box constraints)
        # -α_i <= 0      (для всех i)
        #  α_i <= upper  (для всех i)
        # ----------------------------------------------------------------------
        # Верхние границы для альфа
        C_upper = np.where(y > 0, self.alpha_upper_pos, self.alpha_upper_neg)
        
        # Создаем разреженную матрицу G
        values = [-1.0] * n_samples + [1.0] * n_samples
        rows = list(range(2 * n_samples))
        cols = list(range(n_samples)) * 2
        
        G_cvx = spmatrix(values, rows, cols, (2 * n_samples, n_samples))
        
        # Вектор h (правая часть неравенств)
        h_lower = np.zeros(n_samples)
        h_upper = C_upper
        h_combined = np.hstack([h_lower, h_upper])
        h_cvx = matrix(h_combined.astype(np.float64))
        
        # ----------------------------------------------------------------------
        # 4. Равенство A x = b (сумма alpha * y = 0)
        # ----------------------------------------------------------------------
        A_eq = matrix(y.reshape(1, -1).astype(np.float64))
        b_eq = matrix(np.zeros(1).astype(np.float64))
        
        # ----------------------------------------------------------------------
        # 5. Решение QP задачи
        # ----------------------------------------------------------------------
        # Чистим память перед запуском солвера
        del K, YY, P, values, rows, cols
        gc.collect()
        
        # Решаем QP задачу
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_eq, b_eq)
        
        if solution['status'] != 'optimal':
            warnings.warn(f"QP solver status = {solution['status']}")
        
        alphas = np.array(solution['x']).flatten()
        
        # ----------------------------------------------------------------------
        # 6. Вычисление w и b
        # ----------------------------------------------------------------------
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
            # Если нет свободных SV, используем все SV
            b_all = []
            for i in np.where(sv_indices)[0]:
                wx = np.dot(self.w, X[i])
                if y[i] > 0:
                    b_all.append(1.0 - wx)
                else:
                    b_all.append(-self.kappa - wx)
            self.b = np.mean(b_all) if b_all else 0.0
        
        # Сохраняем статистику
        self.iterations = 1
        self.convergence_info = {
            'status': solution['status'],
            'primal_objective': solution.get('primal objective', None),
            'dual_objective': solution.get('dual objective', None)
        }
        
        if self.verbose:
            print(f"QP solved in {self.iterations} iteration(s)")
            print(f"Support vectors: {len(self.alphas)} / {n_samples}")
        
        return self

    def _fit_wss(self, X, y):
        """
        Решение dual задачи через Working Set Selection.
        """
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Solving dual problem with WSS for {n_samples} samples...")
        
        # Инициализация
        alphas = np.zeros(n_samples)
        C_upper = np.where(y > 0, self.alpha_upper_pos, self.alpha_upper_neg)
        q_vec = np.where(y > 0, 1.0, self.kappa)
        gradients = q_vec.copy()
        
        # Active set
        active_set = np.arange(n_samples)
        
        # Статистика
        self._cache_hits = 0
        self._cache_misses = 0
        
        iterator = tqdm(range(self.max_iter), desc="WSS Optimization") if self.verbose else range(self.max_iter)
        
        for iteration in iterator:
            # Выбираем working set
            B = self._select_working_set(alphas, gradients, y, C_upper, active_set)
            
            if len(B) == 0:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Решаем подзадачу
            alpha_B_old = alphas[B].copy()
            alphas[B] = self._solve_subproblem(X, y, alphas, B, C_upper, gradients)
            
            # Обновляем градиенты
            delta_alpha = alphas[B] - alpha_B_old
            if np.max(np.abs(delta_alpha)) > 1e-10:
                self._update_gradients(X, y, delta_alpha, B, active_set, gradients)
            
            # Проверка сходимости
            max_violation = self._compute_max_violation(alphas[active_set], gradients[active_set], y[active_set], C_upper[active_set])
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: violation={max_violation:.6f}, |B|={len(B)}")
            
            if max_violation < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Финализация решения
        self._finalize_solution(X, y, alphas, C_upper)
        
        if self.verbose:
            print(f"Training complete:")
            print(f"  Support vectors: {len(self.alphas)} / {n_samples}")
            print(f"  Cache statistics: hits={self._cache_hits}, misses={self._cache_misses}")
        
        return self

    def _select_working_set(self, alphas, gradients, y, C_upper, candidate_set, q=None):
        """
        Выбор working set с учётом KKT условий.
        """
        if q is None:
            q = self.q
            
        if len(candidate_set) == 0:
            return np.array([], dtype=int)
        
        eps = 1e-8
        candidates = candidate_set
        
        # Маски для разных типов переменных
        at_lower = alphas[candidates] < eps
        at_upper = alphas[candidates] > C_upper[candidates] - eps
        free = ~at_lower & ~at_upper
        
        # Нарушения KKT для dual задачи
        violations = np.zeros(len(candidates))
        violations[at_lower] = np.maximum(0, gradients[candidates][at_lower])  # Хотим grad <= 0
        violations[at_upper] = np.maximum(0, -gradients[candidates][at_upper])  # Хотим grad >= 0
        violations[free] = np.abs(gradients[candidates][free])  # Хотим grad = 0
        
        # Сортируем по нарушениям
        sorted_indices = np.argsort(-violations)
        selected = candidates[sorted_indices[:q]]
        
        return selected

    def _solve_subproblem(self, X, y, alphas, B, C_upper, gradients):
        """
        Решение подзадачи для working set B.
        """
        # Создаем подзадачу QP
        n_B = len(B)
        
        # Матрица P для подзадачи
        K_BB = np.dot(X[B], X[B].T)
        YY_BB = np.outer(y[B], y[B])
        P_B = YY_BB * K_BB
        P_B = P_B + 1e-8 * np.eye(n_B)
        P_B_cvx = matrix(P_B.astype(np.float64))
        
        # Вектор q для подзадачи
        q_B = gradients[B]
        q_B_cvx = matrix(-q_B.astype(np.float64))
        
        # Ограничения для подзадачи
        values = [-1.0] * n_B + [1.0] * n_B
        rows = list(range(2 * n_B))
        cols = list(range(n_B)) * 2
        G_B_cvx = spmatrix(values, rows, cols, (2 * n_B, n_B))
        
        h_lower = np.zeros(n_B)
        h_upper = C_upper[B]
        h_combined = np.hstack([h_lower, h_upper])
        h_B_cvx = matrix(h_combined.astype(np.float64))
        
        # Ограничение равенства
        A_eq_B = matrix(y[B].reshape(1, -1).astype(np.float64))
        b_eq_B = matrix(np.zeros(1).astype(np.float64))
        
        # Решаем подзадачу
        solution = solvers.qp(P_B_cvx, q_B_cvx, G_B_cvx, h_B_cvx, A_eq_B, b_eq_B)
        
        if solution['status'] != 'optimal':
            warnings.warn(f"Subproblem QP solver status = {solution['status']}")
        
        alphas_B_new = np.array(solution['x']).flatten()
        
        return alphas_B_new

    def _update_gradients(self, X, y, delta_alpha, B, active_set, gradients):
        """
        Эффективное обновление градиентов.
        """
        # Вычисляем изменение: gradients[active] -= X[active] @ X[B].T @ (delta_alpha * y[B]) * y[active]
        temp = X[B].T @ (delta_alpha * y[B])  # (n_features,)
        kernel_delta = X[active_set] @ temp  # (n_active,)
        gradients[active_set] -= kernel_delta * y[active_set]

    def _compute_max_violation(self, alphas, gradients, y, C_upper):
        """
        Векторизованное вычисление максимального нарушения KKT.
        
        KKT для dual (min 1/2 α^T Q α - q^T α):
        - gradients = q - Qα, т.е. ∇f = -gradients
        - При α=0: ∇f >= 0 (нарушение если gradients > 0)
        - При α=C: ∇f <= 0 (нарушение если gradients < 0)
        - Free: ∇f = 0 (нарушение если gradients ≠ 0)
        """
        if len(alphas) == 0:
            return 0.0
        
        eps = 1e-8
        
        # Маски
        at_lower = alphas < eps
        at_upper = alphas > C_upper - eps
        free = ~at_lower & ~at_upper
        
        # Нарушения KKT для dual
        violations = np.zeros(len(alphas))
        violations[at_lower] = np.maximum(0, gradients[at_lower])  # Хотим grad <= 0
        violations[at_upper] = np.maximum(0, -gradients[at_upper])  # Хотим grad >= 0
        violations[free] = np.abs(gradients[free])  # Хотим grad = 0
        
        return np.max(violations)

    def _finalize_solution(self, X, y, alphas, C_upper):
        """
        Финализация решения после обучения.
        """
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
        for i in range(len(alphas)):
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
            # Если нет свободных SV, используем все SV
            b_all = []
            for i in np.where(sv_indices)[0]:
                wx = np.dot(self.w, X[i])
                if y[i] > 0:
                    b_all.append(1.0 - wx)
                else:
                    b_all.append(-self.kappa - wx)
            self.b = np.mean(b_all) if b_all else 0.0

    def decision_function(self, X):
        """Вычисление значения решающей функции f(x) = w^T x + b"""
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """Предсказание класса: sign(f(x))"""
        return np.sign(self.decision_function(X))

    def predict_proba(self, X):
        """Вероятностные предсказания (приближение через расстояние до границы)"""
        decisions = self.decision_function(X)
        # Приближение вероятности через сигмоиду
        probs = 1 / (1 + np.exp(-decisions))
        return np.vstack([1 - probs, probs]).T

    def get_support_vectors(self):
        """Возвращает опорные векторы и их метки"""
        return self.support_vectors, self.support_vector_labels

    def get_params(self):
        """Возвращает параметры модели"""
        return {
            'C_slack': self.C,
            'C_pos': self.C_pos,
            'C_neg': self.C_neg,
            'kappa': self.kappa,
            'use_wss': self.use_wss,
            'working_set_size': self.q,
            'max_iter': self.max_iter,
            'tol': self.tol
        }

    def get_convergence_info(self):
        """Возвращает информацию о сходимости"""
        return self.convergence_info