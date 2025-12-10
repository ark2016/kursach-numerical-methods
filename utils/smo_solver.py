"""
Sequential Minimal Optimization (SMO) солвер для Cost-Sensitive SVM.

Реализация строго по статье "Cost-sensitive Support Vector Machines"
(Masnadi-Shirazi et al., arXiv:1212.0975v2)

Алгоритм SMO основан на работе:
- Platt, J. (1998). "Sequential Minimal Optimization: A Fast Algorithm for Training SVMs"
- Fan, R.-E., Chen, P.-H., & Lin, C.-J. (2005). "Working Set Selection Using Second Order Information"

Двойственная задача CS-SVM (уравнение 51):
    max_α Σ_i α_i·q_i - 1/2 Σ_i Σ_j α_i α_j y_i y_j K(x_i,x_j)

    где q_i = (y_i+1)/2 - κ(y_i-1)/2:
        q_i = 1   для y_i = +1
        q_i = κ   для y_i = -1

    s.t. Σ_i α_i y_i = 0
         0 ≤ α_i ≤ C·C₁   ; y_i = +1
         0 ≤ α_i ≤ C/κ    ; y_i = -1

    где κ = 1/(2C_{-1} - 1)

В форме минимизации:
    min f(α) = 1/2 Σ α_i α_j y_i y_j K_ij - Σ α_i q_i

Градиент:
    G_i = Σ_j α_j y_j K_ij - q_i

Условия KKT (для минимизации):
    α_i = 0:      G_i >= -λ y_i
    α_i = C_i:    G_i <= -λ y_i
    0 < α < C:    G_i = -λ y_i

где λ - множитель Лагранжа для Σαy=0.

Из прямой задачи, для свободного SV:
    y_i(w·x_i + b) = m_i
где m_i = 1 для y=+1, m_i = κ для y=-1.
Т.к. m_i = q_i, получаем:
    b = y_i q_i - Σ_j α_j y_j K_ij = y_i q_i - (G_i + q_i)

Оптимизации:
- Memory-efficient: вычисление элементов ядра "на лету" вместо хранения полной матрицы O(N²)
- Numba JIT: ускорение критических циклов SMO в 10-100 раз

Автор: Курсовая работа (исправленная версия с корректным градиентом)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def prange(x):
        return range(x)


@dataclass
class SMOResult:
    """Результат работы SMO солвера."""
    alpha: np.ndarray
    b: float
    n_iterations: int
    n_support_vectors: int
    converged: bool
    objective_value: float


# =============================================================================
# Numba-оптимизированные функции
# =============================================================================

@njit(fastmath=True, cache=True)
def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    q: np.ndarray,
    idx: int
) -> float:
    """
    Вычисляет градиент G_i = Σ_j α_j y_j K_ij - q_i.

    Для минимизации f(α) = 1/2 α^T Q α - q^T α.
    """
    result = -q[idx]
    n = len(alpha)
    for j in range(n):
        if alpha[j] > 1e-10:
            K_ji = np.dot(X[j], X[idx])
            result += alpha[j] * y[j] * K_ji
    return result


@njit(fastmath=True, cache=True)
def compute_all_gradients(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    q: np.ndarray
) -> np.ndarray:
    """Вычисляет градиенты G_i для всех примеров."""
    n = len(alpha)
    G = -q.copy()  # Начинаем с -q

    for j in range(n):
        if alpha[j] > 1e-10:
            for i in range(n):
                K_ji = np.dot(X[j], X[i])
                G[i] += alpha[j] * y[j] * K_ji

    return G


@njit(fastmath=True, cache=True)
def update_gradients(
    G: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int, j: int,
    delta_alpha_i: float,
    delta_alpha_j: float
) -> None:
    """
    Инкрементально обновляет градиенты после изменения α_i, α_j.

    G_new = G_old + delta_alpha_i * Q[:, i] + delta_alpha_j * Q[:, j]
    где Q[:, k] = y[k] * y[:] * K[:, k]
    """
    n = len(G)
    y_i = y[i]
    y_j = y[j]

    for k in range(n):
        K_ik = np.dot(X[i], X[k])
        K_jk = np.dot(X[j], X[k])
        # Q[k, i] = y[k] * y[i] * K[k, i]
        # Q[k, j] = y[k] * y[j] * K[k, j]
        G[k] += delta_alpha_i * y_i * K_ik + delta_alpha_j * y_j * K_jk


@njit(fastmath=True, cache=True)
def select_working_set(
    alpha: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float,
    tol: float
) -> Tuple[int, int, bool]:
    """
    Выбор рабочей пары (i, j) по WSS3 из Fan et al. 2005.

    Для задачи минимизации f(α) = 1/2 α^T Q α - q^T α
    с ограничением Σ α_i y_i = 0.

    I_up = {t | α_t < C_t, y_t = +1} ∪ {t | α_t > 0, y_t = -1}
    I_down = {t | α_t > 0, y_t = +1} ∪ {t | α_t < C_t, y_t = -1}

    Выбираем:
    i = argmax_{t ∈ I_up} (-y_t G_t)
    j = argmin_{s ∈ I_down} (-y_s G_s)

    Если max - min < tol, то KKT выполнены.
    """
    n = len(alpha)

    # Ищем максимальное -y*G среди I_up
    max_i = -1
    max_val = -np.inf

    for t in range(n):
        in_I_up = False
        if y[t] > 0:  # y = +1
            in_I_up = alpha[t] < C[t] - eps
        else:  # y = -1
            in_I_up = alpha[t] > eps

        if in_I_up:
            neg_yG = -y[t] * G[t]
            if neg_yG > max_val:
                max_val = neg_yG
                max_i = t

    if max_i < 0:
        return -1, -1, False

    # Ищем минимальное -y*G среди I_down
    min_j = -1
    min_val = np.inf

    for s in range(n):
        in_I_down = False
        if y[s] > 0:  # y = +1
            in_I_down = alpha[s] > eps
        else:  # y = -1
            in_I_down = alpha[s] < C[s] - eps

        if in_I_down:
            neg_yG = -y[s] * G[s]
            if neg_yG < min_val:
                min_val = neg_yG
                min_j = s

    if min_j < 0:
        return -1, -1, False

    # Проверяем нарушение оптимальности
    if max_val - min_val < tol:
        return -1, -1, False

    return max_i, min_j, True


@njit(fastmath=True, cache=True)
def compute_bounds(
    alpha_i: float, alpha_j: float,
    y_i: float, y_j: float,
    C_i: float, C_j: float
) -> Tuple[float, float]:
    """
    Вычисляет границы L и H для α_j при оптимизации пары.

    Границы выбираются так, чтобы:
    1. 0 <= α_j <= C_j
    2. 0 <= α_i <= C_i
    3. α_i y_i + α_j y_j = const (сохранение Σαy=0)
    """
    if y_i != y_j:
        # α_i y_i + α_j y_j = const
        # При y_i = +1, y_j = -1: α_i - α_j = const
        # α_j = α_i - const
        L = max(0.0, alpha_j - alpha_i)
        H = min(C_j, C_i + alpha_j - alpha_i)
    else:
        # α_i + α_j = const (оба y одного знака)
        L = max(0.0, alpha_i + alpha_j - C_i)
        H = min(C_j, alpha_i + alpha_j)
    return L, H


@njit(fastmath=True, cache=True)
def optimize_pair(
    i: int, j: int,
    alpha: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float
) -> Tuple[float, float, bool]:
    """
    Оптимизация пары (α_i, α_j) методом SMO.

    Для минимизации f(α) = 1/2 α^T Q α - q^T α:

    Фиксируем все α кроме α_i и α_j.
    Из ограничения α_i y_i + α_j y_j = const получаем α_i через α_j.

    Оптимальное обновление (аналитическое решение одномерной QP):
    α_j^new = α_j^old - (G_i - G_j) / η

    где η = K_ii + K_jj - 2 K_ij (вторая производная).

    КРИТИЧНО: α_i вычисляется из равенства и НЕ клипается!
    Границы L, H выбраны так, чтобы гарантировать 0 <= α_i <= C_i.
    """
    alpha_i_old = alpha[i]
    alpha_j_old = alpha[j]
    y_i, y_j = y[i], y[j]
    C_i, C_j = C[i], C[j]

    # Вычисляем границы для α_j
    L, H = compute_bounds(alpha_i_old, alpha_j_old, y_i, y_j, C_i, C_j)

    if L >= H - eps:
        return alpha_i_old, alpha_j_old, False

    # Элементы ядра
    K_ii = np.dot(X[i], X[i])
    K_jj = np.dot(X[j], X[j])
    K_ij = np.dot(X[i], X[j])

    eta = K_ii + K_jj - 2.0 * K_ij

    if eta > eps:
        # Стандартный случай: η > 0
        # Для минимизации: α_j^new = α_j^old - (G_i - G_j) / η
        alpha_j_new = alpha_j_old - (G[i] - G[j]) / eta
    else:
        # η <= 0: вырожденный случай (почти линейно зависимые вектора)
        # Выбираем границу по направлению градиента
        if G[i] - G[j] > 0:
            alpha_j_new = L
        elif G[i] - G[j] < 0:
            alpha_j_new = H
        else:
            alpha_j_new = alpha_j_old

    # Клипируем α_j к границам
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L

    # Проверяем значимость изменения
    if abs(alpha_j_new - alpha_j_old) < eps * (alpha_j_old + alpha_j_new + eps):
        return alpha_i_old, alpha_j_old, False

    # Вычисляем новое α_i из ограничения Σαy=0
    # α_i y_i + α_j y_j = α_i_old y_i + α_j_old y_j
    # α_i = α_i_old + y_i y_j (α_j_old - α_j_new)
    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

    # КРИТИЧНО: НЕ клипаем α_i! Границы L, H гарантируют корректность.
    # Явный клип нарушает Σαy=0!

    return alpha_i_new, alpha_j_new, True


@njit(fastmath=True, cache=True)
def compute_bias_from_sv(
    alpha: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    eps: float
) -> float:
    """
    Вычисляет bias из KKT условий на свободных опорных векторах.

    Для свободного SV (0 < α < C):
    Из прямой задачи: y_i(w·x_i + b) = m_i = q_i
    => w·x_i + b = y_i q_i
    => b = y_i q_i - w·x_i

    Из градиента: G_i = w·x_i - q_i
    => w·x_i = G_i + q_i
    => b = y_i q_i - (G_i + q_i)
    """
    n = len(alpha)
    b_sum = 0.0
    n_free = 0

    for i in range(n):
        if eps < alpha[i] < C[i] - eps:
            # Свободный SV
            # b = y_i * q_i - (G_i + q_i)
            b_i = y[i] * q[i] - (G[i] + q[i])
            b_sum += b_i
            n_free += 1

    if n_free > 0:
        return b_sum / n_free

    # Нет свободных SV: используем границы
    # Для α=0: G_i >= -λ y_i => если y>0: G >= -λ => -G/y = λ/y <= ...
    # Для α=C: G_i <= -λ y_i
    # λ = -b (из KKT), поэтому:
    # α=0, y=+1: G >= b => b <= G
    # α=0, y=-1: G >= -b => b >= -G
    # α=C, y=+1: G <= b => b >= G
    # α=C, y=-1: G <= -b => b <= -G

    b_low = -1e30
    b_high = 1e30

    for i in range(n):
        if alpha[i] < eps:  # α = 0
            if y[i] > 0:
                b_high = min(b_high, G[i])
            else:
                b_low = max(b_low, -G[i])
        elif alpha[i] > C[i] - eps:  # α = C
            if y[i] > 0:
                b_low = max(b_low, G[i])
            else:
                b_high = min(b_high, -G[i])

    if b_low > -1e29 and b_high < 1e29:
        return (b_low + b_high) / 2.0
    elif b_low > -1e29:
        return b_low
    elif b_high < 1e29:
        return b_high
    return 0.0


@njit(fastmath=True, cache=True)
def smo_main_loop(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Основной цикл SMO с корректным градиентом.

    Теперь q используется для вычисления градиента G = Σ α_j y_j K_ij - q_i.
    Bias вычисляется в конце по формуле b = y_i q_i - (G_i + q_i) для свободных SV.

    Returns:
        (alpha, b, n_iterations, converged)
    """
    n = len(alpha)

    # Инициализируем градиенты G = -q (т.к. α = 0)
    G = compute_all_gradients(X, y, alpha, q)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Выбираем рабочую пару
        i, j, found = select_working_set(alpha, G, y, C, eps, tol)

        if not found:
            converged = True
            break

        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]

        # Оптимизируем пару
        alpha_i_new, alpha_j_new, changed = optimize_pair(
            i, j, alpha, G, X, y, C, eps
        )

        if not changed:
            continue

        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        # Обновляем alpha
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new

        # Обновляем градиенты инкрементально
        update_gradients(G, X, y, i, j, delta_alpha_i, delta_alpha_j)

    # Вычисляем bias в конце по свободным SV
    b = compute_bias_from_sv(alpha, G, y, q, C, eps)

    return alpha, b, n_iter, converged


# =============================================================================
# Основной класс солвера
# =============================================================================

class CSSVMDualQPSolver:
    """
    SMO солвер для CS-SVM (Cost-Sensitive Support Vector Machine).

    ИСПРАВЛЕННАЯ ВЕРСИЯ: использует градиент G = Σ α_j y_j K_ij - q_i
    вместо ошибок E, что критично для CS-SVM с q_i != 1.
    """

    def __init__(
        self,
        C_slack: float = 1.0,
        C_pos: float = 3.0,
        C_neg: float = 2.0,
        tol: float = 1e-3,
        eps: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False
    ):
        if C_neg < 1.0:
            raise ValueError(f"C_neg должно быть >= 1, получено {C_neg}")
        min_c_pos = 2 * C_neg - 1
        if C_pos < min_c_pos:
            raise ValueError(f"C_pos должно быть >= {min_c_pos}, получено {C_pos}")

        self.C_slack = C_slack
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose

        # κ = 1/(2C_{-1} - 1)
        self.kappa = 1.0 / (2 * C_neg - 1)

        # Box constraints
        self.C_upper_pos = C_slack * C_pos      # C·C₁ для y=+1
        self.C_upper_neg = C_slack / self.kappa  # C/κ для y=-1

    def solve(self, X: np.ndarray, y: np.ndarray) -> SMOResult:
        """Решает двойственную задачу CS-SVM."""
        n_samples = X.shape[0]

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.where(y > 0, 1.0, -1.0).astype(np.float64)
        y = np.ascontiguousarray(y)

        # Инициализация
        alpha = np.zeros(n_samples, dtype=np.float64)

        # q_i: линейный коэффициент (ТЕПЕРЬ ИСПОЛЬЗУЕТСЯ!)
        q = np.array([1.0 if y_i > 0 else self.kappa for y_i in y], dtype=np.float64)
        q = np.ascontiguousarray(q)

        # C_i: верхняя граница
        C = np.array([self.C_upper_pos if y_i > 0 else self.C_upper_neg for y_i in y], dtype=np.float64)
        C = np.ascontiguousarray(C)

        if self.verbose:
            print(f"SMO solver started: {n_samples} samples")
            print(f"  Memory-efficient mode: Q matrix NOT stored")
            print(f"  Numba JIT: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")

        # Запускаем SMO
        alpha, b, n_iter, converged = smo_main_loop(
            X, y, alpha, C, q,
            self.eps, self.tol, self.max_iter
        )

        n_sv = int(np.sum(alpha > self.eps))
        obj = self._compute_objective(alpha, X, y, q)

        if self.verbose:
            print(f"SMO finished: {n_iter} iterations, {n_sv} support vectors, converged={converged}")
            print(f"  Objective value: {obj:.6f}")

        return SMOResult(
            alpha=alpha,
            b=b,
            n_iterations=n_iter,
            n_support_vectors=n_sv,
            converged=converged,
            objective_value=obj
        )

    def _compute_objective(self, alpha: np.ndarray, X: np.ndarray, y: np.ndarray, q: np.ndarray) -> float:
        """f(α) = q^T α - 1/2 α^T Q α (максимизация)"""
        linear_part = np.dot(q, alpha)

        sv_mask = alpha > self.eps
        if not np.any(sv_mask):
            return linear_part

        sv_idx = np.where(sv_mask)[0]
        alpha_sv = alpha[sv_idx]
        y_sv = y[sv_idx]
        X_sv = X[sv_idx]

        K_sv = np.dot(X_sv, X_sv.T)
        Q_sv = np.outer(y_sv, y_sv) * K_sv
        quad_part = 0.5 * np.dot(alpha_sv, np.dot(Q_sv, alpha_sv))

        return linear_part - quad_part


def solve_cssvm_dual_qp(
    X: np.ndarray,
    y: np.ndarray,
    C_slack: float = 1.0,
    C_pos: float = 3.0,
    C_neg: float = 2.0,
    tol: float = 1e-3,
    max_iter: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Решает двойственную задачу CS-SVM.

    Returns:
        w: Вектор весов
        b: Смещение
        alpha: Множители Лагранжа
    """
    y_norm = np.where(y > 0, 1.0, -1.0).astype(np.float64)

    solver = CSSVMDualQPSolver(
        C_slack=C_slack,
        C_pos=C_pos,
        C_neg=C_neg,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    result = solver.solve(X, y_norm)

    # w = Σ α_i y_i x_i
    w = np.sum((result.alpha * y_norm).reshape(-1, 1) * X, axis=0)

    return w, result.b, result.alpha


# =============================================================================
# Fast версия для малых датасетов
# =============================================================================

class CSSVMDualQPSolverFast:
    """
    Быстрая версия с предвычислением матрицы Q.
    Для датасетов < 15k примеров.

    ИСПРАВЛЕННАЯ ВЕРСИЯ: использует градиент с учётом q.
    """

    def __init__(
        self,
        C_slack: float = 1.0,
        C_pos: float = 3.0,
        C_neg: float = 2.0,
        tol: float = 1e-3,
        eps: float = 1e-8,
        max_iter: int = 10000,
        verbose: bool = False
    ):
        if C_neg < 1.0:
            raise ValueError(f"C_neg >= 1 required")
        min_c_pos = 2 * C_neg - 1
        if C_pos < min_c_pos:
            raise ValueError(f"C_pos >= {min_c_pos} required")

        self.C_slack = C_slack
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose

        self.kappa = 1.0 / (2 * C_neg - 1)
        self.C_upper_pos = C_slack * C_pos
        self.C_upper_neg = C_slack / self.kappa

    def solve(self, X: np.ndarray, y: np.ndarray) -> SMOResult:
        n = X.shape[0]

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.where(y > 0, 1.0, -1.0).astype(np.float64)

        # Предвычисляем K
        K = np.dot(X, X.T)

        q = np.array([1.0 if y_i > 0 else self.kappa for y_i in y])
        C = np.array([self.C_upper_pos if y_i > 0 else self.C_upper_neg for y_i in y])

        alpha = np.zeros(n)

        # Градиент G = Σ α_j y_j K_ij - q_i
        # Начинаем с α=0: G = -q
        G = -q.copy()

        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # Working set selection
            neg_yG = -y * G

            # I_up
            max_i = -1
            max_val = -np.inf
            for t in range(n):
                in_up = (y[t] > 0 and alpha[t] < C[t] - self.eps) or (y[t] < 0 and alpha[t] > self.eps)
                if in_up and neg_yG[t] > max_val:
                    max_val = neg_yG[t]
                    max_i = t

            if max_i < 0:
                converged = True
                break

            # I_down
            min_j = -1
            min_val = np.inf
            for s in range(n):
                in_down = (y[s] > 0 and alpha[s] > self.eps) or (y[s] < 0 and alpha[s] < C[s] - self.eps)
                if in_down and neg_yG[s] < min_val:
                    min_val = neg_yG[s]
                    min_j = s

            if min_j < 0 or max_val - min_val < self.tol:
                converged = True
                break

            i, j = max_i, min_j

            # Оптимизация пары
            alpha_i_old, alpha_j_old = alpha[i], alpha[j]
            y_i, y_j = y[i], y[j]

            if y_i != y_j:
                L = max(0.0, alpha_j_old - alpha_i_old)
                H = min(C[j], C[i] + alpha_j_old - alpha_i_old)
            else:
                L = max(0.0, alpha_i_old + alpha_j_old - C[i])
                H = min(C[j], alpha_i_old + alpha_j_old)

            if L >= H - self.eps:
                continue

            eta = K[i, i] + K[j, j] - 2 * K[i, j]

            if eta > self.eps:
                # ИСПРАВЛЕНО: используем градиент напрямую
                alpha_j_new = alpha_j_old - (G[i] - G[j]) / eta
            else:
                if G[i] - G[j] > 0:
                    alpha_j_new = L
                elif G[i] - G[j] < 0:
                    alpha_j_new = H
                else:
                    alpha_j_new = alpha_j_old

            alpha_j_new = np.clip(alpha_j_new, L, H)

            if abs(alpha_j_new - alpha_j_old) < self.eps:
                continue

            # ИСПРАВЛЕНО: НЕ клипаем alpha_i!
            alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

            delta_i = alpha_i_new - alpha_i_old
            delta_j = alpha_j_new - alpha_j_old

            alpha[i] = alpha_i_new
            alpha[j] = alpha_j_new

            # Обновляем градиенты
            # G += delta_i * Q[:, i] + delta_j * Q[:, j]
            # Q[:, k] = y[k] * y[:] * K[:, k]
            G += delta_i * y_i * K[i, :] + delta_j * y_j * K[j, :]

        # Вычисляем bias
        n_free = 0
        b_sum = 0.0
        for i in range(n):
            if self.eps < alpha[i] < C[i] - self.eps:
                # b = y_i q_i - (G_i + q_i)
                b_i = y[i] * q[i] - (G[i] + q[i])
                b_sum += b_i
                n_free += 1

        if n_free > 0:
            b = b_sum / n_free
        else:
            # Используем границы
            b_low = -1e30
            b_high = 1e30
            for i in range(n):
                if alpha[i] < self.eps:
                    if y[i] > 0:
                        b_high = min(b_high, G[i])
                    else:
                        b_low = max(b_low, -G[i])
                elif alpha[i] > C[i] - self.eps:
                    if y[i] > 0:
                        b_low = max(b_low, G[i])
                    else:
                        b_high = min(b_high, -G[i])

            if b_low > -1e29 and b_high < 1e29:
                b = (b_low + b_high) / 2.0
            elif b_low > -1e29:
                b = b_low
            elif b_high < 1e29:
                b = b_high
            else:
                b = 0.0

        n_sv = int(np.sum(alpha > self.eps))

        # Objective
        linear_part = np.dot(q, alpha)
        sv_mask = alpha > self.eps
        if np.any(sv_mask):
            sv_idx = np.where(sv_mask)[0]
            alpha_sv = alpha[sv_idx]
            y_sv = y[sv_idx]
            K_sv = K[np.ix_(sv_idx, sv_idx)]
            Q_sv = np.outer(y_sv, y_sv) * K_sv
            quad_part = 0.5 * np.dot(alpha_sv, np.dot(Q_sv, alpha_sv))
            obj = linear_part - quad_part
        else:
            obj = linear_part

        return SMOResult(
            alpha=alpha,
            b=b,
            n_iterations=n_iter,
            n_support_vectors=n_sv,
            converged=converged,
            objective_value=obj
        )
