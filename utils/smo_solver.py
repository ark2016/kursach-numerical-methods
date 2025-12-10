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

Оптимизации:
- Memory-efficient: вычисление элементов ядра "на лету" вместо хранения полной матрицы O(N²)
- Numba JIT: ускорение критических циклов SMO в 10-100 раз

Автор: Курсовая работа
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
def compute_f(X: np.ndarray, y: np.ndarray, alpha: np.ndarray, b: float, idx: int) -> float:
    """Вычисляет f(x_idx) = Σ α_j y_j K(x_j, x_idx) + b"""
    result = b
    n = len(alpha)
    for j in range(n):
        if alpha[j] > 1e-10:
            K_j_idx = np.dot(X[j], X[idx])
            result += alpha[j] * y[j] * K_j_idx
    return result


@njit(fastmath=True, cache=True)
def compute_all_f(X: np.ndarray, y: np.ndarray, alpha: np.ndarray, b: float) -> np.ndarray:
    """Вычисляет f(x_i) для всех примеров."""
    n = len(alpha)
    f_vals = np.full(n, b)

    for j in range(n):
        if alpha[j] > 1e-10:
            for i in range(n):
                K_ji = np.dot(X[j], X[i])
                f_vals[i] += alpha[j] * y[j] * K_ji
    return f_vals


@njit(fastmath=True, cache=True)
def update_f_values(
    f_vals: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int, j: int,
    delta_alpha_i: float,
    delta_alpha_j: float
) -> None:
    """Инкрементально обновляет f после изменения α_i, α_j."""
    n = len(f_vals)
    for k in range(n):
        K_ik = np.dot(X[i], X[k])
        K_jk = np.dot(X[j], X[k])
        f_vals[k] += delta_alpha_i * y[i] * K_ik + delta_alpha_j * y[j] * K_jk


@njit(fastmath=True, cache=True)
def select_working_set(
    alpha: np.ndarray,
    f_vals: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    margins: np.ndarray,
    eps: float,
    tol: float
) -> Tuple[int, int, bool]:
    """
    Выбор рабочей пары (i, j) по методу максимального нарушения KKT.

    Для задачи МАКСИМИЗАЦИИ dual:
    - E_i = f(x_i) - y_i * margin_i (ошибка)
    - Если α_i = 0 и y_i*E_i < -tol: может увеличиться (нарушение)
    - Если α_i = C_i и y_i*E_i > tol: может уменьшиться (нарушение)
    - Если 0 < α_i < C_i и |y_i*E_i| > tol: нарушение

    Выбираем i с максимальным нарушением, j - максимизирующий |E_i - E_j|.
    """
    n = len(alpha)

    # Вычисляем ошибки E_i = f(x_i) - y_i * margin_i
    E = np.empty(n)
    for k in range(n):
        E[k] = f_vals[k] - y[k] * margins[k]

    # Множество I_up: примеры где α может увеличиться
    # y_i = +1: α_i < C_i (может расти)
    # y_i = -1: α_i > 0 (может уменьшаться, что для -1 означает рост влияния)

    # Множество I_down: примеры где α может уменьшиться
    # y_i = +1: α_i > 0 (может уменьшаться)
    # y_i = -1: α_i < C_i (может расти)

    # Ищем максимальный -y*E среди I_up
    max_i = -1
    max_neg_yE = -np.inf

    for k in range(n):
        in_I_up = False
        if y[k] > 0:
            in_I_up = alpha[k] < C[k] - eps
        else:
            in_I_up = alpha[k] > eps

        neg_yE = -y[k] * E[k]
        if in_I_up and neg_yE > max_neg_yE:
            max_neg_yE = neg_yE
            max_i = k

    if max_i < 0:
        return -1, -1, False

    i = max_i

    # Ищем минимальный -y*E среди I_down
    min_j = -1
    min_neg_yE = np.inf

    for k in range(n):
        in_I_down = False
        if y[k] > 0:
            in_I_down = alpha[k] > eps
        else:
            in_I_down = alpha[k] < C[k] - eps

        neg_yE = -y[k] * E[k]
        if in_I_down and neg_yE < min_neg_yE:
            min_neg_yE = neg_yE
            min_j = k

    if min_j < 0:
        return -1, -1, False

    # Проверяем нарушение оптимальности
    if max_neg_yE - min_neg_yE < tol:
        return -1, -1, False

    return i, min_j, True


@njit(fastmath=True, cache=True)
def compute_bounds(
    alpha_i: float, alpha_j: float,
    y_i: float, y_j: float,
    C_i: float, C_j: float
) -> Tuple[float, float]:
    """Границы L и H для α_j при оптимизации пары."""
    if y_i != y_j:
        L = max(0.0, alpha_j - alpha_i)
        H = min(C_j, C_i + alpha_j - alpha_i)
    else:
        L = max(0.0, alpha_i + alpha_j - C_i)
        H = min(C_j, alpha_i + alpha_j)
    return L, H


@njit(fastmath=True, cache=True)
def optimize_pair(
    i: int, j: int,
    alpha: np.ndarray,
    f_vals: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    margins: np.ndarray,
    eps: float
) -> Tuple[float, float, bool]:
    """
    Оптимизация пары (α_i, α_j).

    Формула SMO для задачи максимизации:
    α_j_new = α_j_old + y_j * (E_i - E_j) / η
    где η = K_ii + K_jj - 2*K_ij
    """
    alpha_i_old = alpha[i]
    alpha_j_old = alpha[j]
    y_i, y_j = y[i], y[j]
    C_i, C_j = C[i], C[j]

    L, H = compute_bounds(alpha_i_old, alpha_j_old, y_i, y_j, C_i, C_j)

    if L >= H - eps:
        return alpha_i_old, alpha_j_old, False

    # Ошибки
    E_i = f_vals[i] - y_i * margins[i]
    E_j = f_vals[j] - y_j * margins[j]

    # Элементы ядра
    K_ii = np.dot(X[i], X[i])
    K_jj = np.dot(X[j], X[j])
    K_ij = np.dot(X[i], X[j])

    eta = K_ii + K_jj - 2.0 * K_ij

    if eta > eps:
        alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
    else:
        # η <= 0: выбираем границу по значению целевой функции
        # Упрощённо: оставляем как есть или берём ближайшую границу
        s = y_i * y_j
        f1 = y_j * (E_i - E_j)
        # При η=0, df/dα_j = y_j*(E_i - E_j) = f1
        # Если f1 > 0: α_j растёт -> H
        # Если f1 < 0: α_j падает -> L
        if f1 > 0:
            alpha_j_new = H
        elif f1 < 0:
            alpha_j_new = L
        else:
            alpha_j_new = alpha_j_old

    # Clip to bounds
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L

    # Проверяем значимость изменения
    if abs(alpha_j_new - alpha_j_old) < eps * (alpha_j_old + alpha_j_new + eps):
        return alpha_i_old, alpha_j_old, False

    # Вычисляем новое α_i из ограничения α_i*y_i + α_j*y_j = const
    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

    # Clip α_i
    if alpha_i_new > C_i:
        alpha_i_new = C_i
    elif alpha_i_new < 0:
        alpha_i_new = 0.0

    return alpha_i_new, alpha_j_new, True


@njit(fastmath=True, cache=True)
def compute_bias(
    alpha: np.ndarray,
    f_vals: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    margins: np.ndarray,
    eps: float
) -> float:
    """Вычисляет bias из KKT условий на свободных опорных векторах."""
    n = len(alpha)
    b_sum = 0.0
    n_free = 0

    for i in range(n):
        if eps < alpha[i] < C[i] - eps:
            # Свободный SV: f(x_i) = y_i * margin_i
            # b = y_i * margin_i - Σ α_j y_j K_ji
            # f(x_i) = Σ α_j y_j K_ji + b
            # Для свободного SV: f(x_i) должно равняться y_i*m_i
            # Текущий f_vals[i] включает текущий b
            # Но мы пересчитаем из условия:
            # b_i = y_i * margins[i] - (f_vals[i] - текущий_b)
            # Проще: для свободного SV ошибка E_i = 0
            # E_i = f_vals[i] - y_i * margins[i] = 0
            # => b корректируется так чтобы это выполнялось
            b_sum += y[i] * margins[i] - f_vals[i]
            n_free += 1

    if n_free > 0:
        # Среднее смещение которое нужно добавить к текущему b
        return b_sum / n_free

    # Нет свободных SV
    b_low = -1e30
    b_high = 1e30

    for i in range(n):
        y_i = y[i]
        E_i = f_vals[i] - y_i * margins[i]

        if alpha[i] < eps:
            # α = 0: y*E <= 0 (для выполнения KKT)
            if y_i > 0:
                # y=+1: E <= 0 => f <= m => b <= m - (f-b) => b_i = m - f + b_current
                b_high = min(b_high, -E_i)
            else:
                # y=-1: E >= 0 => f >= m (но m отриц) => b >= ...
                b_low = max(b_low, -E_i)
        elif alpha[i] > C[i] - eps:
            # α = C: y*E >= 0
            if y_i > 0:
                b_low = max(b_low, -E_i)
            else:
                b_high = min(b_high, -E_i)

    if b_low > -1e29 and b_high < 1e29:
        return (b_low + b_high) / 2
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
    margins: np.ndarray,
    q: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Основной цикл SMO.

    Returns:
        (alpha, b, n_iterations, converged)
    """
    n = len(alpha)
    b = 0.0

    # Инициализируем f(x_i) = b = 0 для всех (т.к. α=0)
    f_vals = np.zeros(n)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Выбираем рабочую пару
        i, j, found = select_working_set(alpha, f_vals, y, C, margins, eps, tol)

        if not found:
            converged = True
            break

        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]

        alpha_i_new, alpha_j_new, changed = optimize_pair(
            i, j, alpha, f_vals, X, y, C, margins, eps
        )

        if not changed:
            continue

        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new

        # Обновляем f_vals инкрементально
        update_f_values(f_vals, X, y, i, j, delta_alpha_i, delta_alpha_j)

        # Обновляем b
        b_delta = compute_bias(alpha, f_vals, y, C, margins, eps)
        # Добавляем дельту к f_vals (чтобы включить новый b)
        for k in range(n):
            f_vals[k] += b_delta
        b += b_delta

    return alpha, b, n_iter, converged


# =============================================================================
# Основной класс солвера
# =============================================================================

class CSSVMDualQPSolver:
    """
    SMO солвер для CS-SVM (Cost-Sensitive Support Vector Machine).

    Memory-efficient: матрица Q не хранится, элементы вычисляются на лету.
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

        # q_i: линейный коэффициент (не используется напрямую в SMO)
        q = np.array([1.0 if y_i > 0 else self.kappa for y_i in y], dtype=np.float64)

        # C_i: верхняя граница
        C = np.array([self.C_upper_pos if y_i > 0 else self.C_upper_neg for y_i in y], dtype=np.float64)

        # margins: целевые margins для каждого класса
        # Для y=+1: margin = 1 (w·x + b >= 1)
        # Для y=-1: margin = κ (w·x + b <= -κ, т.е. y*(w·x+b) >= κ)
        margins = np.array([1.0 if y_i > 0 else self.kappa for y_i in y], dtype=np.float64)

        if self.verbose:
            print(f"SMO solver started: {n_samples} samples")
            print(f"  Memory-efficient mode: Q matrix NOT stored")
            print(f"  Numba JIT: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")

        # Запускаем SMO
        alpha, b, n_iter, converged = smo_main_loop(
            X, y, alpha, C, margins, q,
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
        """f(α) = q^T α - 1/2 α^T Q α"""
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

        # Предвычисляем K и Q
        K = np.dot(X, X.T)

        q = np.array([1.0 if y_i > 0 else self.kappa for y_i in y])
        C = np.array([self.C_upper_pos if y_i > 0 else self.C_upper_neg for y_i in y])
        margins = np.array([1.0 if y_i > 0 else self.kappa for y_i in y])

        alpha = np.zeros(n)
        b = 0.0
        f_vals = np.zeros(n)  # f(x_i) = Σ α_j y_j K_ji + b

        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # Вычисляем E_i = f(x_i) - y_i * m_i
            E = f_vals - y * margins

            # Выбор рабочей пары
            neg_yE = -y * E

            # I_up
            max_i = -1
            max_val = -np.inf
            for k in range(n):
                in_up = (y[k] > 0 and alpha[k] < C[k] - self.eps) or (y[k] < 0 and alpha[k] > self.eps)
                if in_up and neg_yE[k] > max_val:
                    max_val = neg_yE[k]
                    max_i = k

            if max_i < 0:
                converged = True
                break

            # I_down
            min_j = -1
            min_val = np.inf
            for k in range(n):
                in_down = (y[k] > 0 and alpha[k] > self.eps) or (y[k] < 0 and alpha[k] < C[k] - self.eps)
                if in_down and neg_yE[k] < min_val:
                    min_val = neg_yE[k]
                    min_j = k

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
                alpha_j_new = alpha_j_old + y_j * (E[i] - E[j]) / eta
            else:
                alpha_j_new = alpha_j_old

            alpha_j_new = np.clip(alpha_j_new, L, H)

            if abs(alpha_j_new - alpha_j_old) < self.eps:
                continue

            alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
            alpha_i_new = np.clip(alpha_i_new, 0, C[i])

            delta_i = alpha_i_new - alpha_i_old
            delta_j = alpha_j_new - alpha_j_old

            alpha[i] = alpha_i_new
            alpha[j] = alpha_j_new

            # Обновляем f_vals
            f_vals += delta_i * y_i * K[i, :] + delta_j * y_j * K[j, :]

            # Обновляем b
            n_free = 0
            b_sum = 0.0
            for k in range(n):
                if self.eps < alpha[k] < C[k] - self.eps:
                    b_sum += y[k] * margins[k] - f_vals[k]
                    n_free += 1

            if n_free > 0:
                b_delta = b_sum / n_free
                f_vals += b_delta
                b += b_delta

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
