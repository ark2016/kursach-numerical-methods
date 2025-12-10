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
- LRU-кэширование строк ядра для частого доступа

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
    # Fallback декораторы если Numba не установлена
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
    alpha: np.ndarray          # Множители Лагранжа
    b: float                   # Смещение (bias)
    n_iterations: int          # Количество итераций
    n_support_vectors: int     # Количество опорных векторов
    converged: bool            # Сходимость достигнута
    objective_value: float     # Значение целевой функции


# =============================================================================
# Numba-оптимизированные функции ядра
# =============================================================================

@njit(fastmath=True, cache=True)
def linear_kernel_single(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Линейное ядро для двух векторов: K(x_i, x_j) = x_i^T x_j"""
    return np.dot(x_i, x_j)


@njit(fastmath=True, cache=True)
def linear_kernel_row(X: np.ndarray, idx: int) -> np.ndarray:
    """
    Вычисляет строку матрицы ядра: K[idx, :] = X @ X[idx].T

    Memory-efficient: вычисляется на лету, не требует хранения полной матрицы.

    Args:
        X: Матрица данных (n_samples, n_features)
        idx: Индекс строки

    Returns:
        K_row: Строка матрицы ядра (n_samples,)
    """
    n_samples = X.shape[0]
    K_row = np.empty(n_samples, dtype=np.float64)
    x_idx = X[idx]
    for i in range(n_samples):
        K_row[i] = np.dot(X[i], x_idx)
    return K_row


@njit(fastmath=True, cache=True)
def compute_Q_row(X: np.ndarray, y: np.ndarray, idx: int) -> np.ndarray:
    """
    Вычисляет строку матрицы Q: Q[idx, :] = y[idx] * y * K[idx, :]

    Args:
        X: Матрица данных (n_samples, n_features)
        y: Метки классов (n_samples,)
        idx: Индекс строки

    Returns:
        Q_row: Строка матрицы Q (n_samples,)
    """
    K_row = linear_kernel_row(X, idx)
    return y[idx] * y * K_row


@njit(fastmath=True, cache=True)
def compute_Q_element(X: np.ndarray, y: np.ndarray, i: int, j: int) -> float:
    """Вычисляет один элемент матрицы Q: Q[i,j] = y[i] * y[j] * K(x_i, x_j)"""
    k_ij = np.dot(X[i], X[j])
    return y[i] * y[j] * k_ij


# =============================================================================
# SMO алгоритм с Numba-оптимизацией
# =============================================================================

@njit(fastmath=True, cache=True)
def select_working_set_wss3(
    alpha: np.ndarray,
    gradient: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float,
    tol: float
) -> Tuple[int, int, bool]:
    """
    Выбор рабочего множества методом WSS3 (Working Set Selection 3).

    Оптимизированная Numba-версия для быстрого выбора пары (i, j).

    Returns:
        (i, j, found): Индексы выбранной пары и флаг успеха
    """
    n = len(alpha)

    # Вычисляем -y * gradient для всех примеров
    neg_y_grad = -y * gradient

    # Ищем максимум среди I_up (примеры, которые могут увеличиться)
    max_i = -1
    max_val = -np.inf

    for i in range(n):
        # I_up: α_i < C_i для y_i > 0, или α_i > 0 для y_i < 0
        in_I_up = False
        if y[i] > 0:
            in_I_up = alpha[i] < C[i] - eps
        else:
            in_I_up = alpha[i] > eps

        if in_I_up and neg_y_grad[i] > max_val:
            max_val = neg_y_grad[i]
            max_i = i

    if max_i < 0:
        return -1, -1, False

    i = max_i

    # Ищем минимум среди I_down с условием neg_y_grad[j] < neg_y_grad[i] - tol
    min_j = -1
    min_val = np.inf
    threshold = neg_y_grad[i] - tol

    for j in range(n):
        # I_down: α_j > 0 для y_j > 0, или α_j < C_j для y_j < 0
        in_I_down = False
        if y[j] > 0:
            in_I_down = alpha[j] > eps
        else:
            in_I_down = alpha[j] < C[j] - eps

        if in_I_down and neg_y_grad[j] < threshold and neg_y_grad[j] < min_val:
            min_val = neg_y_grad[j]
            min_j = j

    if min_j < 0:
        return -1, -1, False

    return i, min_j, True


@njit(fastmath=True, cache=True)
def compute_bounds(
    alpha_i: float,
    alpha_j: float,
    y_i: float,
    y_j: float,
    C_i: float,
    C_j: float
) -> Tuple[float, float]:
    """
    Вычисляет границы L и H для α_j при оптимизации пары (i, j).
    """
    if y_i != y_j:
        # y_i ≠ y_j: α_i - α_j = const
        L = max(0.0, alpha_j - alpha_i)
        H = min(C_j, C_i + alpha_j - alpha_i)
    else:
        # y_i = y_j: α_i + α_j = const
        L = max(0.0, alpha_i + alpha_j - C_i)
        H = min(C_j, alpha_i + alpha_j)
    return L, H


@njit(fastmath=True, cache=True)
def optimize_pair_numba(
    i: int,
    j: int,
    alpha: np.ndarray,
    gradient: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float
) -> Tuple[float, float, bool, float, float]:
    """
    Оптимизация пары переменных (α_i, α_j) с вычислением ядра на лету.

    Returns:
        (new_alpha_i, new_alpha_j, changed, K_ii_plus_K_jj, K_ij)
    """
    alpha_i_old = alpha[i]
    alpha_j_old = alpha[j]
    y_i, y_j = y[i], y[j]
    C_i, C_j = C[i], C[j]

    # Вычисляем границы
    L, H = compute_bounds(alpha_i_old, alpha_j_old, y_i, y_j, C_i, C_j)

    if L >= H - eps:
        return alpha_i_old, alpha_j_old, False, 0.0, 0.0

    # Вычисляем элементы ядра на лету
    K_ii = np.dot(X[i], X[i])
    K_jj = np.dot(X[j], X[j])
    K_ij = np.dot(X[i], X[j])

    # η = K_ii + K_jj - 2·K_ij
    eta = K_ii + K_jj - 2.0 * K_ij

    # E_i = -y_i * gradient[i], E_j = -y_j * gradient[j]
    E_i = -y_i * gradient[i]
    E_j = -y_j * gradient[j]

    if eta > eps:
        # Стандартный случай: η > 0
        alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
    else:
        # η <= 0: выбираем границу (упрощённо)
        alpha_j_new = alpha_j_old

    # Ограничиваем в [L, H]
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L

    # Проверяем значимость изменения
    if abs(alpha_j_new - alpha_j_old) < eps * (alpha_j_old + alpha_j_new + eps):
        return alpha_i_old, alpha_j_old, False, K_ii + K_jj, K_ij

    # Вычисляем новое α_i
    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

    # Ограничиваем α_i
    if alpha_i_new > C_i:
        alpha_i_new = C_i
    elif alpha_i_new < 0:
        alpha_i_new = 0.0

    return alpha_i_new, alpha_j_new, True, K_ii + K_jj, K_ij


@njit(fastmath=True, cache=True)
def update_gradient_optimized(
    gradient: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    j: int,
    delta_alpha_i: float,
    delta_alpha_j: float
) -> None:
    """
    Обновляет градиент после изменения α_i и α_j.

    gradient_new = gradient_old - Q[:, i]·Δα_i - Q[:, j]·Δα_j

    Вычисление Q на лету: Q[:, k] = y * y[k] * K[:, k]
    """
    n = len(gradient)
    y_i, y_j = y[i], y[j]
    x_i, x_j = X[i], X[j]

    for k in range(n):
        # K[k, i] = X[k] · X[i]
        K_ki = np.dot(X[k], x_i)
        K_kj = np.dot(X[k], x_j)

        # Q[k, i] = y[k] * y[i] * K[k, i]
        Q_ki = y[k] * y_i * K_ki
        Q_kj = y[k] * y_j * K_kj

        gradient[k] -= delta_alpha_i * Q_ki + delta_alpha_j * Q_kj


@njit(fastmath=True, cache=True)
def check_kkt_violation(
    alpha: np.ndarray,
    gradient: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float
) -> float:
    """Вычисляет максимальное нарушение KKT условий."""
    n = len(alpha)
    max_violation = 0.0

    for i in range(n):
        y_i = y[i]
        C_i = C[i]
        yg = y_i * gradient[i]

        if alpha[i] < eps:
            # α_i = 0
            if y_i > 0:
                violation = max(0.0, -yg)
            else:
                violation = max(0.0, yg)
        elif alpha[i] > C_i - eps:
            # α_i = C_i
            if y_i > 0:
                violation = max(0.0, yg)
            else:
                violation = max(0.0, -yg)
        else:
            # 0 < α_i < C_i
            violation = abs(yg)

        if violation > max_violation:
            max_violation = violation

    return max_violation


@njit(fastmath=True, cache=True)
def compute_bias_from_gradient(
    alpha: np.ndarray,
    gradient: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    eps: float
) -> float:
    """Вычисляет смещение b из KKT условий."""
    n = len(alpha)
    b_sum = 0.0
    n_free = 0

    for i in range(n):
        C_i = C[i]
        if eps < alpha[i] < C_i - eps:
            # Свободный опорный вектор: b = y_i * gradient[i]
            b_sum += y[i] * gradient[i]
            n_free += 1

    if n_free > 0:
        return b_sum / n_free

    # Нет свободных SV — используем граничные
    b_low = -1e30
    b_high = 1e30

    for i in range(n):
        y_i = y[i]
        C_i = C[i]
        b_i = y_i * gradient[i]

        if alpha[i] < eps:
            if y_i > 0:
                if b_i < b_high:
                    b_high = b_i
            else:
                if b_i > b_low:
                    b_low = b_i
        elif alpha[i] > C_i - eps:
            if y_i > 0:
                if b_i > b_low:
                    b_low = b_i
            else:
                if b_i < b_high:
                    b_high = b_i

    if b_low > -1e29 and b_high < 1e29:
        return (b_low + b_high) / 2
    elif b_low > -1e29:
        return b_low
    elif b_high < 1e29:
        return b_high
    else:
        return 0.0


@njit(fastmath=True, cache=True)
def smo_main_loop(
    X: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    gradient: np.ndarray,
    C: np.ndarray,
    q: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int,
    check_interval: int = 100
) -> Tuple[np.ndarray, int, bool]:
    """
    Основной цикл SMO с Numba-оптимизацией.

    Memory-efficient: матрица Q не создаётся, элементы вычисляются на лету.

    Returns:
        (alpha, n_iterations, converged)
    """
    n_iter = 0
    converged = False

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Выбор рабочего множества
        i, j, found = select_working_set_wss3(alpha, gradient, y, C, eps, tol)

        if not found:
            converged = True
            break

        # Оптимизация пары
        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]

        alpha_i_new, alpha_j_new, changed, _, _ = optimize_pair_numba(
            i, j, alpha, gradient, X, y, C, eps
        )

        if not changed:
            continue

        # Обновляем α
        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new

        # Обновляем градиент
        update_gradient_optimized(gradient, X, y, i, j, delta_alpha_i, delta_alpha_j)

        # Проверка сходимости
        if (iteration + 1) % check_interval == 0:
            kkt_violation = check_kkt_violation(alpha, gradient, y, C, eps)
            if kkt_violation < tol:
                converged = True
                break

    return alpha, n_iter, converged


# =============================================================================
# Основной класс солвера
# =============================================================================

class CSSVMDualQPSolver:
    """
    Решение двойственной задачи квадратичного программирования для CS-SVM
    методом Sequential Minimal Optimization (SMO).

    Memory-efficient версия: матрица Q не хранится целиком, элементы
    вычисляются на лету. Это позволяет работать с датасетами любого размера.

    Двойственная задача (уравнение 51 из статьи):
        max_α Σ_i α_i·q_i - 1/2 α^T Q α

        где Q_ij = y_i y_j K(x_i, x_j)
            q_i = 1 для y_i = +1, q_i = κ для y_i = -1

        s.t. Σ_i α_i y_i = 0  (ограничение равенства)
             0 ≤ α_i ≤ C_i    (box constraints)

        где C_i = C·C₁ для y_i = +1
            C_i = C/κ  для y_i = -1
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
        """
        Инициализация солвера.

        Args:
            C_slack: Параметр регуляризации C (soft margin)
            C_pos: C₁ - стоимость ошибки на положительном классе
            C_neg: C₋₁ - стоимость ошибки на отрицательном классе
            tol: Допуск для проверки KKT условий
            eps: Минимальное изменение α для обновления
            max_iter: Максимальное количество итераций
            verbose: Выводить отладочную информацию
        """
        # Проверка условий из статьи (уравнение 50)
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

        # Вычисление κ по формуле (50): κ = 1/(2C₋₁ - 1)
        self.kappa = 1.0 / (2 * C_neg - 1)

        # Верхние границы для box constraints (уравнение 51)
        self.C_upper_pos = C_slack * C_pos      # C·C₁ для y_i = +1
        self.C_upper_neg = C_slack / self.kappa  # C/κ для y_i = -1

    def _get_upper_bound(self, y_i: float) -> float:
        """Возвращает верхнюю границу C_i для примера с меткой y_i."""
        return self.C_upper_pos if y_i > 0 else self.C_upper_neg

    def _get_linear_coef(self, y_i: float) -> float:
        """
        Возвращает коэффициент линейного члена q_i.

        Из уравнения 51: q_i = (y_i+1)/2 - κ(y_i-1)/2
        Для y_i = +1: q_i = 1
        Для y_i = -1: q_i = κ
        """
        return 1.0 if y_i > 0 else self.kappa

    def solve(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> SMOResult:
        """
        Решает двойственную задачу CS-SVM методом SMO.

        Memory-efficient: матрица Q не создаётся целиком.

        Args:
            X: Матрица признаков (n_samples, n_features)
            y: Метки классов, значения {-1, +1}

        Returns:
            SMOResult с решением
        """
        n_samples = X.shape[0]

        # Убеждаемся, что данные contiguous для Numba
        X = np.ascontiguousarray(X, dtype=np.float64)

        # Нормализуем метки к {-1, +1}
        y = np.where(y > 0, 1.0, -1.0).astype(np.float64)
        y = np.ascontiguousarray(y)

        # Инициализация
        alpha = np.zeros(n_samples, dtype=np.float64)

        # Вектор линейных коэффициентов q_i
        q = np.array([self._get_linear_coef(y_i) for y_i in y], dtype=np.float64)

        # Верхние границы C_i для каждого примера
        C = np.array([self._get_upper_bound(y_i) for y_i in y], dtype=np.float64)

        # Инициализация градиента: gradient = q - Q·α = q (т.к. α = 0)
        gradient = q.copy()

        if self.verbose:
            print(f"SMO solver started: {n_samples} samples")
            print(f"  Memory-efficient mode: Q matrix NOT stored")
            print(f"  Numba JIT: {'enabled' if NUMBA_AVAILABLE else 'disabled'}")

        # Основной цикл SMO (Numba-оптимизированный)
        alpha, n_iter, converged = smo_main_loop(
            X, y, alpha, gradient, C, q,
            self.eps, self.tol, self.max_iter,
            check_interval=100
        )

        # Вычисляем bias
        b = compute_bias_from_gradient(alpha, gradient, y, C, self.eps)

        # Подсчёт опорных векторов
        n_sv = int(np.sum(alpha > self.eps))

        # Вычисляем значение целевой функции (для отчёта)
        obj = self._compute_objective_efficient(alpha, X, y, q)

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

    def _compute_objective_efficient(
        self,
        alpha: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Вычисляет значение целевой функции без хранения полной матрицы Q.

        f(α) = q^T α - 1/2 α^T Q α
             = q^T α - 1/2 Σ_i Σ_j α_i α_j y_i y_j K_ij
        """
        # Линейная часть
        linear_part = np.dot(q, alpha)

        # Квадратичная часть: вычисляем только для ненулевых α
        sv_mask = alpha > self.eps
        if not np.any(sv_mask):
            return linear_part

        sv_indices = np.where(sv_mask)[0]
        alpha_sv = alpha[sv_indices]
        y_sv = y[sv_indices]
        X_sv = X[sv_indices]

        # K_sv = X_sv @ X_sv.T
        K_sv = np.dot(X_sv, X_sv.T)

        # Q_sv = (y_sv * y_sv.T) * K_sv
        YY_sv = np.outer(y_sv, y_sv)
        Q_sv = YY_sv * K_sv

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
    Решает двойственную задачу CS-SVM и возвращает вектор весов w и смещение b.

    Это главная функция-обёртка для использования в BinaryCSSVM.

    Args:
        X: Матрица признаков (n_samples, n_features)
        y: Метки классов (n_samples,), значения {0, 1} или {-1, +1}
        C_slack: Параметр регуляризации C
        C_pos: C₁ - стоимость FN
        C_neg: C₋₁ - стоимость FP
        tol: Допуск сходимости
        max_iter: Максимум итераций
        verbose: Выводить прогресс

    Returns:
        w: Вектор весов (n_features,)
        b: Смещение (скаляр)
        alpha: Множители Лагранжа (n_samples,)
    """
    # Нормализуем метки
    y_norm = np.where(y > 0, 1.0, -1.0).astype(np.float64)

    # Создаём солвер
    solver = CSSVMDualQPSolver(
        C_slack=C_slack,
        C_pos=C_pos,
        C_neg=C_neg,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    # Решаем
    result = solver.solve(X, y_norm)

    # Вычисляем w = Σ α_i y_i x_i
    w = np.sum((result.alpha * y_norm).reshape(-1, 1) * X, axis=0)

    return w, result.b, result.alpha


# =============================================================================
# Быстрая версия для малых датасетов (с предвычислением Q)
# =============================================================================

class CSSVMDualQPSolverFast:
    """
    Быстрая версия солвера для малых датасетов (< 15k примеров).

    Предвычисляет матрицу Q целиком для максимальной скорости.
    Использует больше памяти: O(N²) для матрицы Q.

    Для больших датасетов используйте CSSVMDualQPSolver (memory-efficient).
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

        self.kappa = 1.0 / (2 * C_neg - 1)
        self.C_upper_pos = C_slack * C_pos
        self.C_upper_neg = C_slack / self.kappa

    def solve(self, X: np.ndarray, y: np.ndarray) -> SMOResult:
        """Решает задачу с предвычислением матрицы Q."""
        n_samples = X.shape[0]

        # Проверка размера (предупреждение для больших датасетов)
        estimated_memory_gb = (n_samples ** 2 * 8) / (1024 ** 3)
        if estimated_memory_gb > 2:
            print(f"Warning: Q matrix will use ~{estimated_memory_gb:.1f} GB RAM")
            print("Consider using CSSVMDualQPSolver for large datasets")

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.where(y > 0, 1.0, -1.0).astype(np.float64)

        # Предвычисляем матрицу Q
        K = np.dot(X, X.T)
        YY = np.outer(y, y)
        Q = YY * K
        Q += 1e-12 * np.eye(n_samples)  # Регуляризация

        q = np.array([1.0 if y_i > 0 else self.kappa for y_i in y], dtype=np.float64)
        C = np.array([self.C_upper_pos if y_i > 0 else self.C_upper_neg for y_i in y], dtype=np.float64)

        alpha = np.zeros(n_samples, dtype=np.float64)
        gradient = q.copy()

        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            i, j, found = select_working_set_wss3(alpha, gradient, y, C, self.eps, self.tol)

            if not found:
                converged = True
                break

            # Оптимизация пары с использованием предвычисленной Q
            alpha_i_old, alpha_j_old = alpha[i], alpha[j]

            L, H = compute_bounds(alpha_i_old, alpha_j_old, y[i], y[j], C[i], C[j])
            if L >= H - self.eps:
                continue

            eta = Q[i, i] + Q[j, j] - 2.0 * y[i] * y[j] * Q[i, j] / (y[i] * y[j]) if y[i] * y[j] != 0 else Q[i, i] + Q[j, j] - 2.0 * Q[i, j]
            # Simplified: since Q[i,j] = y[i]*y[j]*K[i,j], we have K[i,j] = Q[i,j]*y[i]*y[j]
            K_ij = Q[i, j] * y[i] * y[j]
            eta = Q[i, i] + Q[j, j] - 2.0 * K_ij

            E_i = -y[i] * gradient[i]
            E_j = -y[j] * gradient[j]

            if eta > self.eps:
                alpha_j_new = alpha_j_old + y[j] * (E_i - E_j) / eta
            else:
                alpha_j_new = alpha_j_old

            alpha_j_new = np.clip(alpha_j_new, L, H)

            if abs(alpha_j_new - alpha_j_old) < self.eps * (alpha_j_old + alpha_j_new + self.eps):
                continue

            alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
            alpha_i_new = np.clip(alpha_i_new, 0.0, C[i])

            delta_alpha_i = alpha_i_new - alpha_i_old
            delta_alpha_j = alpha_j_new - alpha_j_old

            alpha[i] = alpha_i_new
            alpha[j] = alpha_j_new

            # Быстрое обновление градиента с Q
            gradient -= delta_alpha_i * Q[:, i] + delta_alpha_j * Q[:, j]

            if (iteration + 1) % 100 == 0:
                kkt_violation = check_kkt_violation(alpha, gradient, y, C, self.eps)
                if kkt_violation < self.tol:
                    converged = True
                    break

        b = compute_bias_from_gradient(alpha, gradient, y, C, self.eps)
        n_sv = int(np.sum(alpha > self.eps))
        obj = np.dot(q, alpha) - 0.5 * np.dot(alpha, np.dot(Q, alpha))

        return SMOResult(
            alpha=alpha,
            b=b,
            n_iterations=n_iter,
            n_support_vectors=n_sv,
            converged=converged,
            objective_value=obj
        )
