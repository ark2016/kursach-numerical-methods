"""
Тесты для SMO солвера CS-SVM.

Проверяет:
1. Корректность решения на синтетических данных
2. Сравнение с sklearn SVM
3. Memory-efficient режим vs Fast режим
4. Производительность с Numba
"""

import numpy as np
import time
from sklearn.datasets import make_classification, make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.smo_solver import (
    solve_cssvm_dual_qp,
    CSSVMDualQPSolver,
    CSSVMDualQPSolverFast,
    NUMBA_AVAILABLE
)


def test_basic_linearly_separable():
    """Тест на простых линейно разделимых данных."""
    print("\n" + "="*60)
    print("Test 1: Basic Linearly Separable Data")
    print("="*60)

    # Создаём простые линейно разделимые данные
    np.random.seed(42)
    n_samples = 100

    # Класс 1: x > 0
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    y_pos = np.ones(n_samples // 2)

    # Класс -1: x < 0
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    y_neg = -np.ones(n_samples // 2)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])

    # Перемешиваем
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Обучаем наш SMO солвер
    print("Training CS-SVM SMO solver...")
    start = time.time()
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=1000, verbose=True
    )
    smo_time = time.time() - start

    # Предсказания
    y_pred_smo = np.sign(np.dot(X, w) + b)
    acc_smo = accuracy_score(y, y_pred_smo)

    print(f"  SMO Time: {smo_time:.3f}s")
    print(f"  SMO Accuracy: {acc_smo:.4f}")
    print(f"  Support vectors: {np.sum(alpha > 1e-6)}")
    print(f"  w = {w[:5]}..." if len(w) > 5 else f"  w = {w}")
    print(f"  b = {b:.6f}")

    # Сравнение со sklearn SVC
    print("\nTraining sklearn SVC for comparison...")
    start = time.time()
    svc = SVC(kernel='linear', C=1.0)
    svc.fit(X, y)
    sklearn_time = time.time() - start

    y_pred_sklearn = svc.predict(X)
    acc_sklearn = accuracy_score(y, y_pred_sklearn)

    print(f"  sklearn Time: {sklearn_time:.3f}s")
    print(f"  sklearn Accuracy: {acc_sklearn:.4f}")

    assert acc_smo >= 0.9, f"SMO accuracy too low: {acc_smo}"
    print("\n[PASS] Basic test passed!")

    return acc_smo, acc_sklearn


def test_imbalanced_data():
    """Тест на несбалансированных данных (CS-SVM должен справляться лучше)."""
    print("\n" + "="*60)
    print("Test 2: Imbalanced Data (Cost-Sensitive)")
    print("="*60)

    np.random.seed(42)

    # Создаём несбалансированные данные: 90% негативных, 10% позитивных
    n_pos = 50
    n_neg = 450

    X_pos = np.random.randn(n_pos, 2) + np.array([1.5, 1.5])
    y_pos = np.ones(n_pos)

    X_neg = np.random.randn(n_neg, 2) + np.array([-0.5, -0.5])
    y_neg = -np.ones(n_neg)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])

    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # CS-SVM с высокой стоимостью FN (C_pos >> C_neg)
    print("Training CS-SVM with C_pos=10, C_neg=1 (penalizing FN more)...")
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=1.0, C_pos=10.0, C_neg=1.0,
        tol=1e-3, max_iter=5000, verbose=False
    )

    y_pred = np.sign(np.dot(X, w) + b)

    # Считаем метрики для положительного класса
    tp = np.sum((y == 1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))
    fp = np.sum((y == -1) & (y_pred == 1))
    tn = np.sum((y == -1) & (y_pred == -1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"  True Positives: {tp}, False Negatives: {fn}")
    print(f"  True Negatives: {tn}, False Positives: {fp}")
    print(f"  Recall (sensitivity): {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Support vectors: {np.sum(alpha > 1e-6)}")

    # CS-SVM должен иметь высокий recall (мало FN)
    assert recall >= 0.5, f"Recall too low for CS-SVM: {recall}"
    print("\n[PASS] Imbalanced data test passed!")

    return recall, precision


def test_memory_efficient_vs_fast():
    """Сравнение memory-efficient и fast версий солвера."""
    print("\n" + "="*60)
    print("Test 3: Memory-Efficient vs Fast Solver")
    print("="*60)

    np.random.seed(42)
    n_samples = 500
    n_features = 20

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=10, n_redundant=5,
        n_classes=2, random_state=42
    )
    y = np.where(y == 0, -1, 1).astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Memory-efficient версия
    print("\nMemory-efficient solver:")
    solver_efficient = CSSVMDualQPSolver(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=5000, verbose=True
    )
    start = time.time()
    result_efficient = solver_efficient.solve(X, y)
    time_efficient = time.time() - start

    print(f"  Time: {time_efficient:.3f}s")
    print(f"  Iterations: {result_efficient.n_iterations}")
    print(f"  Support vectors: {result_efficient.n_support_vectors}")
    print(f"  Converged: {result_efficient.converged}")

    # Fast версия (с предвычислением Q)
    print("\nFast solver (precomputed Q):")
    solver_fast = CSSVMDualQPSolverFast(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=5000, verbose=False
    )
    start = time.time()
    result_fast = solver_fast.solve(X, y)
    time_fast = time.time() - start

    print(f"  Time: {time_fast:.3f}s")
    print(f"  Iterations: {result_fast.n_iterations}")
    print(f"  Support vectors: {result_fast.n_support_vectors}")
    print(f"  Converged: {result_fast.converged}")

    # Сравниваем результаты
    alpha_diff = np.abs(result_efficient.alpha - result_fast.alpha).max()
    b_diff = abs(result_efficient.b - result_fast.b)

    print(f"\nComparison:")
    print(f"  Max alpha difference: {alpha_diff:.6f}")
    print(f"  Bias difference: {b_diff:.6f}")
    print(f"  Objective (efficient): {result_efficient.objective_value:.6f}")
    print(f"  Objective (fast): {result_fast.objective_value:.6f}")

    # Результаты должны быть близки
    assert alpha_diff < 0.1, f"Alpha values differ too much: {alpha_diff}"
    print("\n[PASS] Memory-efficient vs Fast test passed!")

    return time_efficient, time_fast


def test_scaling_performance():
    """Тест производительности на разных размерах данных."""
    print("\n" + "="*60)
    print("Test 4: Scaling Performance")
    print("="*60)

    print(f"Numba available: {NUMBA_AVAILABLE}")

    sizes = [100, 500, 1000, 2000]
    n_features = 50

    results = []

    for n_samples in sizes:
        np.random.seed(42)
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=20, random_state=42
        )
        y = np.where(y == 0, -1, 1).astype(np.float64)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Ограничиваем итерации для теста
        start = time.time()
        w, b, alpha = solve_cssvm_dual_qp(
            X, y, C_slack=1.0, C_pos=3.0, C_neg=2.0,
            tol=1e-2, max_iter=500, verbose=False
        )
        elapsed = time.time() - start

        n_sv = np.sum(alpha > 1e-6)
        results.append((n_samples, elapsed, n_sv))

        print(f"  N={n_samples:5d}: time={elapsed:.3f}s, SV={n_sv}")

    print("\n[PASS] Scaling test completed!")
    return results


def test_convergence():
    """Тест сходимости на реальных условиях."""
    print("\n" + "="*60)
    print("Test 5: Convergence Test")
    print("="*60)

    np.random.seed(42)

    # Moons dataset - нелинейно разделимый, но линейный SVM должен сходиться
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    y = np.where(y == 0, -1, 1).astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    solver = CSSVMDualQPSolver(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=10000, verbose=True
    )

    result = solver.solve(X, y)

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Support vectors: {result.n_support_vectors}")
    print(f"  Objective: {result.objective_value:.6f}")

    # Проверяем предсказания
    w = np.sum((result.alpha * y).reshape(-1, 1) * X, axis=0)
    y_pred = np.sign(np.dot(X, w) + result.b)
    acc = accuracy_score(y, y_pred)

    print(f"  Training accuracy: {acc:.4f}")

    assert result.converged or result.n_iterations < 10000, "Failed to converge"
    print("\n[PASS] Convergence test passed!")

    return result.converged, acc


def test_kkt_conditions():
    """Проверка KKT условий после оптимизации."""
    print("\n" + "="*60)
    print("Test 6: KKT Conditions Verification")
    print("="*60)

    np.random.seed(42)
    n_samples = 200

    X, y = make_classification(
        n_samples=n_samples, n_features=10,
        n_informative=5, random_state=42
    )
    y = np.where(y == 0, -1, 1).astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    C_slack = 1.0
    C_pos = 3.0
    C_neg = 2.0
    kappa = 1.0 / (2 * C_neg - 1)

    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=C_slack, C_pos=C_pos, C_neg=C_neg,
        tol=1e-4, max_iter=10000, verbose=False
    )

    # Проверяем KKT условия
    # 1. Σ α_i y_i = 0
    sum_alpha_y = np.sum(alpha * y)
    print(f"  Σ α_i y_i = {sum_alpha_y:.6f} (should be ~0)")

    # 2. 0 ≤ α_i ≤ C_i
    C_upper = np.where(y > 0, C_slack * C_pos, C_slack / kappa)
    violations_lower = np.sum(alpha < -1e-6)
    violations_upper = np.sum(alpha > C_upper + 1e-6)
    print(f"  Box constraint violations: lower={violations_lower}, upper={violations_upper}")

    # 3. Complementary slackness
    margins = y * (np.dot(X, w) + b)

    n_violations = 0
    for i in range(len(alpha)):
        m_i = 1.0 if y[i] > 0 else kappa

        if alpha[i] < 1e-6:
            # α_i = 0: y_i(w·x_i + b) >= m_i
            if margins[i] < m_i - 1e-3:
                n_violations += 1
        elif alpha[i] > C_upper[i] - 1e-6:
            # α_i = C_i: y_i(w·x_i + b) <= m_i
            if margins[i] > m_i + 1e-3:
                n_violations += 1
        else:
            # 0 < α_i < C_i: y_i(w·x_i + b) = m_i
            if abs(margins[i] - m_i) > 1e-2:
                n_violations += 1

    print(f"  Complementary slackness violations: {n_violations}")

    assert abs(sum_alpha_y) < 0.1, f"Equality constraint violated: {sum_alpha_y}"
    assert violations_lower == 0, f"Lower bound violated"
    assert violations_upper == 0, f"Upper bound violated"

    print("\n[PASS] KKT conditions test passed!")
    return sum_alpha_y, n_violations


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "="*70)
    print("  CS-SVM SMO Solver Test Suite")
    print("  Numba JIT: " + ("ENABLED" if NUMBA_AVAILABLE else "DISABLED"))
    print("="*70)

    tests = [
        ("Basic Linearly Separable", test_basic_linearly_separable),
        ("Imbalanced Data", test_imbalanced_data),
        ("Memory-Efficient vs Fast", test_memory_efficient_vs_fast),
        ("Scaling Performance", test_scaling_performance),
        ("Convergence", test_convergence),
        ("KKT Conditions", test_kkt_conditions),
    ]

    results = {}
    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = ("PASS", result)
            passed += 1
        except Exception as e:
            results[name] = ("FAIL", str(e))
            failed += 1
            print(f"\n[FAIL] {name}: {e}")

    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    for name, (status, _) in results.items():
        icon = "[✓]" if status == "PASS" else "[✗]"
        print(f"  {icon} {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*70)

    return passed, failed


if __name__ == "__main__":
    run_all_tests()
