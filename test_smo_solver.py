"""
Тесты для SMO солвера CS-SVM.

Проверяет:
1. Корректность решения на синтетических данных
2. Сравнение с sklearn SVM
3. Memory-efficient режим vs Fast режим
4. Производительность с Numba
5. KKT условия
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


# =============================================================================
# Вспомогательные функции
# =============================================================================

def assert_valid_solution(alpha, y, w, b, X, min_sv=1, min_accuracy=0.7, name=""):
    """Базовые проверки корректности решения."""
    n_sv = np.sum(alpha > 1e-6)
    
    # 1. Должны быть опорные векторы
    assert n_sv >= min_sv, f"{name}: Expected at least {min_sv} SV, got {n_sv}"
    
    # 2. w не должен быть нулевым
    w_norm = np.linalg.norm(w)
    assert w_norm > 1e-6, f"{name}: w is zero vector!"
    
    # 3. Минимальная точность
    y_pred = np.sign(np.dot(X, w) + b)
    acc = accuracy_score(y, y_pred)
    assert acc >= min_accuracy, f"{name}: Accuracy {acc:.4f} < {min_accuracy}"
    
    return n_sv, w_norm, acc


def create_perfect_separable_data(n_samples=100, margin=2.0, seed=42):
    """Создаёт идеально разделимые данные с известным решением."""
    np.random.seed(seed)
    n_half = n_samples // 2
    
    # Идеальное разделение по первой координате
    # Класс +1: x[0] > margin/2
    # Класс -1: x[0] < -margin/2
    X_pos = np.column_stack([
        np.random.uniform(margin/2 + 0.1, margin/2 + 2, n_half),
        np.random.randn(n_half)
    ])
    X_neg = np.column_stack([
        np.random.uniform(-margin/2 - 2, -margin/2 - 0.1, n_half),
        np.random.randn(n_half)
    ])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * n_half + [-1.0] * n_half)
    
    # Перемешиваем
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# =============================================================================
# Тесты внутренней логики через публичные интерфейсы
# =============================================================================

def test_solver_initialization():
    """Тест инициализации солвера и проверка параметров."""
    print("\n" + "="*60)
    print("Test: Solver Initialization")
    print("="*60)
    
    # Проверка корректных параметров
    solver = CSSVMDualQPSolver(C_slack=1.0, C_pos=3.0, C_neg=2.0)
    print(f"  C_slack = {solver.C_slack}")
    print(f"  C_pos = {solver.C_pos}")
    print(f"  C_neg = {solver.C_neg}")
    print(f"  kappa = {solver.kappa:.6f}")
    print(f"  C_upper_pos = {solver.C_upper_pos:.6f}")
    print(f"  C_upper_neg = {solver.C_upper_neg:.6f}")
    
    # Проверка вычислений
    expected_kappa = 1.0 / (2 * 2.0 - 1)  # 1/3
    assert abs(solver.kappa - expected_kappa) < 1e-10, f"Wrong kappa: {solver.kappa}"
    assert solver.C_upper_pos == 3.0, f"Wrong C_upper_pos: {solver.C_upper_pos}"
    assert abs(solver.C_upper_neg - 3.0) < 1e-10, f"Wrong C_upper_neg: {solver.C_upper_neg}"
    
    # Проверка некорректных параметров
    try:
        solver = CSSVMDualQPSolver(C_slack=1.0, C_pos=2.0, C_neg=0.5)
        assert False, "Should raise ValueError for C_neg < 1"
    except ValueError as e:
        print(f"  Correctly rejected C_neg=0.5: {e}")
    
    try:
        solver = CSSVMDualQPSolver(C_slack=1.0, C_pos=1.0, C_neg=2.0)
        assert False, "Should raise ValueError for C_pos < 2*C_neg - 1"
    except ValueError as e:
        print(f"  Correctly rejected C_pos=1.0 with C_neg=2.0: {e}")
    
    print("\n[PASS] Solver initialization test passed!")
    return True


def test_gradient_computation():
    """Тест начального градиента и его обновления через решение простой задачи."""
    print("\n" + "="*60)
    print("Test: Gradient Computation (via simple problem)")
    print("="*60)
    
    # Создаём простейшую задачу: 2 точки
    X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    y = np.array([1.0, -1.0])
    
    solver = CSSVMDualQPSolver(
        C_slack=10.0, C_pos=3.0, C_neg=2.0,
        tol=1e-6, max_iter=100, verbose=True
    )
    
    result = solver.solve(X, y)
    
    print(f"  Alpha: {result.alpha}")
    print(f"  Bias: {result.b:.6f}")
    print(f"  Support vectors: {result.n_support_vectors}")
    print(f"  Iterations: {result.n_iterations}")
    
    # Проверяем, что алгоритм делает хотя бы несколько итераций
    assert result.n_iterations > 1, f"Should take more than 1 iteration, got {result.n_iterations}"
    
    # Проверяем, что есть опорные векторы
    assert result.n_support_vectors >= 1, f"Should have at least 1 SV, got {result.n_support_vectors}"
    
    # Проверяем сходимость
    assert result.converged, "Should converge on simple problem"
    
    print("\n[PASS] Gradient computation test passed!")
    return True


# =============================================================================
# Основные интеграционные тесты
# =============================================================================

def test_basic_linearly_separable():
    """Тест на простых линейно разделимых данных."""
    print("\n" + "="*60)
    print("Test 1: Basic Linearly Separable Data")
    print("="*60)
    
    # Создаём чётко разделимые данные
    X, y = create_perfect_separable_data(n_samples=100, margin=2.0, seed=42)
    
    print("Training CS-SVM SMO solver...")
    start = time.time()
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=1000, verbose=True
    )
    smo_time = time.time() - start
    
    # КРИТИЧЕСКИЕ ПРОВЕРКИ
    n_sv, w_norm, acc_smo = assert_valid_solution(
        alpha, y, w, b, X, 
        min_sv=2,  # Минимум 2 SV для разделимых данных
        min_accuracy=0.95,  # Должна быть высокая точность
        name="Basic Separable"
    )
    
    print(f"  SMO Time: {smo_time:.3f}s")
    print(f"  SMO Accuracy: {acc_smo:.4f}")
    print(f"  Support vectors: {n_sv}")
    print(f"  ||w|| = {w_norm:.6f}")
    print(f"  w = {w}")
    print(f"  b = {b:.6f}")
    
    # Проверяем, что w указывает примерно в направлении [1, 0]
    # (первая координата разделяет классы)
    w_normalized = w / w_norm
    assert abs(w_normalized[0]) > 0.7, f"w should point along x[0], got {w_normalized}"
    
    # Сравнение со sklearn SVC
    print("\nTraining sklearn SVC for comparison...")
    svc = SVC(kernel='linear', C=1.0)
    svc.fit(X, y)
    acc_sklearn = svc.score(X, y)
    print(f"  sklearn Accuracy: {acc_sklearn:.4f}")
    
    # Наша точность должна быть сравнима
    assert acc_smo >= acc_sklearn - 0.1, \
        f"SMO accuracy {acc_smo} much worse than sklearn {acc_sklearn}"
    
    print("\n[PASS] Basic test passed!")
    return acc_smo, acc_sklearn


def test_imbalanced_data():
    """Тест на несбалансированных данных."""
    print("\n" + "="*60)
    print("Test 2: Imbalanced Data (Cost-Sensitive)")
    print("="*60)
    
    np.random.seed(42)
    
    # Создаём несбалансированные, но разделимые данные
    n_pos = 30
    n_neg = 270
    
    # Классы чётко разделены
    X_pos = np.random.randn(n_pos, 2) * 0.5 + np.array([2, 2])
    y_pos = np.ones(n_pos)
    
    X_neg = np.random.randn(n_neg, 2) * 0.5 + np.array([-1, -1])
    y_neg = -np.ones(n_neg)
    
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # CS-SVM с высокой стоимостью FN
    print("Training CS-SVM with C_pos=10, C_neg=1 (penalizing FN more)...")
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=1.0, C_pos=10.0, C_neg=1.0,
        tol=1e-3, max_iter=5000, verbose=True
    )
    
    # КРИТИЧЕСКИЕ ПРОВЕРКИ
    n_sv, w_norm, acc = assert_valid_solution(
        alpha, y, w, b, X,
        min_sv=2,
        min_accuracy=0.8,
        name="Imbalanced Data"
    )
    
    y_pred = np.sign(np.dot(X, w) + b)
    
    tp = np.sum((y == 1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == -1))
    fp = np.sum((y == -1) & (y_pred == 1))
    tn = np.sum((y == -1) & (y_pred == -1))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"  True Positives: {tp}, False Negatives: {fn}")
    print(f"  True Negatives: {tn}, False Positives: {fp}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Support vectors: {n_sv}")
    
    # При C_pos >> C_neg recall должен быть высоким
    assert recall >= 0.7, f"Recall too low for CS-SVM with high C_pos: {recall}"
    
    print("\n[PASS] Imbalanced data test passed!")
    return recall, precision


def test_memory_efficient_vs_fast():
    """Сравнение memory-efficient и fast версий."""
    print("\n" + "="*60)
    print("Test 3: Memory-Efficient vs Fast Solver")
    print("="*60)
    
    X, y = create_perfect_separable_data(n_samples=200, margin=2.0, seed=42)
    
    # Memory-efficient версия
    print("\nMemory-efficient solver:")
    solver_efficient = CSSVMDualQPSolver(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=5000, verbose=True
    )
    start = time.time()
    result_efficient = solver_efficient.solve(X, y)
    time_efficient = time.time() - start
    
    w_eff = np.sum((result_efficient.alpha * y).reshape(-1, 1) * X, axis=0)
    
    # Проверяем корректность efficient версии
    assert result_efficient.n_support_vectors >= 2, \
        f"Efficient solver: expected SV >= 2, got {result_efficient.n_support_vectors}"
    
    print(f"  Time: {time_efficient:.3f}s")
    print(f"  Iterations: {result_efficient.n_iterations}")
    print(f"  Support vectors: {result_efficient.n_support_vectors}")
    
    # Fast версия
    print("\nFast solver (precomputed Q):")
    solver_fast = CSSVMDualQPSolverFast(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=5000, verbose=False
    )
    start = time.time()
    result_fast = solver_fast.solve(X, y)
    time_fast = time.time() - start
    
    # Проверяем корректность fast версии
    assert result_fast.n_support_vectors >= 2, \
        f"Fast solver: expected SV >= 2, got {result_fast.n_support_vectors}"
    
    print(f"  Time: {time_fast:.3f}s")
    print(f"  Iterations: {result_fast.n_iterations}")
    print(f"  Support vectors: {result_fast.n_support_vectors}")
    
    # Сравниваем результаты
    alpha_diff = np.abs(result_efficient.alpha - result_fast.alpha).max()
    
    # Вычисляем предсказания обоих
    w_fast = np.sum((result_fast.alpha * y).reshape(-1, 1) * X, axis=0)
    
    y_pred_eff = np.sign(np.dot(X, w_eff) + result_efficient.b)
    y_pred_fast = np.sign(np.dot(X, w_fast) + result_fast.b)
    
    pred_match = np.mean(y_pred_eff == y_pred_fast)
    
    print(f"\nComparison:")
    print(f"  Max alpha difference: {alpha_diff:.6f}")
    print(f"  Prediction match: {pred_match:.2%}")
    print(f"  Objective (efficient): {result_efficient.objective_value:.6f}")
    print(f"  Objective (fast): {result_fast.objective_value:.6f}")
    
    # Результаты должны совпадать (с некоторым допуском)
    assert pred_match >= 0.90, f"Predictions differ too much: {pred_match:.2%} match"
    
    print("\n[PASS] Memory-efficient vs Fast test passed!")
    return time_efficient, time_fast


def test_scaling_performance():
    """Тест производительности."""
    print("\n" + "="*60)
    print("Test 4: Scaling Performance")
    print("="*60)
    
    print(f"Numba available: {NUMBA_AVAILABLE}")
    
    sizes = [100, 500, 1000]
    results = []
    
    for n_samples in sizes:
        X, y = create_perfect_separable_data(n_samples=n_samples, margin=2.0, seed=42)
        
        start = time.time()
        w, b, alpha = solve_cssvm_dual_qp(
            X, y, C_slack=1.0, C_pos=3.0, C_neg=2.0,
            tol=1e-2, max_iter=500, verbose=False
        )
        elapsed = time.time() - start
        
        n_sv = np.sum(alpha > 1e-6)
        y_pred = np.sign(np.dot(X, w) + b)
        acc = accuracy_score(y, y_pred)
        
        results.append((n_samples, elapsed, n_sv, acc))
        
        print(f"  N={n_samples:5d}: time={elapsed:.3f}s, SV={n_sv}, acc={acc:.3f}")
        
        # Проверяем корректность на каждом размере
        assert n_sv >= 2, f"N={n_samples}: expected SV >= 2, got {n_sv}"
        assert acc >= 0.9, f"N={n_samples}: accuracy too low: {acc}"
    
    print("\n[PASS] Scaling test passed!")
    return results


def test_convergence():
    """Тест сходимости."""
    print("\n" + "="*60)
    print("Test 5: Convergence Test")
    print("="*60)
    
    X, y = create_perfect_separable_data(n_samples=200, margin=1.5, seed=42)
    
    solver = CSSVMDualQPSolver(
        C_slack=1.0, C_pos=3.0, C_neg=2.0,
        tol=1e-3, max_iter=10000, verbose=True
    )
    
    result = solver.solve(X, y)
    
    w = np.sum((result.alpha * y).reshape(-1, 1) * X, axis=0)
    y_pred = np.sign(np.dot(X, w) + result.b)
    acc = accuracy_score(y, y_pred)
    
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Support vectors: {result.n_support_vectors}")
    print(f"  Objective: {result.objective_value:.6f}")
    print(f"  Training accuracy: {acc:.4f}")
    
    # Проверки
    assert result.converged, f"Should converge, but didn't after {result.n_iterations} iterations"
    assert result.n_iterations > 1, f"Should take more than 1 iteration, got {result.n_iterations}"
    assert result.n_support_vectors >= 2, f"Expected SV >= 2, got {result.n_support_vectors}"
    assert acc >= 0.95, f"Accuracy too low: {acc}"
    
    print("\n[PASS] Convergence test passed!")
    return result.converged, acc


def test_kkt_conditions():
    """Проверка KKT условий."""
    print("\n" + "="*60)
    print("Test 6: KKT Conditions Verification")
    print("="*60)
    
    X, y = create_perfect_separable_data(n_samples=100, margin=2.0, seed=42)
    
    C_slack = 1.0
    C_pos = 3.0
    C_neg = 2.0
    kappa = 1.0 / (2 * C_neg - 1)
    
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=C_slack, C_pos=C_pos, C_neg=C_neg,
        tol=1e-4, max_iter=10000, verbose=True
    )
    
    # Предварительная проверка
    n_sv = np.sum(alpha > 1e-6)
    assert n_sv >= 2, f"Expected SV >= 2, got {n_sv}"
    
    # 1. Σ α_i y_i = 0
    sum_alpha_y = np.sum(alpha * y)
    print(f"  Σ α_i y_i = {sum_alpha_y:.6f} (should be ~0)")
    assert abs(sum_alpha_y) < 0.01, f"Equality constraint violated: {sum_alpha_y}"
    
    # 2. 0 ≤ α_i ≤ C_i
    C_upper = np.where(y > 0, C_slack * C_pos, C_slack / kappa)
    violations_lower = np.sum(alpha < -1e-6)
    violations_upper = np.sum(alpha > C_upper + 1e-6)
    print(f"  Box constraint violations: lower={violations_lower}, upper={violations_upper}")
    assert violations_lower == 0, f"Lower bound violated {violations_lower} times"
    assert violations_upper == 0, f"Upper bound violated {violations_upper} times"
    
    # 3. Complementary slackness (с допуском)
    margins = y * (np.dot(X, w) + b)
    
    n_slack_violations = 0
    for i in range(len(alpha)):
        m_i = 1.0 if y[i] > 0 else kappa
        
        if alpha[i] < 1e-6:
            # α_i = 0: y_i(w·x_i + b) >= m_i
            if margins[i] < m_i - 0.1:
                n_slack_violations += 1
        elif alpha[i] > C_upper[i] - 1e-6:
            # α_i = C_i: y_i(w·x_i + b) <= m_i
            if margins[i] > m_i + 0.1:
                n_slack_violations += 1
        else:
            # 0 < α_i < C_i: y_i(w·x_i + b) ≈ m_i
            if abs(margins[i] - m_i) > 0.1:
                n_slack_violations += 1
    
    print(f"  Complementary slackness violations: {n_slack_violations}")
    max_allowed_violations = len(alpha) * 0.15  # Допускаем 15% нарушений
    assert n_slack_violations <= max_allowed_violations, \
        f"Too many slackness violations: {n_slack_violations} > {max_allowed_violations}"
    
    print("\n[PASS] KKT conditions test passed!")
    return sum_alpha_y, n_slack_violations


def test_known_solution():
    """Тест на данных с известным решением."""
    print("\n" + "="*60)
    print("Test 7: Known Solution Test")
    print("="*60)
    
    # Создаём простейший случай: 2 точки, далеко друг от друга
    X = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float64)
    y = np.array([1.0, -1.0])
    
    w, b, alpha = solve_cssvm_dual_qp(
        X, y, C_slack=10.0, C_pos=3.0, C_neg=2.0,
        tol=1e-6, max_iter=1000, verbose=True
    )
    
    print(f"  w = {w}")
    print(f"  b = {b:.6f}")
    print(f"  alpha = {alpha}")
    
    # Проверяем
    assert np.sum(alpha > 1e-6) >= 1, "Should have at least 1 support vector"
    
    # w должен указывать в направлении [1, 0]
    if np.linalg.norm(w) > 1e-6:
        w_norm = w / np.linalg.norm(w)
        assert abs(w_norm[0]) > 0.9, f"w should be along x-axis, got {w_norm}"
    
    # Предсказания должны быть правильными
    y_pred = np.sign(np.dot(X, w) + b)
    assert np.allclose(y_pred, y), f"Wrong predictions: {y_pred} vs {y}"
    
    print("\n[PASS] Known solution test passed!")
    return w, b


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "="*70)
    print("  CS-SVM SMO Solver Test Suite")
    print("  Numba JIT: " + ("ENABLED" if NUMBA_AVAILABLE else "DISABLED"))
    print("="*70)
    
    tests = [
        # Базовые тесты
        ("Solver Initialization", test_solver_initialization),
        ("Gradient Computation", test_gradient_computation),
        ("Known Solution", test_known_solution),
        # Интеграционные тесты
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
        except AssertionError as e:
            results[name] = ("FAIL", str(e))
            failed += 1
            print(f"\n[FAIL] {name}: {e}")
        except Exception as e:
            results[name] = ("ERROR", str(e))
            failed += 1
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
    
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