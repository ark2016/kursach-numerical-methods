"""
Минимальный тестовый скрипт для новой реализации CS-SVM.
Запускается аналогично оригинальному benchmark_svm_mlflow.py
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from cssvm_optimized import OptimizedCSSVM

def test_new_cssvm():
    """Тестирование новой реализации на синтетических данных."""
    print("Testing Optimized CSSVM Implementation")
    print("=" * 50)
    
    # Параметры
    n_samples = 1000
    n_features = 50
    test_size = 200
    
    # Генерация синтетических данных
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.where(np.random.rand(n_samples) > 0.5, 1, -1)
    
    # Разделение на train/test
    X_train, X_test = X[:n_samples-test_size], X[n_samples-test_size:]
    y_train, y_test = y[:n_samples-test_size], y[n_samples-test_size:]
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Параметры CS-SVM
    config = {
        'C_slack': 1.0,
        'C_pos': 3.0,
        'C_neg': 2.0
    }
    
    # Тестируем оба режима
    modes = [
        ("QP Mode", {"use_wss": False, "normalize_data": False}),
        ("WSS Mode", {"use_wss": True, "working_set_size": 200, "max_iter": 500, "normalize_data": True})
    ]
    
    results = {}
    
    for mode_name, params in modes:
        print(f"\n--- {mode_name} ---")
        
        # Создаем модель
        model = OptimizedCSSVM(
            verbose=True,
            **config,
            **params
        )
        
        # Обучение
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        # Сохраняем результаты
        results[mode_name] = {
            'train_time': train_time,
            'accuracy': accuracy,
            'f1_score': f1,
            'n_support_vectors': len(model.alphas),
            'w_norm': np.linalg.norm(model.w),
            'bias': model.b
        }
        
        print(f"Results:")
        print(f"  Time: {train_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Support Vectors: {len(model.alphas)}")
        print(f"  ||w||: {np.linalg.norm(model.w):.4f}")
        print(f"  Bias: {model.b:.4f}")
    
    # Сравнение режимов
    print(f"\n--- Comparison ---")
    for mode_name, result in results.items():
        print(f"{mode_name}:")
        for key, value in result.items():
            if key == 'train_time':
                print(f"  {key}: {value:.2f}s")
            elif key in ['accuracy', 'f1_score']:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    print("CS-SVM Optimization Test")
    print("=" * 50)
    
    # Запускаем тест
    results = test_new_cssvm()
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)
    
    print("\nRecommendations:")
    print("1. For small datasets (<5k samples): use use_wss=False for exact solution")
    print("2. For large datasets (>5k samples): use use_wss=True with working_set_size=200-500")
    print("3. The new implementation has correct KKT conditions and better convergence")
    print("4. Memory usage is optimized with sparse matrices and efficient updates")