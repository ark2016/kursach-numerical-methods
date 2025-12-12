"""
Патч для обновления benchmark_svm_mlflow.py с новой оптимизированной реализацией CS-SVM.

Этот скрипт показывает, как интегрировать новую реализацию в существующий код.
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем новую реализацию
from cssvm_optimized import OptimizedCSSVM

# Создаем адаптер для совместимости со старым кодом
class BinaryCSSVM_WSS_Optimized(OptimizedCSSVM):
    """
    Адаптер для совместимости с существующим кодом в benchmark_svm_mlflow.py
    """
    
    def __init__(self, C_slack=1.0, C_pos=3.0, C_neg=2.0,
                 working_set_size=200, max_iter=1000, tol=1e-3,
                 shrinking=False, verbose=False, adaptive_ws=False,
                 patience=10, max_cache_size_mb=100):
        
        # Конвертируем параметры в новый формат
        super().__init__(
            C_slack=C_slack,
            C_pos=C_pos,
            C_neg=C_neg,
            use_wss=True,
            working_set_size=working_set_size,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            normalize_data=True  # Включаем нормализацию для стабильности
        )
        
        # Сохраняем старые параметры для совместимости
        self.shrinking = shrinking
        self.adaptive_ws = adaptive_ws
        self.patience = patience
        self.max_cache_size_mb = max_cache_size_mb

# Создаем адаптер для QP режима
class BinaryCSSVM_QP_Optimized(OptimizedCSSVM):
    """
    Адаптер для совместимости с существующим кодом в benchmark_svm_mlflow.py
    """
    
    def __init__(self, C_slack=1.0, C_pos=3.0, C_neg=2.0):
        
        # Конвертируем параметры в новый формат
        super().__init__(
            C_slack=C_slack,
            C_pos=C_pos,
            C_neg=C_neg,
            use_wss=False,  # Полное QP решение
            verbose=False,
            normalize_data=False  # Без нормализации для совместимости
        )

# Обновляем классы в модуле benchmark_svm_mlflow
import benchmark_svm_mlflow

# Сохраняем оригинальные реализации
BinaryCSSVM_WSS_Original = benchmark_svm_mlflow.BinaryCSSVM_WSS
BinaryCSSVM_QP_Original = benchmark_svm_mlflow.BinaryCSSVM_QP

# Заменяем на новые оптимизированные реализации
benchmark_svm_mlflow.BinaryCSSVM_WSS = BinaryCSSVM_WSS_Optimized
benchmark_svm_mlflow.BinaryCSSVM_QP = BinaryCSSVM_QP_Optimized

print("✅ Successfully patched benchmark_svm_mlflow.py with new CS-SVM implementation!")
print("   The old implementations have been replaced with OptimizedCSSVM")
print("   - BinaryCSSVM_WSS now uses WSS mode with normalization")
print("   - BinaryCSSVM_QP now uses full QP mode")
print("   - Both modes have correct KKT conditions and better convergence")
print("\nOriginal implementations are still available as:")
print("   - BinaryCSSVM_WSS_Original")
print("   - BinaryCSSVM_QP_Original")