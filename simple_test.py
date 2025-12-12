"""
Простейший тест новой реализации CS-SVM.
Показывает, как заменить старую реализацию на новую.
"""

import numpy as np
from cssvm_optimized import OptimizedCSSVM

# Генерация простых данных
np.random.seed(42)
X = np.random.randn(100, 20)  # 100 образцов, 20 признаков
y = np.where(np.random.rand(100) > 0.5, 1, -1)  # бинарные метки

print("Simple CS-SVM Test")
print("=" * 30)
print(f"Data shape: {X.shape}")
print(f"Labels: {np.sum(y > 0)} positive, {np.sum(y < 0)} negative")

# Создаем и обучаем модель
print("\nTraining model...")
model = OptimizedCSSVM(
    C_slack=1.0,
    C_pos=3.0,
    C_neg=2.0,
    use_wss=False,  # Для маленьких данных используем полное QP решение
    verbose=True
)

model.fit(X, y)

# Предсказания
print("\nMaking predictions...")
y_pred = model.predict(X)

# Оценка качества
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Support vectors: {len(model.alphas)}/{len(y)}")
print(f"Weight vector norm: {np.linalg.norm(model.w):.4f}")
print(f"Bias: {model.b:.4f}")

print("\n" + "=" * 30)
print("Test completed successfully!")
print("\nUsage in your code:")
print("  from cssvm_optimized import OptimizedCSSVM")
print("  model = OptimizedCSSVM(C_slack=1.0, C_pos=3.0, C_neg=2.0)")
print("  model.fit(X_train, y_train)")
print("  predictions = model.predict(X_test)")