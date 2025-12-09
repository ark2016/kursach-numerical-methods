"""
Маппинг меток между датасетами ru_go_emotions (28 классов) и ru-izard-emotions (10 классов).

ru_go_emotions: admiration, amusement, anger, annoyance, approval, caring, confusion,
                curiosity, desire, disappointment, disapproval, disgust, embarrassment,
                excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
                pride, realization, relief, remorse, sadness, surprise, neutral

ru-izard-emotions: neutral, joy, sadness, anger, enthusiasm, surprise, disgust, fear, guilt, shame
"""

# Метки датасета seara/ru_go_emotions (28 классов)
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Метки датасета Djacon/ru-izard-emotions (10 классов)
IZARD_EMOTIONS_LABELS = [
    "neutral", "joy", "sadness", "anger", "enthusiasm",
    "surprise", "disgust", "fear", "guilt", "shame"
]

# Маппинг go_emotions -> izard_emotions
# Многие метки из 28 классов группируются в 10 базовых эмоций Изарда
GO_TO_IZARD_MAPPING = {
    # Прямые соответствия
    "neutral": ["neutral"],
    "joy": ["joy"],
    "sadness": ["sadness"],
    "anger": ["anger"],
    "surprise": ["surprise"],
    "disgust": ["disgust"],
    "fear": ["fear"],

    # Группировки по семантической близости
    "amusement": ["joy"],           # веселье -> радость
    "excitement": ["enthusiasm"],    # возбуждение -> энтузиазм
    "optimism": ["enthusiasm"],      # оптимизм -> энтузиазм
    "pride": ["enthusiasm", "joy"],  # гордость -> энтузиазм/радость

    "grief": ["sadness"],            # горе -> грусть
    "disappointment": ["sadness"],   # разочарование -> грусть
    "remorse": ["guilt"],            # раскаяние -> вина
    "embarrassment": ["shame"],      # смущение -> стыд

    "annoyance": ["anger"],          # раздражение -> злость
    "disapproval": ["anger", "disgust"],  # неодобрение -> злость/отвращение

    "nervousness": ["fear"],         # нервозность -> страх

    # Метки без прямого соответствия - маппим на наиболее близкие
    "admiration": ["joy", "enthusiasm"],  # восхищение -> радость/энтузиазм
    "approval": ["joy"],             # одобрение -> радость
    "caring": ["joy"],               # забота -> радость
    "gratitude": ["joy"],            # признательность -> радость
    "love": ["joy"],                 # любовь -> радость
    "relief": ["joy"],               # облегчение -> радость

    "confusion": ["surprise", "fear"],    # непонимание -> удивление/страх
    "curiosity": ["surprise"],       # любопытство -> удивление
    "desire": ["enthusiasm"],        # желание -> энтузиазм
    "realization": ["surprise"],     # осознание -> удивление
}

# Обратный маппинг izard -> go_emotions (какие метки go_emotions включает каждая метка izard)
IZARD_TO_GO_MAPPING = {
    "neutral": ["neutral"],
    "joy": ["joy", "amusement", "approval", "caring", "gratitude", "love", "relief", "admiration", "pride"],
    "sadness": ["sadness", "grief", "disappointment"],
    "anger": ["anger", "annoyance", "disapproval"],
    "enthusiasm": ["excitement", "optimism", "pride", "desire", "admiration"],
    "surprise": ["surprise", "confusion", "curiosity", "realization"],
    "disgust": ["disgust", "disapproval"],
    "fear": ["fear", "nervousness", "confusion"],
    "guilt": ["remorse"],
    "shame": ["embarrassment"],
}


def convert_go_emotions_to_izard(go_labels: list[int], go_label_names: list[str] = None) -> list[int]:
    """
    Конвертирует метки из формата go_emotions в формат izard_emotions.

    Args:
        go_labels: Список индексов меток go_emotions
        go_label_names: Список имён меток go_emotions (если None, используется GO_EMOTIONS_LABELS)

    Returns:
        Список индексов меток izard_emotions
    """
    if go_label_names is None:
        go_label_names = GO_EMOTIONS_LABELS

    izard_labels = set()
    for label_idx in go_labels:
        label_name = go_label_names[label_idx]
        if label_name in GO_TO_IZARD_MAPPING:
            for izard_label in GO_TO_IZARD_MAPPING[label_name]:
                izard_labels.add(IZARD_EMOTIONS_LABELS.index(izard_label))

    return sorted(list(izard_labels))


def convert_go_emotions_binary_to_izard(go_binary: list[int]) -> list[int]:
    """
    Конвертирует бинарный вектор меток go_emotions (28 dim) в izard_emotions (10 dim).

    Args:
        go_binary: Бинарный вектор [0,1,0,1,...] длины 28

    Returns:
        Бинарный вектор [0,1,0,...] длины 10
    """
    izard_binary = [0] * len(IZARD_EMOTIONS_LABELS)

    for i, val in enumerate(go_binary):
        if val == 1:
            label_name = GO_EMOTIONS_LABELS[i]
            if label_name in GO_TO_IZARD_MAPPING:
                for izard_label in GO_TO_IZARD_MAPPING[label_name]:
                    izard_idx = IZARD_EMOTIONS_LABELS.index(izard_label)
                    izard_binary[izard_idx] = 1

    return izard_binary


def get_common_labels() -> list[str]:
    """Возвращает метки, которые присутствуют в обоих датасетах напрямую."""
    return ["neutral", "joy", "sadness", "anger", "surprise", "disgust", "fear"]


def get_mapping_matrix():
    """
    Создаёт матрицу маппинга (28 x 10) для преобразования предсказаний.
    mapping_matrix[i, j] = 1, если go_emotions[i] маппится на izard_emotions[j]
    """
    import numpy as np

    matrix = np.zeros((len(GO_EMOTIONS_LABELS), len(IZARD_EMOTIONS_LABELS)))

    for i, go_label in enumerate(GO_EMOTIONS_LABELS):
        if go_label in GO_TO_IZARD_MAPPING:
            for izard_label in GO_TO_IZARD_MAPPING[go_label]:
                j = IZARD_EMOTIONS_LABELS.index(izard_label)
                matrix[i, j] = 1

    return matrix

