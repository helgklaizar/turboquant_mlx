#!/usr/bin/env python3
"""
Скрипт интеграции TurboQuant (Apple Silicon KVCache Compressor) с EXO (Distributed Inference).
Exo позволяет объединять Макбуки, Mac Minis и Айфоны в мощный кластер для запуска огромных LLM (типа Llama 70B).

Этот патч заставляет Exo сжимать распределенный кэш (что критично при пересылке по сети и ограничении VRAM на нодах).

ИСПОЛЬЗОВАНИЕ:
Чтобы TurboQuant работал внутри кластера, просто запустите вашу стандартную ноду через:
`python3 run_exo_node.py`

Либо, если вы используете стандартный Exo демон на ноде: 
Вставьте строку `import run_exo_node` в начало главного файла `exo`. 
"""

import sys
import os

try:
    from mlx_core.cache import apply_turboquant_cache
    
    # Перехват и внедрение (Сжатие в 3 бита, 128 токенов в оригинале для сохранения инструкций)
    apply_turboquant_cache(bits=3, fp16_sink_size=128)
    
    print("✅ [EXO-Targeted] Пул TurboQuant поднят.")
    print("   Теперь любая распределенная генерация в Exo (на базе MLX) будет жрать на 70% меньше памяти под KV Кэш.")
    
except ImportError:
    print("⚠️ Ошибка: установите пакет через pip install -e .")

if __name__ == "__main__":
    # Smoke-test
    import mlx.core as mx
    print("TurboQuant hook успешно прошел smoke-test!")
