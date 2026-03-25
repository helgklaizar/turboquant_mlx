# TurboQuant Mac — оперативная память проекта

## Стек
- Python 3 (pyproject.toml)
- Apple MLX (`mlx.core`, Metal GPU acceleration)
- `mlx_lm` (API integration and LLM generation)

## Запуск
```bash
# Установка как пакета
pip install -e .

# Поднятие сжатого API Сервера совместимого с OpenAI
python3 scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Запуск тестов генерации на 5 моделях
PYTHONPATH=. python3 scripts/run_needle_test.py
```

## Архитектура
Цель: реализовать алгоритмы QJL и PolarQuant для беспрецедентного сжатия KV Cache LLM до 3 бит без потерь.
- `mlx_core/mlx_turboquant.py` — Metal-оптимизированный конвейер полного квантования (Keys)
- `mlx_core/mlx_polarquant.py` — Быстрое MSE квантование (Values)
- `mlx_core/cache.py` — Динамическая подмена класса `KVCache` для `mlx_lm` со вшитым чанкингом
- `scripts/` — Удобные скрипты для тестов, локального сервера и EXO-кластера

## Ключевые решения
- **Monkey-patch mlx_lm:** Интегрируемся напрямую в функцию `make_prompt_cache` и класс `KVCache`, что гарантирует сжатие во всех модулях Llama/Gemma.
- **Асимметричное сжатие:** Keys сжимаются через сверхточный `TurboQuant`, а Values - через более легкий `PolarQuant`.
- **Heavy Hitter Caching / FP16 Sink:** Первые 128 токенов (System Prompt) остаются без сжатия, спасая Instruction Following при экстремальном бит-рейте.

## Известные проблемы / Tech Debt
- Написание кастомных `.metal` шейдеров отложено (пока питоновский API `mlx.core` справляется за счет lazy-вычислений).
- Архитектуры Qwen/Gemma/Phi-3 менее стабильны с 3-битным кэшем на текущих гиперпараметрах, Llama 3 и 3.2 работают идеально.

## Что делали последним (2026-03-25)
- Проект упакован в продакшен `pip`-пакет. Опубликовано на GitHub.
- Написана интеграционная обертка `apply_turboquant_cache` (dynamic chunking по 64 токена) 
- Добавлен точный счетчик памяти `memory_size` и скрипт интеграции с фреймворком EXO `run_exo_node.py`.
- Протестирован стресс-тест *Needle-in-a-Haystack* на 5 моделях `mlx-community`. Семейство Meta Llama показало абсолютный иммунитет к 3-битному сжатию, сэкономив до 75% ОЗУ.
- Написан благодарственный Issue коллегам из другого репозитория с советами от нашей архитектуры.

## Следующие задачи
1. Сбор обратной связи от комьюнити.
2. Подбор гиперпараметров (Theta_bits, Radius_bits) для стабилизации GQA моделей от Google и Alibaba.
