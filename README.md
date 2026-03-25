# TurboQuant MLX

Сверхстепеничное сжатие (компрессия) KV-кэша для языковых моделей на базе Apple MLX (Metal). Уменьшает потребление VRAM (памяти видеокарты) при генерации кэша до 5 раз.

Алгоритм использует конвейер TurboQuant (PolarQuant + QJL Transform), описанный в работах Google Research, и адаптирован специально для Apple Silicon (M1/M2/M3/M4).

## Как это работает?
Вместо хранения 16-битных чисел (float16) каждого токена, TurboQuant сжимает радиусы и фазы в полярной системе координата (до 2-5 бит) без потерь на хранение scale/zero-point констант. Смещение (bias) скалярного произведения, присущее агрессивному квантованию, нивелируется 1-битным остаточным QJL.

В результате мы получаем **~0.99 корреляции (схожести)** оригинальных скалярных произведений при потреблении в среднем **3-4 бит** на элемент слоя Attention.

Оба алгоритма ускорены при помощи `mlx.core` (отрисованы на Metal).

## Установка

```bash
git clone https://github.com/your-username/turboquant_mlx.git
cd turboquant_mlx
pip install -e .
```

## Быстрый старт (Программный кэш для mlx-lm)

Вы можете бесшовно подключить этот кэш к любой модели из экосистемы `mlx_lm` (например Llama 3 или Gemma 2).

```python
import mlx.core as mx
from mlx_lm import load
from mlx_core.cache import apply_turboquant_cache

# Загружаем вашу любимую модель
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

# Применяем TurboQuant
# bits - битрейт полярного сжатия
# fp16_sink_size - первые N токенов (часто Системный Промпт) не сжимаются для сохранения идеального качества команд!
apply_turboquant_cache(model, bits=3, fp16_sink_size=128)

# Дальше генерируем как обычно
```

## Встроенный OpenAI-совместимый сервер
Вы можете поднять API сервер для своей любимой обертки (Chatbox, Bolt и т.д.):
```bash
python scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```

## Тесты производительности
Мы провели симуляцию размера кэша ключей/значений в LLM на 32K токенов и сравнили `numpy (CPU)` из `core/` и `mlx (Metal)` из `mlx_core/`.
Отклонение (MSE) при 5-битах: `0.0048`. MLX Metal GPU версии в **2.8х** быстрее при компрессии векторных блоков.
RoPE (Rotary Embeddings) поддерживается нативно (вращение происходит до компрессии). Асимметричное сжатие (TurboQuant для Keys, PolarQuant для Values) ускоряет генерацию еще на 10%.

## Благодарности / Acknowledgements
Огромное спасибо репозиторию **[DeadByDawn101/turboquant-mlx](https://github.com/DeadByDawn101/turboquant-mlx)** за архитектурное вдохновение для этого проекта. Идеи по интеграции с кластером EXO, патчингу внутренностей `make_prompt_cache` в `mlx_lm`, а также строгий трекинг памяти в байтах были адаптированы в нашу кодовую базу благодаря их невероятной работе!
