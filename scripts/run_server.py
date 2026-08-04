#!/usr/bin/env python3
import sys

def main():
    print("🚀 Запуск TurboQuant-MLX OpenAI-совместимого сервера...")
    
    # 1. Сначала встраиваем наш патч до загрузки любых слоев mlx_lm
    from turboquant_mlx.plugins.cache_plugin import apply_turboquant_cache
    apply_turboquant_cache(k_theta_bits=8, v_theta_bits=3, fp16_sink_size=128)
    
    # 2. Передаем управление в стандартный API сервер
    try:
        from mlx_lm.server import main as mlx_server
        mlx_server()
    except ImportError:
        print("Ошибка: mlx-lm не установлен. Запустите pip install mlx-lm")
        sys.exit(1)

if __name__ == "__main__":
    main()
