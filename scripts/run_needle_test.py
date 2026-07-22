import mlx.core as mx
from mlx_lm import load, generate
from turboquant_mlx.plugins.cache_plugin import apply_turboquant_cache
import traceback

def main():
    models = [
        "mlx-community/Qwen2.5-32B-Instruct-4bit"
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\n=================================================")
        print(f"Тест модели: {model_name}")
        print(f"=================================================")
        
        try:
            # Загрузка
            model, tokenizer = load(model_name)
            
            # Внедрение компрессии с системным префиксом 64 токена
            apply_turboquant_cache(model, k_theta_bits=8, v_theta_bits=3, fp16_sink_size=64)
            
            # Внедрение SSD-стриминга весов (Layer-by-Layer Evaluation)
            import types
            for i, layer in enumerate(model.model.layers if hasattr(model, "model") else model.layers):
                orig_call = layer.__call__
                def stream_layer(self, x, mask=None, cache=None, **kwargs):
                    res = orig_call(x, mask=mask, cache=cache, **kwargs)
                    # Forcing evaluation to clear graph and un-wire memory
                    mx.eval(res)
                    if cache is not None:
                        # cache is either a tuple of caches or dict
                        if isinstance(cache, list) and len(cache) > self.layer_idx:
                            c = cache[self.layer_idx]
                            mx.eval(c.state)
                    mx.metal.clear_cache()
                    return res
                layer.layer_idx = i
                layer.__call__ = types.MethodType(stream_layer, layer)
            
            print("[TurboQuant] Включен SSD-стриминг весов (Layer-by-layer evaluation)!")
            
            # Подготовка Needle and Haystack
            needle = f"\nСекретный пароль для {model_name.split('/')[1]} — 'AppleSiliconM4Turbo'.\n"
            haystack_chunk = "Это обычный текст, заполняющий контекст для проверки памяти. "
            
            haystack = (haystack_chunk * 100) + needle + (haystack_chunk * 50)
            prompt = f"System: Найди точный секретный пароль в следующем тексте и выведи только его.\n\nТекст:\n{haystack}\n\nAssistant: Секретный пароль:"
            
            # Генерация
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=20, 
                verbose=False
            )
            
            if "AppleSiliconM4Turbo" in response.replace(" ", "").replace("'", ""):
                print(f"✅ УСПЕХ: {model_name}\nОтвет модели: {response.strip()}")
                results[model_name] = "УСПЕХ"
            else:
                print(f"❌ ПРОВАЛ: {model_name}\nОтвет модели: {response.strip()}")
                results[model_name] = "ПРОВАЛ"
                
            # Освобождаем память для следующего монстра
            del model
            del tokenizer
            mx.metal.clear_cache()
            
        except Exception as e:
            print(f"⚠️ ОШИБКА ЗАГРУЗКИ: {model_name} ({e})")
            results[model_name] = "ОШИБКА"
            
    print("\n\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ СЖАТИЯ ПОЛЯРНЫМ КЭШЕМ (3-bit) ===")
    for m, r in results.items():
        if r == "УСПЕХ":
            print(f"✅ {m}")
        else:
            print(f"❌ {m}")

if __name__ == "__main__":
    main()
