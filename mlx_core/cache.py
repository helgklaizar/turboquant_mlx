import mlx.core as mx
from .mlx_turboquant import MLXTurboQuant

class TurboQuantKVCache:
    """
    KVCache реализация для Apple MLX.
    Заменяет стандартный mlx_lm.models.cache.KVCache на нашу сжатую версию TurboQuant.
    Она сжимает ключи (и значения по желанию) во время префил-фазы генерации.
    """
    def __init__(self, head_dim: int, n_kv_heads: int, pq_bits: int = 3, qjl_features: int = 2048, fp16_sink_size: int = 128):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        
        self.k_compressor = MLXTurboQuant(feature_dim=head_dim, pq_bits=pq_bits, qjl_features=qjl_features)
        
        from .mlx_polarquant import MLXPolarQuantCompressor
        self.v_compressor = MLXPolarQuantCompressor(feature_dim=head_dim, bits=pq_bits)
        
        self.offset = 0
        self.chunk_size = 64
        self.fp16_sink_size = fp16_sink_size
        
        # Буфер для первых важных токенов без сжатия (Attention Sink)
        self.sink_keys = None
        self.sink_values = None
        
        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []
        
        self.key_buffer = None
        self.value_buffer = None

    def _compress_and_store(self, k: mx.array, v: mx.array):
        b, h, s, d = k.shape
        k_2d = mx.reshape(k, (-1, d))
        compressed_k = self.k_compressor.compress(k_2d)
        self.compressed_keys_chunks.append((compressed_k, (b, h, s, d)))
        
        v_2d = mx.reshape(v, (-1, d))
        compressed_v = self.v_compressor.compress(v_2d)
        self.compressed_values_chunks.append((compressed_v, (b, h, s, d)))

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev_offset = self.offset
        self.offset += keys.shape[2]
        
        # 1. Логика Attention Sink (первые fp16_sink_size токенов оставляем в fp16)
        if prev_offset < self.fp16_sink_size:
            remaining_sink = self.fp16_sink_size - prev_offset
            
            # Забираем токен(ы), которые влезают в Sink
            k_sink_part = keys[:, :, :remaining_sink, :]
            v_sink_part = values[:, :, :remaining_sink, :]
            
            if self.sink_keys is None:
                self.sink_keys = k_sink_part
                self.sink_values = v_sink_part
            else:
                self.sink_keys = mx.concatenate([self.sink_keys, k_sink_part], axis=2)
                self.sink_values = mx.concatenate([self.sink_values, v_sink_part], axis=2)
                
            # Оставшиеся идут в буфер сжатия
            k_compress_part = keys[:, :, remaining_sink:, :]
            v_compress_part = values[:, :, remaining_sink:, :]
        else:
            k_compress_part = keys
            v_compress_part = values
            
        # 2. Логика сжатия для оставшихся
        if k_compress_part.shape[2] > 0:
            if self.key_buffer is None:
                self.key_buffer = k_compress_part
                self.value_buffer = v_compress_part
            else:
                self.key_buffer = mx.concatenate([self.key_buffer, k_compress_part], axis=2)
                self.value_buffer = mx.concatenate([self.value_buffer, v_compress_part], axis=2)
                
            while self.key_buffer is not None and self.key_buffer.shape[2] >= self.chunk_size:
                chunk_k = self.key_buffer[:, :, :self.chunk_size, :]
                chunk_v = self.value_buffer[:, :, :self.chunk_size, :]
                
                self._compress_and_store(chunk_k, chunk_v)
                
                if self.key_buffer.shape[2] > self.chunk_size:
                    self.key_buffer = self.key_buffer[:, :, self.chunk_size:, :]
                    self.value_buffer = self.value_buffer[:, :, self.chunk_size:, :]
                else:
                    self.key_buffer = None
                    self.value_buffer = None
                
        # 3. Декомпрессия старых чанков в единый кэш на отдачу
        full_keys = []
        full_values = []
        
        if self.sink_keys is not None:
            full_keys.append(self.sink_keys)
            full_values.append(self.sink_values)
            
        for comp_k, shape in self.compressed_keys_chunks:
            k_approx_2d = self.k_compressor.decompress(comp_k)
            full_keys.append(mx.reshape(k_approx_2d, shape))
            
        for comp_v, shape in self.compressed_values_chunks:
            v_approx_2d = self.v_compressor.decompress(comp_v)
            full_values.append(mx.reshape(v_approx_2d, shape))
            
        if self.key_buffer is not None:
            full_keys.append(self.key_buffer)
            full_values.append(self.value_buffer)
            
        if not full_keys:
            return keys, values
            
        return mx.concatenate(full_keys, axis=2), mx.concatenate(full_values, axis=2)

    @property
    def state(self):
        # Поддержка mlx_lm API, чтобы внутренние фреймворки не ломались
        k = []
        v = []
        if self.sink_keys is not None:
            k.append(self.sink_keys)
            v.append(self.sink_values)
        if self.key_buffer is not None:
            k.append(self.key_buffer)
            v.append(self.value_buffer)
            
        ret_k = mx.concatenate(k, axis=2) if k else mx.array([])
        ret_v = mx.concatenate(v, axis=2) if v else mx.array([])
        return ret_k, ret_v
        
    @property
    def memory_size(self):
        """Возвращает точное потребление памяти этого кэша (в байтах)."""
        total_bytes = 0
        
        # Буферы и Sink (uncompressed, fp16 = 2 bytes per float)
        for t in [self.sink_keys, self.sink_values, self.key_buffer, self.value_buffer]:
            if t is not None:
                total_bytes += t.size * 2
                
        # Сжатые чанками (асимметричные данные)
        for comp, _ in self.compressed_keys_chunks:
            total_bytes += comp["pq_data"]["r_quant"].size * comp["pq_data"]["r_quant"].dtype.size
            total_bytes += comp["pq_data"]["theta_quant"].size * comp["pq_data"]["theta_quant"].dtype.size
            if "qjl_data" in comp:
                # 1 bit (упакованный в uint8) = 1 byte per 8 logical bits
                total_bytes += comp["qjl_data"].size * 1 
                total_bytes += comp["qjl_norm"].size * 2
                
        for comp, _ in self.compressed_values_chunks:
            total_bytes += comp["pq_data"]["r_quant"].size * comp["pq_data"]["r_quant"].dtype.size
            total_bytes += comp["pq_data"]["theta_quant"].size * comp["pq_data"]["theta_quant"].dtype.size
            
        return total_bytes

def apply_turboquant_cache(model=None, bits: int = 3, qjl_features: int = 2048, fp16_sink_size: int = 128):
    """
    Monkey-patch / Hook для интеграции TurboQuant напрямик в любую LLM (Llama, Gemma) на mlx-lm.
    Подменяет сам генератор KVCache внутри фабрики моделей MLX.
    """
    try:
        import mlx_lm.models.cache as cache_module
    except ImportError:
        print("[TurboQuant] Ошибка: mlx-lm не установлен.")
        return
        
    class PatchedCache(TurboQuantKVCache):
        def __init__(self, head_dim: int, n_kv_heads: int, **kwargs):
            super().__init__(
                head_dim=head_dim, 
                n_kv_heads=n_kv_heads, 
                pq_bits=bits, 
                qjl_features=qjl_features,
                fp16_sink_size=fp16_sink_size
            )

    cache_module.KVCache = PatchedCache
    
    # Также перехватываем функцию фабрики, если модель обходит класс
    if hasattr(cache_module, 'make_prompt_cache'):
        _original_make = cache_module.make_prompt_cache
        def patched_make_prompt_cache(model, max_kv_size=None):
            if hasattr(model, "make_cache"):
                return model.make_cache()
            return [PatchedCache(head_dim=l.head_dim, n_kv_heads=getattr(l, "n_kv_heads", l.n_heads)) for l in model.layers]
        cache_module.make_prompt_cache = patched_make_prompt_cache

    print(f"[TurboQuant] Глобальный патч установлен: mlx_lm.models.cache.KVCache подменен.")
    print(f"[TurboQuant] Настройки: {bits} бит/кэш, Attention Sink (Системный Промпт в FP16): первые {fp16_sink_size} токенов.")
